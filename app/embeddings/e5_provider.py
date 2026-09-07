"""Embeddings via l'API Hugging Face Inference Providers (provider hf-inference),
pour le modèle intfloat/multilingual-e5-base — vérifié en direct : vecteurs
numériquement identiques (cosinus ~1.0, écart max ~1e-7) à une inférence
locale du même modèle. Remplace un chargement local via sentence-transformers/
torch (>1 Go de RAM à eux seuls, incompatible avec un conteneur à mémoire
limitée) par un simple appel HTTP, sans migration des données déjà indexées.

Format E5 obligatoire : préfixer "passage: " pour les documents indexés et
"query: " pour les questions — l'espace vectoriel est asymétrique par
construction, mélanger les deux dégrade fortement le retrieval.
"""

from __future__ import annotations

import httpx

from app.config import settings
from app.embeddings.base import EmbeddingProvider

_HF_ROUTER_URL = "https://router.huggingface.co/hf-inference/models/{model_id}/pipeline/feature-extraction"


class EmbeddingProviderError(Exception):
    """L'API d'embedding distante est indisponible, mal configurée, ou a
    retourné une réponse invalide."""


class E5EmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        model_id: str | None = None,
        api_token: str | None = None,
        model_version: str | None = None,
        timeout_seconds: float | None = None,
        dimension: int | None = None,
    ) -> None:
        self.model_id = model_id or settings.embedding_model_id
        self.model_version = model_version or settings.embedding_model_version
        self._token = api_token if api_token is not None else settings.hf_token
        self._timeout = timeout_seconds or settings.embedding_request_timeout_seconds
        self._dimension = dimension or settings.embedding_dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _embed(self, prefixed_texts: list[str]) -> list[list[float]]:
        if not self._token:
            raise EmbeddingProviderError("HF_TOKEN non configuré : impossible d'appeler l'API d'embedding.")

        url = _HF_ROUTER_URL.format(model_id=self.model_id)
        headers = {"Authorization": f"Bearer {self._token}"}
        # `inputs` toujours envoyé comme une LISTE (même pour un seul texte) :
        # l'API renvoie alors systématiquement une liste de vecteurs, jamais un
        # vecteur plat — vérifié en direct, forme de réponse sinon ambiguë.
        payload = {"inputs": prefixed_texts, "normalize": True}

        try:
            response = httpx.post(url, json=payload, headers=headers, timeout=self._timeout)
        except httpx.TimeoutException as exc:
            raise EmbeddingProviderError(f"Timeout après {self._timeout}s en appelant l'API d'embedding.") from exc
        except httpx.RequestError as exc:
            raise EmbeddingProviderError(f"Erreur réseau en appelant l'API d'embedding ({exc}).") from exc

        if response.status_code != 200:
            raise EmbeddingProviderError(
                f"API d'embedding : réponse inattendue ({response.status_code}): {response.text[:200]}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise EmbeddingProviderError(f"API d'embedding : réponse JSON invalide ({exc}).") from exc

        if not isinstance(data, list) or len(data) != len(prefixed_texts):
            raise EmbeddingProviderError(
                f"API d'embedding : forme de réponse inattendue (attendu {len(prefixed_texts)} vecteurs)."
            )
        return data

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed([f"passage: {t}" for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._embed([f"query: {text}"])[0]


e5_embedding_provider = E5EmbeddingProvider()
