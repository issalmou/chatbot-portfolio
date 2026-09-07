"""Embeddings locaux via intfloat/multilingual-e5-base (sentence-transformers,
MIT, 768 dim, 100 langues dont fr/en/ar). Format E5 obligatoire : préfixer
"passage: " pour les documents indexés et "query: " pour les questions —
l'espace vectoriel est asymétrique par construction, mélanger les deux
dégrade fortement le retrieval. Modèle chargé une seule fois par process
(singleton paresseux par model_id)."""

from __future__ import annotations

import threading
from typing import Any

from app.config import settings
from app.embeddings.base import EmbeddingProvider

_models: dict[str, Any] = {}
_lock = threading.Lock()


class E5EmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        model_id: str | None = None,
        device: str | None = None,
        model_version: str | None = None,
    ) -> None:
        self.model_id = model_id or settings.embedding_model_id
        self.model_version = model_version or settings.embedding_model_version
        self._device = device or settings.embedding_device
        self._dimension: int | None = None

    def _get_model(self):
        if self.model_id not in _models:
            with _lock:
                if self.model_id not in _models:
                    from sentence_transformers import SentenceTransformer

                    _models[self.model_id] = SentenceTransformer(self.model_id, device=self._device)
        return _models[self.model_id]

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            model = self._get_model()
            # sentence-transformers >= 6 renomme get_sentence_embedding_dimension()
            # en get_embedding_dimension() ; repli sur l'ancien nom si absent.
            if hasattr(model, "get_embedding_dimension"):
                self._dimension = model.get_embedding_dimension()
            else:
                self._dimension = model.get_sentence_embedding_dimension()
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        prefixed = [f"passage: {t}" for t in texts]
        vectors = model.encode(prefixed, normalize_embeddings=True, convert_to_numpy=True)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        model = self._get_model()
        vector = model.encode([f"query: {text}"], normalize_embeddings=True, convert_to_numpy=True)[0]
        return vector.tolist()


e5_embedding_provider = E5EmbeddingProvider()
