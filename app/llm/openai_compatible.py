"""Provider générique pour toute API compatible avec le protocole OpenAI
(POST /chat/completions). Utilisé pour Mistral, Groq et OpenAI : seuls
l'api_key, la base_url et le modèle changent entre ces trois fournisseurs,
l'implémentation HTTP est strictement partagée.
"""

from __future__ import annotations

import time

import httpx

from app.llm.base import (
    ERROR_AUTH,
    ERROR_INVALID_RESPONSE,
    ERROR_MISSING_KEY,
    ERROR_MODEL_UNAVAILABLE,
    ERROR_NETWORK,
    ERROR_RATE_LIMIT,
    ERROR_SERVER,
    ERROR_TIMEOUT,
    ERROR_UNKNOWN,
    LLMMessage,
    LLMProvider,
    LLMProviderError,
    LLMResult,
)


class OpenAICompatibleProvider(LLMProvider):
    def __init__(
        self,
        provider_name: str,
        api_key: str | None,
        base_url: str | None,
        model: str,
        timeout_seconds: float,
    ) -> None:
        self.name = provider_name
        self._api_key = api_key
        self._base_url = base_url.rstrip("/") if base_url else None
        self.model = model
        self._timeout = timeout_seconds

    def is_configured(self) -> bool:
        return bool(self._api_key and self._base_url and self.model)

    def generate(self, messages: list[LLMMessage]) -> LLMResult:
        if not self.is_configured():
            raise LLMProviderError(f"{self.name} non configuré (clé/API manquante).", ERROR_MISSING_KEY)

        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": 0.2,
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}

        t0 = time.perf_counter()
        try:
            response = httpx.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self._timeout,
            )
        except httpx.TimeoutException as exc:
            raise LLMProviderError(f"{self.name}: timeout après {self._timeout}s.", ERROR_TIMEOUT) from exc
        except httpx.RequestError as exc:
            raise LLMProviderError(f"{self.name}: erreur réseau ({exc}).", ERROR_NETWORK) from exc
        latency_ms = (time.perf_counter() - t0) * 1000

        if response.status_code in (401, 403):
            raise LLMProviderError(f"{self.name}: authentification refusée (clé API invalide).", ERROR_AUTH)
        if response.status_code == 404:
            raise LLMProviderError(f"{self.name}: modèle '{self.model}' indisponible.", ERROR_MODEL_UNAVAILABLE)
        if response.status_code == 429:
            raise LLMProviderError(f"{self.name}: quota/rate limit dépassé.", ERROR_RATE_LIMIT)
        if response.status_code >= 500:
            raise LLMProviderError(f"{self.name}: erreur serveur ({response.status_code}).", ERROR_SERVER)
        if response.status_code != 200:
            raise LLMProviderError(
                f"{self.name}: réponse inattendue ({response.status_code}).", ERROR_UNKNOWN
            )

        try:
            data = response.json()
            text = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError(f"{self.name}: format de réponse invalide ({exc}).", ERROR_INVALID_RESPONSE) from exc

        if not text:
            raise LLMProviderError(f"{self.name}: réponse vide.", ERROR_INVALID_RESPONSE)

        return LLMResult(text=text, provider=self.name, model=self.model, latency_ms=latency_ms)
