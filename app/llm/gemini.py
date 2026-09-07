"""Provider Gemini — API native (pas de protocole OpenAI-compatible)."""

from __future__ import annotations

import time

from google import genai
from google.genai import types

from app.llm.base import (
    ERROR_AUTH,
    ERROR_INVALID_RESPONSE,
    ERROR_MISSING_KEY,
    ERROR_MODEL_UNAVAILABLE,
    ERROR_RATE_LIMIT,
    ERROR_SERVER,
    ERROR_TIMEOUT,
    ERROR_UNKNOWN,
    LLMMessage,
    LLMProvider,
    LLMProviderError,
    LLMResult,
)


def _classify_gemini_error(exc: Exception) -> str:
    message = str(exc).lower()
    if "api key" in message or "unauthenticated" in message or "401" in message or "permission" in message:
        return ERROR_AUTH
    if "429" in message or "quota" in message or "rate limit" in message or "resource_exhausted" in message:
        return ERROR_RATE_LIMIT
    if "timeout" in message or "deadline" in message:
        return ERROR_TIMEOUT
    if "not found" in message or "404" in message:
        return ERROR_MODEL_UNAVAILABLE
    if "500" in message or "503" in message or "internal" in message or "unavailable" in message:
        return ERROR_SERVER
    return ERROR_UNKNOWN


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, api_key: str | None, model: str, timeout_seconds: float) -> None:
        self.model = model
        self._client: genai.Client | None = None
        if api_key:
            try:
                self._client = genai.Client(
                    api_key=api_key,
                    http_options=types.HttpOptions(timeout=int(timeout_seconds * 1000)),
                )
            except Exception:
                self._client = None

    def is_configured(self) -> bool:
        return self._client is not None

    def generate(self, messages: list[LLMMessage]) -> LLMResult:
        if self._client is None:
            raise LLMProviderError("Gemini non configuré (clé API manquante).", ERROR_MISSING_KEY)

        system = next((m.content for m in messages if m.role == "system"), None)
        user_content = "\n\n".join(m.content for m in messages if m.role == "user")

        t0 = time.perf_counter()
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=user_content,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=0.2,
                    # "thinking_level" (bas) plutôt que l'ancien "thinking_budget=0" :
                    # vérifié en direct, les modèles Gemini 3.x rejettent
                    # thinking_budget avec une 400 INVALID_ARGUMENT, alors que
                    # thinking_level est accepté aussi bien par les modèles
                    # Gemini 2.5 que 3.x — c'est le paramètre à privilégier.
                    thinking_config=types.ThinkingConfig(thinking_level="low"),
                ),
            )
        except Exception as exc:
            raise LLMProviderError(f"Gemini: {exc}", _classify_gemini_error(exc)) from exc
        latency_ms = (time.perf_counter() - t0) * 1000

        if not response.text:
            raise LLMProviderError("Gemini: réponse vide.", ERROR_INVALID_RESPONSE)

        return LLMResult(text=response.text, provider=self.name, model=self.model, latency_ms=latency_ms)
