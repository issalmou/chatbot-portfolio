"""Orchestration du fallback multi-provider avec circuit breaker en mémoire.

Ordre de priorité strict : Gemini -> Mistral -> Groq -> OpenAI. Un provider
non configuré est ignoré silencieusement ; un provider qui échoue est mis en
cooldown (`llm_circuit_breaker_cooldown_seconds`) et sauté sans appel réseau
jusqu'à expiration, puis retenté automatiquement — son rang ne change jamais.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from app.config import settings
from app.llm.base import ERROR_UNKNOWN, LLMMessage, LLMProvider, LLMProviderError, LLMResult
from app.llm.gemini import GeminiProvider
from app.llm.openai_compatible import OpenAICompatibleProvider


@dataclass
class _BreakerState:
    open_until: float = 0.0
    last_error_type: str | None = None


@dataclass
class GenerationOutcome:
    result: LLMResult
    fallback_used: bool
    attempted_providers: list[dict] = field(default_factory=list)


class LLMProviderManager:
    def __init__(self, providers: list[LLMProvider], cooldown_seconds: float) -> None:
        self._providers = providers  # ordre = ordre de priorité strict, jamais réordonné
        self._cooldown = cooldown_seconds
        self._breakers: dict[str, _BreakerState] = {p.name: _BreakerState() for p in providers}

    def _is_open(self, name: str) -> bool:
        return time.monotonic() < self._breakers[name].open_until

    def _trip(self, name: str, error_type: str) -> None:
        self._breakers[name].open_until = time.monotonic() + self._cooldown
        self._breakers[name].last_error_type = error_type

    def _reset(self, name: str) -> None:
        self._breakers[name].open_until = 0.0
        self._breakers[name].last_error_type = None

    def current_provider(self) -> tuple[str, str] | None:
        """Provider qui serait tenté en premier, sans appel réseau — sert à
        construire la clé de cache de réponse avant l'appel réel (voir
        rag/retrieval.py, qui reclé sous le provider ayant réellement répondu)."""
        for provider in self._providers:
            if provider.is_configured() and not self._is_open(provider.name):
                return provider.name, getattr(provider, "model", "unknown")
        return None

    def status(self) -> dict:
        now = time.monotonic()
        return {
            p.name: {
                "configured": p.is_configured(),
                "available": p.is_configured() and now >= self._breakers[p.name].open_until,
                "cooldown_remaining_s": max(0, round(self._breakers[p.name].open_until - now, 1)),
                "last_error_type": self._breakers[p.name].last_error_type,
            }
            for p in self._providers
        }

    def generate(self, messages: list[LLMMessage]) -> GenerationOutcome:
        attempted: list[dict] = []

        for provider in self._providers:
            if not provider.is_configured():
                continue
            if self._is_open(provider.name):
                attempted.append({"provider": provider.name, "error_type": "circuit_open"})
                continue

            try:
                result = provider.generate(messages)
            except LLMProviderError as exc:
                self._trip(provider.name, exc.error_type)
                attempted.append({"provider": provider.name, "error_type": exc.error_type})
                continue
            except Exception as exc:  # garde-fou : un provider ne doit jamais faire planter le manager
                self._trip(provider.name, ERROR_UNKNOWN)
                attempted.append({"provider": provider.name, "error_type": ERROR_UNKNOWN, "detail": str(exc)})
                continue

            self._reset(provider.name)
            return GenerationOutcome(result=result, fallback_used=bool(attempted), attempted_providers=attempted)

        raise LLMProviderError(
            f"Tous les providers LLM configurés sont indisponibles (tentatives : {attempted}).",
            ERROR_UNKNOWN,
        )


def build_default_manager() -> LLMProviderManager:
    timeout = settings.llm_request_timeout_seconds
    providers: list[LLMProvider] = [
        GeminiProvider(api_key=settings.gemini_api_key, model=settings.gemini_model, timeout_seconds=timeout),
        OpenAICompatibleProvider(
            provider_name="mistral",
            api_key=settings.mistral_api_key,
            base_url=settings.mistral_base_url,
            model=settings.mistral_model,
            timeout_seconds=timeout,
        ),
        OpenAICompatibleProvider(
            provider_name="groq",
            api_key=settings.groq_api_key,
            base_url=settings.groq_base_url,
            model=settings.groq_model,
            timeout_seconds=timeout,
        ),
        OpenAICompatibleProvider(
            provider_name="openai",
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.openai_model,
            timeout_seconds=timeout,
        ),
    ]
    return LLMProviderManager(providers, cooldown_seconds=settings.llm_circuit_breaker_cooldown_seconds)


llm_manager = build_default_manager()
