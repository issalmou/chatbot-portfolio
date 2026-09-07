"""Interface commune à tous les fournisseurs LLM de génération.

Le reste de l'application (rag/retrieval.py) ne connaît que cette interface :
il ne sait jamais si la réponse vient de Gemini, Mistral, Groq ou OpenAI.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

# Types d'erreur utilisés uniquement pour l'observabilité (logs/health) : le
# comportement de fallback est le même pour tous (on passe au provider
# suivant), donc on ne branche aucune logique conditionnelle dessus.
ERROR_MISSING_KEY = "missing_api_key"
ERROR_AUTH = "authentication_error"
ERROR_RATE_LIMIT = "rate_limit"
ERROR_TIMEOUT = "timeout"
ERROR_NETWORK = "network_error"
ERROR_SERVER = "server_error"
ERROR_MODEL_UNAVAILABLE = "model_unavailable"
ERROR_INVALID_RESPONSE = "invalid_response"
ERROR_UNKNOWN = "unknown_error"


class LLMProviderError(Exception):
    def __init__(self, message: str, error_type: str = ERROR_UNKNOWN) -> None:
        super().__init__(message)
        self.error_type = error_type


@dataclass(frozen=True)
class LLMMessage:
    role: str  # "system" | "user"
    content: str


@dataclass(frozen=True)
class LLMResult:
    text: str
    provider: str
    model: str
    latency_ms: float


class LLMProvider(ABC):
    name: str

    @abstractmethod
    def is_configured(self) -> bool:
        """False si la clé API (ou autre paramètre requis) est absente —
        le provider est alors ignoré silencieusement par le manager, sans
        jamais être appelé ni compter comme une erreur."""

    @abstractmethod
    def generate(self, messages: list[LLMMessage]) -> LLMResult:
        """Doit lever LLMProviderError (jamais une exception brute non
        typée si possible) en cas d'échec, pour que le manager puisse
        logger un error_type exploitable."""
