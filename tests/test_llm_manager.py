import time

import pytest

from app.llm.base import ERROR_AUTH, ERROR_TIMEOUT, LLMMessage, LLMProviderError
from app.llm.manager import LLMProviderManager
from tests.conftest import FakeLLMProvider

MSG = [LLMMessage(role="system", content="sys"), LLMMessage(role="user", content="hello")]


def test_first_configured_provider_is_used_when_healthy():
    gemini = FakeLLMProvider("gemini", reply="from gemini")
    mistral = FakeLLMProvider("mistral", reply="from mistral")
    manager = LLMProviderManager([gemini, mistral], cooldown_seconds=60)

    outcome = manager.generate(MSG)

    assert outcome.result.provider == "gemini"
    assert outcome.result.text == "from gemini"
    assert outcome.fallback_used is False
    assert mistral.call_count == 0


def test_missing_key_provider_is_skipped_without_being_called():
    gemini = FakeLLMProvider("gemini", configured=False)
    mistral = FakeLLMProvider("mistral", reply="from mistral")
    manager = LLMProviderManager([gemini, mistral], cooldown_seconds=60)

    outcome = manager.generate(MSG)

    assert outcome.result.provider == "mistral"
    assert gemini.call_count == 0  # jamais appelé : ignoré silencieusement


@pytest.mark.parametrize(
    "error",
    [
        LLMProviderError("invalid key", ERROR_AUTH),
        LLMProviderError("timeout", ERROR_TIMEOUT),
    ],
)
def test_gemini_failure_falls_back_to_mistral(error):
    gemini = FakeLLMProvider("gemini", error=error)
    mistral = FakeLLMProvider("mistral", reply="from mistral")
    manager = LLMProviderManager([gemini, mistral], cooldown_seconds=60)

    outcome = manager.generate(MSG)

    assert outcome.result.provider == "mistral"
    assert outcome.fallback_used is True
    assert outcome.attempted_providers[0]["provider"] == "gemini"


def test_falls_back_through_groq_when_gemini_and_mistral_fail():
    gemini = FakeLLMProvider("gemini", error=LLMProviderError("x", ERROR_AUTH))
    mistral = FakeLLMProvider("mistral", error=LLMProviderError("x", ERROR_TIMEOUT))
    groq = FakeLLMProvider("groq", reply="from groq")
    manager = LLMProviderManager([gemini, mistral, groq], cooldown_seconds=60)

    outcome = manager.generate(MSG)

    assert outcome.result.provider == "groq"


def test_falls_back_to_openai_as_last_resort():
    gemini = FakeLLMProvider("gemini", error=LLMProviderError("x", ERROR_AUTH))
    mistral = FakeLLMProvider("mistral", error=LLMProviderError("x", ERROR_AUTH))
    groq = FakeLLMProvider("groq", error=LLMProviderError("x", ERROR_AUTH))
    openai = FakeLLMProvider("openai", reply="from openai")
    manager = LLMProviderManager([gemini, mistral, groq, openai], cooldown_seconds=60)

    outcome = manager.generate(MSG)

    assert outcome.result.provider == "openai"


def test_all_providers_failing_raises_clean_error():
    providers = [FakeLLMProvider(n, error=LLMProviderError("x", ERROR_AUTH)) for n in ["gemini", "mistral", "groq", "openai"]]
    manager = LLMProviderManager(providers, cooldown_seconds=60)

    with pytest.raises(LLMProviderError):
        manager.generate(MSG)


def test_provider_in_cooldown_is_skipped_then_retried_after_expiry():
    gemini = FakeLLMProvider("gemini", error=LLMProviderError("x", ERROR_AUTH))
    mistral = FakeLLMProvider("mistral", reply="from mistral")
    manager = LLMProviderManager([gemini, mistral], cooldown_seconds=0.1)

    first = manager.generate(MSG)
    assert first.result.provider == "mistral"
    assert gemini.call_count == 1

    # Gemini est en cooldown : on ne le rappelle pas immédiatement.
    second = manager.generate(MSG)
    assert second.result.provider == "mistral"
    assert gemini.call_count == 1  # toujours 1, pas de nouvel appel pendant le cooldown

    time.sleep(0.15)
    gemini._error = None  # simule la récupération du provider
    gemini._reply = "from gemini again"

    third = manager.generate(MSG)
    assert third.result.provider == "gemini"  # redevient prioritaire automatiquement
    assert third.fallback_used is False


def test_status_never_exposes_api_keys():
    gemini = FakeLLMProvider("gemini")
    manager = LLMProviderManager([gemini], cooldown_seconds=60)
    status = manager.status()

    assert "api_key" not in str(status).lower()
    for info in status.values():
        assert set(info.keys()) == {"configured", "available", "cooldown_remaining_s", "last_error_type"}
