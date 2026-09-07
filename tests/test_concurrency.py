"""Tests d'isolation multi-utilisateur / concurrence (audit pré-production).

Ce backend est STATELESS par conception : aucune conversation n'est jamais
stockée côté serveur (voir app/rag/memory.py — le client renvoie l'historique
à chaque requête). Les trois SEULS états partagés entre requêtes sont les
singletons de cache (app/cache/caches.py) : ces tests vérifient qu'ils ne
peuvent jamais mélanger le contexte de deux utilisateurs, y compris sous
charge concurrente réelle (threads), et même pour deux conversations dont le
texte de la DERNIÈRE question est identique mot pour mot mais dont le
référent résolu diffère (le cas explicitement à risque).
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.cache.caches import embedding_cache, response_cache, retrieval_cache
from app.ingestion.pipeline import ingest_portfolio
from app.llm.base import LLMResult
from app.llm.manager import LLMProviderManager
from app.rag.memory import Turn
from app.rag.retrieval import answer_question
from tests.conftest import FakeChromaStore, FakeEmbedder, FakeLLMProvider, SAMPLE_TRANSLATIONS_JS
from tests.test_retrieval import JS_FIVE_PROJECTS


@pytest.fixture(autouse=True)
def _clear_shared_caches():
    # Ces caches sont de VRAIS singletons partagés au niveau du module (voir
    # app/cache/caches.py) : ce sont eux qu'on veut tester sous concurrence
    # réelle ici, pas des doublures isolées par test.
    embedding_cache.clear()
    retrieval_cache.clear()
    response_cache.clear()
    yield
    embedding_cache.clear()
    retrieval_cache.clear()
    response_cache.clear()


class RevealingProvider(FakeLLMProvider):
    """Répond en révélant quel CONTEXTE (quel projet) il a effectivement reçu,
    pour pouvoir prouver après coup qu'aucun thread n'a reçu la réponse
    destinée à un autre thread — un vrai LLM ferait la même chose
    naturellement puisqu'il lit le CONTEXTE de sa propre requête HTTP, jamais
    celui d'une autre (aucun état partagé entre deux appels `generate`)."""

    _PROJECT_NAMES = ("AGEP", "ESTICAR", "WIREDWAVE", "SPEECHLY", "DESCRIPTOAI")

    def __init__(self, name: str = "gemini"):
        super().__init__(name, reply="")

    def generate(self, messages):
        self.call_count += 1
        user_msg = next(m.content for m in messages if m.role == "user")
        context_part = user_msg.split("QUESTION")[0].upper()
        for candidate in self._PROJECT_NAMES:
            if candidate in context_part:
                return LLMResult(text=f"ANSWER_ABOUT_{candidate}", provider=self.name, model=self.model, latency_ms=1.0)
        return LLMResult(text="ANSWER_GENERIC_NO_PROJECT", provider=self.name, model=self.model, latency_ms=1.0)


@pytest.fixture()
def five_projects_store():
    store = FakeChromaStore()
    embedder = FakeEmbedder()
    ingest_portfolio(JS_FIVE_PROJECTS.encode("utf-8"), client=embedder, store=store)
    return store, embedder


def _run_conversation(store, embedder, conversation, follow_up: str, barrier: threading.Barrier | None = None):
    """Simule un utilisateur : chaque thread a son PROPRE manager/provider
    (comme deux vraies requêtes HTTP concurrentes, chacune construisant ses
    propres objets Python locaux — voir main.py::chat_endpoint), mais tous
    partagent les MÊMES caches globaux (embedding_cache/retrieval_cache/
    response_cache), le point critique à vérifier."""
    provider = RevealingProvider()
    manager = LLMProviderManager([provider], cooldown_seconds=60)
    if barrier is not None:
        barrier.wait()  # maximise le recouvrement temporel réel entre threads
    result = answer_question(follow_up, embedder=embedder, store=store, manager=manager, conversation=conversation)
    return result


# --- Scénario critique explicite : follow-up textuellement identique,
# référent résolu différent (AGEP vs un autre projet) ---

def test_identical_follow_up_text_never_crosses_between_two_users(five_projects_store):
    store, embedder = five_projects_store
    follow_up = "What technologies does it use?"

    conv_a = [
        Turn(role="user", content="Tell me about the AGEP project."),
        Turn(role="assistant", content="AGEP is a management platform."),
    ]
    conv_b = [
        Turn(role="user", content="Tell me about the WiredWave project."),
        Turn(role="assistant", content="WiredWave is an e-commerce platform."),
    ]

    barrier = threading.Barrier(2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(_run_conversation, store, embedder, conv_a, follow_up, barrier)
        future_b = pool.submit(_run_conversation, store, embedder, conv_b, follow_up, barrier)
        result_a = future_a.result(timeout=10)
        result_b = future_b.result(timeout=10)

    assert result_a.response == "ANSWER_ABOUT_AGEP"
    assert result_b.response == "ANSWER_ABOUT_WIREDWAVE"
    assert result_a.response != result_b.response


def test_identical_follow_up_repeated_concurrently_many_times_never_corrupts_cache(five_projects_store):
    # Répété plusieurs fois avec des paires (thread A, thread B) fraîches à
    # chaque itération, pour maximiser la chance d'observer une éventuelle
    # course critique sur les caches partagés (OrderedDict non thread-safe
    # nativement — voir app/cache/local_cache.py, corrigé par un verrou).
    store, embedder = five_projects_store
    follow_up = "What technologies does it use?"
    pairs = [
        (
            [Turn(role="user", content="Tell me about the AGEP project."), Turn(role="assistant", content="AGEP is a platform.")],
            "ANSWER_ABOUT_AGEP",
        ),
        (
            [Turn(role="user", content="Tell me about the EstiCar project."), Turn(role="assistant", content="EstiCar predicts prices.")],
            "ANSWER_ABOUT_ESTICAR",
        ),
    ]

    for _ in range(20):
        embedding_cache.clear()
        retrieval_cache.clear()
        response_cache.clear()
        barrier = threading.Barrier(2)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(_run_conversation, store, embedder, conv, follow_up, barrier)
                for conv, _ in pairs
            ]
            results = [f.result(timeout=10) for f in futures]
        for result, (_, expected) in zip(results, pairs):
            assert result.response == expected


# --- 5 "utilisateurs" simultanés, scénarios variés (langue, hors-sujet, projets) ---

def test_five_simultaneous_users_never_mix_context(five_projects_store):
    store, embedder = five_projects_store

    scenarios = {
        "user_A": (
            [Turn(role="user", content="Tell me about the AGEP project."), Turn(role="assistant", content="AGEP is a platform.")],
            "What technologies does it use?",
            "en",
        ),
        "user_B": (
            [Turn(role="user", content="Tell me about the WiredWave project."), Turn(role="assistant", content="WiredWave is a platform.")],
            "What technologies does it use?",
            "en",
        ),
        "user_C": (None, "What is the capital of Japan?", "en"),
        "user_D": (None, "Quels sont ses projets ?", "fr"),
        "user_E": (None, "ما هي مهاراته؟", "ar"),
    }

    barrier = threading.Barrier(len(scenarios))
    results = {}
    with ThreadPoolExecutor(max_workers=len(scenarios)) as pool:
        futures = {
            name: pool.submit(_run_conversation, store, embedder, conv, query, barrier)
            for name, (conv, query, _lang) in scenarios.items()
        }
        for name, future in futures.items():
            results[name] = future.result(timeout=10)

    assert results["user_A"].response == "ANSWER_ABOUT_AGEP"
    assert results["user_B"].response == "ANSWER_ABOUT_WIREDWAVE"
    assert results["user_A"].response != results["user_B"].response

    assert results["user_C"].metrics["scope"] == "out_of_scope"
    assert "Japan" not in results["user_C"].response

    for name, (_conv, _query, expected_lang) in scenarios.items():
        assert results[name].lang == expected_lang, f"{name}: expected lang {expected_lang}, got {results[name].lang}"

    # user_A/B ne doivent jamais avoir reçu la langue ou le contenu d'un autre.
    assert results["user_D"].lang == "fr"
    assert results["user_E"].lang == "ar"


# --- Circuit breaker / fallback partagé : ne mélange jamais le CONTENU,
# même si l'état de santé provider (global, par conception) est partagé ---

def test_shared_circuit_breaker_does_not_leak_content_between_users(five_projects_store):
    from app.llm.base import LLMProviderError, ERROR_RATE_LIMIT

    store, embedder = five_projects_store

    class FailingThenRevealing(RevealingProvider):
        def __init__(self, name, fail_first: bool):
            super().__init__(name)
            self._fail_first = fail_first
            self._failed_once = False

        def generate(self, messages):
            if self._fail_first and not self._failed_once:
                self._failed_once = True
                raise LLMProviderError("simulated rate limit", ERROR_RATE_LIMIT)
            return super().generate(messages)

    # Un SEUL manager partagé (comme le llm_manager global réel) : le
    # provider "gemini" échoue une fois côté utilisateur A puis un fallback
    # "groq" prend le relais ; l'utilisateur B doit continuer à recevoir SA
    # PROPRE réponse, jamais celle d'A, quel que soit l'état du circuit breaker.
    gemini = FailingThenRevealing("gemini", fail_first=True)
    groq = RevealingProvider("groq")
    shared_manager = LLMProviderManager([gemini, groq], cooldown_seconds=60)

    conv_a = [Turn(role="user", content="Tell me about the AGEP project."), Turn(role="assistant", content="AGEP is a platform.")]
    conv_b = [Turn(role="user", content="Tell me about the WiredWave project."), Turn(role="assistant", content="WiredWave is a platform.")]

    result_a = answer_question("What technologies does it use?", embedder=embedder, store=store, manager=shared_manager, conversation=conv_a)
    result_b = answer_question("What technologies does it use?", embedder=embedder, store=store, manager=shared_manager, conversation=conv_b)

    assert result_a.response == "ANSWER_ABOUT_AGEP"
    assert result_b.response == "ANSWER_ABOUT_WIREDWAVE"
    assert result_a.metrics["fallback_used"] is True  # gemini a échoué -> groq a répondu
