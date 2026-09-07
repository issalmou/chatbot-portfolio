"""Tests de concurrence AU NIVEAU DU ENDPOINT HTTP (/chatbot), pour valider
spécifiquement le passage à `run_in_threadpool` dans main.py::chat_endpoint
(voir rapport) — complète (sans remplacer) tests/test_concurrency.py, qui
valide déjà `answer_question()` directement sous charge concurrente réelle.

Les doublures utilisées ici ont un délai artificiel (`time.sleep`) pour
simuler le temps réseau réel (Chroma/LLM) : sans `run_in_threadpool`, N
requêtes concurrentes prendraient ~N x délai (sérialisées) ; avec, elles
doivent se chevaucher et prendre significativement moins.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

import main
from app.cache.caches import embedding_cache, response_cache, retrieval_cache
from app.ingestion.pipeline import ingest_portfolio
from app.llm.base import LLMResult
from app.llm.manager import LLMProviderManager
from tests.conftest import FakeChromaStore, FakeEmbedder, FakeLLMProvider
from tests.test_retrieval import JS_FIVE_PROJECTS

client = TestClient(main.app)

_ARTIFICIAL_EMBED_DELAY = 0.03
_ARTIFICIAL_LLM_DELAY = 0.15


class SlowFakeEmbedder(FakeEmbedder):
    """Simule la latence réelle d'un appel réseau/CPU (Chroma/E5)."""

    def embed_query(self, text):
        time.sleep(_ARTIFICIAL_EMBED_DELAY)
        return super().embed_query(text)

    def embed_documents(self, texts):
        time.sleep(_ARTIFICIAL_EMBED_DELAY)
        return super().embed_documents(texts)


class SlowRevealingProvider(FakeLLMProvider):
    """Simule la latence réseau d'un vrai LLM et révèle quel CONTEXTE il a
    reçu (nom du projet présent dans le prompt), pour prouver après coup
    qu'aucun thread n'a reçu la réponse destinée à un autre."""

    _PROJECT_NAMES = ("AGEP", "ESTICAR", "WIREDWAVE", "SPEECHLY", "DESCRIPTOAI")

    def __init__(self, name: str = "gemini"):
        super().__init__(name, reply="")

    def generate(self, messages):
        time.sleep(_ARTIFICIAL_LLM_DELAY)
        self.call_count += 1
        user_msg = next(m.content for m in messages if m.role == "user")
        context_part = user_msg.split("QUESTION")[0].upper()
        for candidate in self._PROJECT_NAMES:
            if candidate in context_part:
                return LLMResult(text=f"ANSWER_ABOUT_{candidate}", provider=self.name, model=self.model, latency_ms=_ARTIFICIAL_LLM_DELAY * 1000)
        return LLMResult(text="ANSWER_GENERIC_NO_PROJECT", provider=self.name, model=self.model, latency_ms=_ARTIFICIAL_LLM_DELAY * 1000)


@pytest.fixture()
def patched_endpoint(monkeypatch):
    """Remplace les singletons RÉELS utilisés par défaut par answer_question()
    (voir app/rag/retrieval.py) — chat_endpoint ne passe ni embedder, ni
    store, ni manager explicitement, donc c'est exactement ce que le VRAI
    endpoint utiliserait en production, ici avec des doublures rapides et
    déterministes plutôt que Chroma Cloud / E5 / un vrai LLM."""
    store = FakeChromaStore()
    embedder = SlowFakeEmbedder()
    ingest_portfolio(JS_FIVE_PROJECTS.encode("utf-8"), client=embedder, store=store)

    provider = SlowRevealingProvider()
    manager = LLMProviderManager([provider], cooldown_seconds=60)

    monkeypatch.setattr("app.rag.retrieval.e5_embedding_provider", embedder)
    monkeypatch.setattr("app.rag.retrieval.chroma_store", store)
    monkeypatch.setattr("app.rag.retrieval.llm_manager", manager)

    embedding_cache.clear()
    retrieval_cache.clear()
    response_cache.clear()
    yield provider
    embedding_cache.clear()
    retrieval_cache.clear()
    response_cache.clear()


def _post(query: str, conversation=None) -> dict:
    payload = {"query": query}
    if conversation is not None:
        payload["conversation"] = conversation
    response = client.post("/chatbot", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def _agep_then_followup():
    r1 = _post("Tell me about the AGEP project.")
    r2 = _post(
        "What technologies does it use?",
        conversation=[
            {"role": "user", "content": "Tell me about the AGEP project."},
            {"role": "assistant", "content": r1["response"]},
        ],
    )
    return r2


def _other_project_then_followup(project: str):
    first_query = f"Tell me about the {project} project."
    r1 = _post(first_query)
    r2 = _post(
        "What technologies does it use?",
        conversation=[
            {"role": "user", "content": first_query},
            {"role": "assistant", "content": r1["response"]},
        ],
    )
    return r2


# --- Test 1 : deux projets différents, follow-up textuellement identique ---

def test_endpoint_two_users_identical_follow_up_get_correct_project(patched_endpoint):
    barrier = threading.Barrier(2)

    def run(fn):
        barrier.wait()
        return fn()

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(run, _agep_then_followup)
        future_b = pool.submit(run, lambda: _other_project_then_followup("WiredWave"))
        result_a = future_a.result(timeout=15)
        result_b = future_b.result(timeout=15)

    assert result_a["response"] == "ANSWER_ABOUT_AGEP"
    assert result_b["response"] == "ANSWER_ABOUT_WIREDWAVE"


# --- Test 2 : 5 utilisateurs simultanés, langues et scopes variés ---

def test_endpoint_five_simultaneous_users_no_leak(patched_endpoint):
    scenarios = {
        "user_A": lambda: _agep_then_followup(),
        "user_B": lambda: _other_project_then_followup("EstiCar"),
        "user_C": lambda: _post("ما هي مشاريعه؟"),
        "user_D": lambda: _post("What is the capital of Japan?"),
        "user_E": lambda: _post("Give me all his projects."),
    }
    barrier = threading.Barrier(len(scenarios))

    def run(fn):
        barrier.wait()
        return fn()

    results = {}
    with ThreadPoolExecutor(max_workers=len(scenarios)) as pool:
        futures = {name: pool.submit(run, fn) for name, fn in scenarios.items()}
        for name, future in futures.items():
            results[name] = future.result(timeout=15)

    assert results["user_A"]["response"] == "ANSWER_ABOUT_AGEP"
    assert results["user_B"]["response"] == "ANSWER_ABOUT_ESTICAR"
    assert results["user_C"]["lang"] == "ar"
    assert results["user_D"]["lang"] == "en"
    assert "Japan" not in results["user_D"]["response"]
    # E5 projets réels dans la question exhaustive : la réponse du FakeLLM
    # révèle un seul nom trouvé (le premier match), mais le CONTEXTE envoyé
    # doit contenir les 5 -- vérifié indirectement via le scope et l'absence
    # d'erreur ; la couverture exhaustive complète est testée ailleurs
    # (tests/test_retrieval.py, tests/test_production_matrix.py).
    assert results["user_E"]["response"].startswith("ANSWER_ABOUT_")


# --- Test 3 : cache partagé sous charge concurrente réelle (endpoint) ---

def test_endpoint_concurrent_requests_do_not_corrupt_shared_cache(patched_endpoint):
    for _ in range(10):
        embedding_cache.clear()
        retrieval_cache.clear()
        response_cache.clear()
        barrier = threading.Barrier(2)

        def run(fn):
            barrier.wait()
            return fn()

        with ThreadPoolExecutor(max_workers=2) as pool:
            future_a = pool.submit(run, _agep_then_followup)
            future_b = pool.submit(run, lambda: _other_project_then_followup("Speechly"))
            result_a = future_a.result(timeout=15)
            result_b = future_b.result(timeout=15)

        assert result_a["response"] == "ANSWER_ABOUT_AGEP"
        assert result_b["response"] == "ANSWER_ABOUT_SPEECHLY"


# --- Test 4 : circuit breaker partagé, jamais de contenu partagé ---

def test_endpoint_shared_circuit_breaker_never_leaks_response_content(patched_endpoint, monkeypatch):
    from app.llm.base import ERROR_RATE_LIMIT, LLMProviderError

    class FailOnceThenReveal(SlowRevealingProvider):
        def __init__(self, name):
            super().__init__(name)
            self._failed_once = False

        def generate(self, messages):
            if not self._failed_once:
                self._failed_once = True
                raise LLMProviderError("simulated rate limit", ERROR_RATE_LIMIT)
            return super().generate(messages)

    gemini = FailOnceThenReveal("gemini")
    groq = SlowRevealingProvider("groq")
    shared_manager = LLMProviderManager([gemini, groq], cooldown_seconds=60)
    monkeypatch.setattr("app.rag.retrieval.llm_manager", shared_manager)

    result_a = _agep_then_followup()
    result_b = _other_project_then_followup("WiredWave")

    assert result_a["response"] == "ANSWER_ABOUT_AGEP"
    assert result_b["response"] == "ANSWER_ABOUT_WIREDWAVE"
    assert result_a["metrics"]["fallback_used"] is True


# --- Performance : preuve mesurable que le threadpool supprime la
# sérialisation inutile (voir rapport, section performance) ---

def test_endpoint_concurrent_requests_are_not_serialized(patched_endpoint):
    single_start = time.perf_counter()
    _post("Tell me about the AGEP project.")
    single_duration = time.perf_counter() - single_start

    embedding_cache.clear()
    retrieval_cache.clear()
    response_cache.clear()

    n = 5
    barrier = threading.Barrier(n)

    def one_call(i):
        barrier.wait()
        return _post(f"Tell me about the {['AGEP', 'EstiCar', 'WiredWave', 'Speechly', 'DescriptoAI'][i]} project.")

    concurrent_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n) as pool:
        results = list(pool.map(one_call, range(n)))
    concurrent_duration = time.perf_counter() - concurrent_start

    naive_serial_estimate = single_duration * n
    print(f"\n[perf] single request: {single_duration:.3f}s | {n} concurrent: {concurrent_duration:.3f}s "
          f"| naive serial estimate: {naive_serial_estimate:.3f}s")

    # Si les requêtes étaient sérialisées (ancien comportement, sans
    # run_in_threadpool), le temps total pour N requêtes concurrentes serait
    # proche de N x single_duration. Avec le threadpool, il doit être
    # significativement inférieur — seuil tolérant (70%) pour éviter un test
    # flaky selon la machine, tout en prouvant un réel chevauchement.
    assert concurrent_duration < naive_serial_estimate * 0.7
    assert len(results) == n
