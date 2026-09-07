"""Tests du garde-fou multi-niveaux anti-fuite de langue (app/rag/retrieval.py).

Principe testé : la langue de la RÉPONSE doit toujours être celle de la
QUESTION, jamais celle du CONTEXTE récupéré (qui peut légitimement être dans
une autre langue, le retrieval étant volontairement cross-langue). Le
mécanisme de secours (niveau 4, retry borné) n'est déclenché que lorsque la
validation locale (sans réseau) détecte un mismatch net.
"""

import pytest

from app.cache.caches import embedding_cache, response_cache, retrieval_cache
from app.ingestion.pipeline import ingest_portfolio
from app.llm.base import LLMMessage, LLMResult
from app.llm.manager import LLMProviderManager
from app.rag.retrieval import answer_question
from tests.conftest import SAMPLE_TRANSLATIONS_JS, FakeLLMProvider


class SequencedLLMProvider(FakeLLMProvider):
    """Retourne une réponse différente à chaque appel, pour simuler un
    premier essai dans la mauvaise langue suivi d'une correction au retry."""

    def __init__(self, name: str, replies: list[str]):
        super().__init__(name, reply=replies[0])
        self._replies = replies

    def generate(self, messages: list[LLMMessage]) -> LLMResult:
        index = min(self.call_count, len(self._replies) - 1)
        self.call_count += 1
        return LLMResult(text=self._replies[index], provider=self.name, model=self.model, latency_ms=1.0)


def _manager(replies: list[str]):
    provider = SequencedLLMProvider("gemini", replies)
    return LLMProviderManager([provider], cooldown_seconds=60), provider


@pytest.fixture(autouse=True)
def _clear_caches():
    embedding_cache.clear()
    retrieval_cache.clear()
    response_cache.clear()
    yield
    embedding_cache.clear()
    retrieval_cache.clear()
    response_cache.clear()


@pytest.fixture()
def seeded_store(tmp_store, fake_embedder):
    ingest_portfolio(SAMPLE_TRANSLATIONS_JS.encode("utf-8"), client=fake_embedder, store=tmp_store)
    return tmp_store


# --- Matrice explicite demandée : question X, contexte de langue Y -> réponse toujours en X ---

def test_french_question_english_context_answers_french(seeded_store, fake_embedder):
    # "AGEP" est présent en EN et FR : la requête cible spécifiquement AGEP,
    # dont le premier essai simule une réponse restée en anglais (comme si le
    # LLM avait suivi la langue du contexte), corrigée ensuite par le retry.
    manager, provider = _manager([
        "AGEP is a web platform for paramedical teams.",
        "AGEP est une plateforme web pour les équipes paramédicales et bien plus encore.",
    ])
    result = answer_question("Parle-moi du projet AGEP.", embedder=fake_embedder, store=seeded_store, manager=manager)
    assert result.lang == "fr"
    assert result.metrics["language_retry"] is True
    assert provider.call_count == 2


def test_english_question_french_context_answers_english(seeded_store, fake_embedder):
    manager, provider = _manager([
        "AGEP est une plateforme de gestion pour les équipes paramédicales.",
        "AGEP is a management platform for paramedical teams and much more indeed.",
    ])
    result = answer_question("Tell me about the AGEP project.", embedder=fake_embedder, store=seeded_store, manager=manager)
    assert result.lang == "en"
    assert result.metrics["language_retry"] is True
    assert provider.call_count == 2


def test_arabic_question_english_context_answers_arabic(seeded_store, fake_embedder):
    manager, provider = _manager([
        "AGEP is a platform used to manage paramedical teams efficiently.",
        "AGEP هي منصة تُستخدم لإدارة الفرق شبه الطبية بكفاءة كبيرة جدا.",
    ])
    result = answer_question("حدثني عن مشروع AGEP.", embedder=fake_embedder, store=seeded_store, manager=manager)
    assert result.lang == "ar"
    assert result.metrics["language_retry"] is True
    assert provider.call_count == 2


def test_arabic_question_french_context_answers_arabic(seeded_store, fake_embedder):
    manager, provider = _manager([
        "AGEP est une plateforme destinée à la gestion des équipes paramédicales.",
        "AGEP هي منصة مخصصة لإدارة الفرق شبه الطبية بشكل جيد جدا.",
    ])
    result = answer_question("حدثني عن مشروع AGEP.", embedder=fake_embedder, store=seeded_store, manager=manager)
    assert result.lang == "ar"
    assert result.metrics["language_retry"] is True
    assert provider.call_count == 2


def test_no_retry_when_first_answer_already_matches_question_language(seeded_store, fake_embedder):
    manager, provider = _manager([
        "AGEP is a platform for managing paramedical teams and their schedules.",
        "This second reply should never be used in this test.",
    ])
    result = answer_question("Tell me about the AGEP project.", embedder=fake_embedder, store=seeded_store, manager=manager)
    assert result.lang == "en"
    assert result.metrics["language_retry"] is False
    assert provider.call_count == 1
    assert "second reply" not in result.response


def test_short_reply_does_not_falsely_trigger_retry(seeded_store, fake_embedder):
    # Les réponses très courtes ne sont pas fiables pour la détection de
    # langue (ex. "OK", "AGEP") : elles ne doivent jamais déclencher de retry.
    manager, provider = _manager(["OK", "should not be called"])
    result = answer_question("Tell me about the AGEP project.", embedder=fake_embedder, store=seeded_store, manager=manager)
    assert result.metrics["language_retry"] is False
    assert provider.call_count == 1
    assert result.response == "OK"

