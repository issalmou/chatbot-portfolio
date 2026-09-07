"""Tests du comportement anti-hallucination, hors-sujet, ambiguïté et
fallback contextualisé (app/rag/retrieval.py + app/rag/sections.py).

Ces tests portent sur la PLUMBING backend (ce que le backend calcule et
transmet au LLM), pas sur la qualité littéraire d'un vrai LLM : le prompt
final et les métriques sont l'objet vérifié, conformément à l'architecture
(c'est le backend qui décide de la destination suggérée, jamais le LLM).
"""

import pytest

from app.cache.caches import embedding_cache, response_cache, retrieval_cache
from app.ingestion.pipeline import ingest_portfolio
from app.llm.base import LLMMessage
from app.llm.manager import LLMProviderManager
from app.rag.retrieval import SYSTEM_INSTRUCTION, answer_question
from tests.conftest import SAMPLE_TRANSLATIONS_JS, FakeLLMProvider


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


class CapturingProvider(FakeLLMProvider):
    def __init__(self, name: str, reply: str):
        super().__init__(name, reply=reply)
        self.captured: list[LLMMessage] = []

    def generate(self, messages):
        self.captured.extend(messages)
        return super().generate(messages)


# --- Anti-hallucination : le system prompt interdit explicitement l'invention ---

def test_system_prompt_forbids_fabrication_and_general_knowledge():
    assert "invente" in SYSTEM_INSTRUCTION.lower()
    assert "CONTEXTE" in SYSTEM_INSTRUCTION


def test_context_lacking_requested_fact_is_the_only_source_sent_to_llm(seeded_store, fake_embedder):
    # Le CONTEXTE transmis au LLM ne doit contenir QUE des extraits réels du
    # portfolio (jamais une réponse pré-écrite au sujet demandé) : si
    # l'information n'y figure pas, le LLM n'a structurellement aucun moyen
    # de la restituer sans halluciner - c'est le CONTEXTE lui-même qu'on vérifie.
    provider = CapturingProvider("gemini", reply="I don't have that specific detail in the portfolio right now.")
    manager = LLMProviderManager([provider], cooldown_seconds=60)
    answer_question(
        "What is his favorite color?", embedder=fake_embedder, store=seeded_store, manager=manager
    )
    user_message = next(m.content for m in provider.captured if m.role == "user")
    context_part = user_message.split("QUESTION")[0]
    # Le bloc CONTEXTE (avant "QUESTION DE L'UTILISATEUR") ne doit jamais
    # contenir le fait demandé : aucune couleur préférée n'existe dans le
    # portfolio, donc rien de tel ne peut légitimement s'y trouver.
    assert "favorite color" not in context_part.lower()
    assert "couleur" not in context_part.lower()


def test_suggested_section_url_is_backend_resolved_never_llm_invented(seeded_store, fake_embedder):
    provider = CapturingProvider("gemini", reply="Sure, here is what I found in the portfolio about skills.")
    manager = LLMProviderManager([provider], cooldown_seconds=60)
    result = answer_question(
        "What are his skills?", embedder=fake_embedder, store=seeded_store, manager=manager
    )
    user_message = next(m.content for m in provider.captured if m.role == "user")
    assert result.metrics["suggested_route"] == "/about#skills"
    assert "SUGGESTED_SECTION_URL = /about#skills" in user_message


def test_no_suggested_section_for_certification_topic(seeded_store, fake_embedder):
    # Aucune page Certifications n'existe, et aucun chunk n'a ce type
    # d'entité dans les données réelles : le filtrage exhaustif par
    # entity_type="certification" ne retrouve donc rien, et le système
    # retombe sur le message canonique déterministe (sans appel LLM) —
    # aucune URL ne doit jamais être suggérée dans tous les cas.
    provider = CapturingProvider("gemini", reply="should never be called")
    manager = LLMProviderManager([provider], cooldown_seconds=60)
    result = answer_question(
        "Does he have any certifications?", embedder=fake_embedder, store=seeded_store, manager=manager
    )
    assert result.metrics["suggested_route"] is None
    assert provider.call_count == 0
    assert "should never be called" not in result.response


def test_no_suggested_section_for_out_of_scope_question(seeded_store, fake_embedder):
    # Aucun sujet du portfolio n'est identifiable dans cette question : depuis
    # l'ajout de la classification de périmètre (app/rag/scope.py), une telle
    # question est désormais court-circuitée AVANT tout retrieval — donc
    # aucune section ne peut jamais lui être associée.
    provider = CapturingProvider("gemini", reply="should never be called")
    manager = LLMProviderManager([provider], cooldown_seconds=60)
    result = answer_question(
        "What is the capital of Japan?", embedder=fake_embedder, store=seeded_store, manager=manager
    )
    assert result.metrics.get("suggested_route") is None
    assert provider.call_count == 0


# --- Ambiguïté : question anaphorique sans sujet identifiable, sans historique ---

def test_ambiguous_question_asks_for_clarification_without_calling_llm(seeded_store, fake_embedder):
    provider = FakeLLMProvider("gemini", reply="should never be called")
    manager = LLMProviderManager([provider], cooldown_seconds=60)
    result = answer_question("Tell me more about it", embedder=fake_embedder, store=seeded_store, manager=manager)
    assert provider.call_count == 0
    assert result.metrics["intent"] == "ambiguous"
    assert "should never be called" not in result.response


def test_ambiguous_question_in_french_gets_french_clarification(seeded_store, fake_embedder):
    provider = FakeLLMProvider("gemini", reply="should never be called")
    manager = LLMProviderManager([provider], cooldown_seconds=60)
    result = answer_question("Dis-m'en plus", embedder=fake_embedder, store=seeded_store, manager=manager)
    assert result.lang == "fr"
    assert provider.call_count == 0


def test_ambiguous_question_in_arabic_gets_arabic_clarification(seeded_store, fake_embedder):
    provider = FakeLLMProvider("gemini", reply="should never be called")
    manager = LLMProviderManager([provider], cooldown_seconds=60)
    result = answer_question("أخبرني أكثر", embedder=fake_embedder, store=seeded_store, manager=manager)
    assert result.lang == "ar"
    assert provider.call_count == 0


def test_question_with_identifiable_topic_is_not_treated_as_ambiguous(seeded_store, fake_embedder):
    # "tell me more about" seul serait ambigu, mais nommer "projects" lève
    # l'ambiguïté : on répond directement, sans demander de clarification.
    manager, provider = LLMProviderManager([FakeLLMProvider("gemini", reply="Here is more about his projects.")], cooldown_seconds=60), None
    result = answer_question(
        "Tell me more about his projects", embedder=fake_embedder, store=seeded_store, manager=manager
    )
    assert result.metrics["intent"] != "ambiguous"


# --- Hors-sujet : aucune tentative de répondre à partir de connaissances générales ---

def test_out_of_scope_question_never_calls_llm_and_stays_on_topic(seeded_store, fake_embedder):
    # La classification de périmètre (app/rag/scope.py) court-circuite
    # désormais localement les questions hors-sujet manifestes : plus
    # fiable ET moins coûteux qu'avant (l'ancien comportement laissait le
    # LLM se retenir via le prompt, avec un appel LLM gaspillé à chaque fois).
    provider = CapturingProvider("gemini", reply="should never be called")
    manager = LLMProviderManager([provider], cooldown_seconds=60)
    result = answer_question(
        "What's the capital of Japan?", embedder=fake_embedder, store=seeded_store, manager=manager
    )
    assert provider.call_count == 0
    assert "Japan" not in result.response
    assert result.metrics["scope"] == "out_of_scope"


def test_out_of_scope_system_prompt_still_defends_scope_for_edge_cases(seeded_store, fake_embedder):
    # Défense en profondeur : même si la classification locale devait
    # laisser passer un cas limite non couvert, le system prompt du LLM
    # contient toujours sa propre règle "hors sujet".
    assert "hors sujet" in SYSTEM_INSTRUCTION.lower() or "hors-sujet" in SYSTEM_INSTRUCTION.lower()
