"""Tests de la mémoire conversationnelle minimale (app/rag/memory.py) : mêmes
scénarios EN/FR/AR que ceux fournis dans la demande, plus les invariants
critiques (jamais de fuite de langue, jamais d'invention de référent).
"""

import pytest

from app.cache.caches import embedding_cache, response_cache, retrieval_cache
from app.ingestion.pipeline import ingest_portfolio
from app.rag.memory import Turn, resolve_reference
from app.rag.retrieval import answer_question
from tests.conftest import SAMPLE_TRANSLATIONS_JS, FakeLLMProvider
from tests.test_retrieval import JS_FIVE_PROJECTS
from app.llm.manager import LLMProviderManager


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


def _manager(reply="réponse générée"):
    provider = FakeLLMProvider("gemini", reply=reply)
    return LLMProviderManager([provider], cooldown_seconds=60), provider


# --- Unit : resolve_reference() sur les 3 conversations fournies (EN/FR/AR) ---

def test_resolve_reference_english_it_refers_to_previous_project():
    conversation = [
        Turn(role="user", content="Tell me about the AGEP project."),
        Turn(role="assistant", content="AGEP is a management platform."),
    ]
    resolution = resolve_reference("What technologies does it use?", "en", conversation)
    assert resolution.applied is True
    assert resolution.needs_clarification is False
    assert resolution.focus_hint == "AGEP"


def test_resolve_reference_french_il_refers_to_previous_project():
    conversation = [
        Turn(role="user", content="Parle-moi du projet AGEP."),
        Turn(role="assistant", content="AGEP est une plateforme de gestion."),
    ]
    resolution = resolve_reference("Quelles technologies utilise-t-il ?", "fr", conversation)
    assert resolution.applied is True
    assert resolution.needs_clarification is False
    assert resolution.focus_hint == "AGEP"


def test_resolve_reference_arabic_suffix_refers_to_previous_project():
    conversation = [
        Turn(role="user", content="حدثني عن مشروع AGEP."),
        Turn(role="assistant", content="AGEP هي منصة لإدارة الفرق."),
    ]
    resolution = resolve_reference("ما هي التقنيات التي يستخدمها؟", "ar", conversation)
    assert resolution.applied is True
    assert resolution.needs_clarification is False
    assert resolution.focus_hint == "AGEP"


def test_resolve_reference_topic_level_focus_skills_then_filter():
    # "What are his main skills?" -> "Which ones are related to frontend
    # development?" : le référent est le SUJET (skills), pas un nom de projet.
    conversation = [
        Turn(role="user", content="What are his main skills?"),
        Turn(role="assistant", content="His main skills include backend and frontend development."),
    ]
    resolution = resolve_reference("Which ones are related to frontend development?", "en", conversation)
    assert resolution.applied is True
    assert resolution.focus_topic == "skills"


# --- Jamais d'invention : aucune conversation / aucun focus trouvable ---

def test_resolve_reference_never_guesses_without_any_context():
    resolution = resolve_reference("Tell me more about it.", "en", conversation=[])
    assert resolution.needs_clarification is True
    assert resolution.applied is False


def test_resolve_reference_never_guesses_when_conversation_has_no_focus():
    conversation = [
        Turn(role="user", content="Hello!"),
        Turn(role="assistant", content="Hi, how can I help you today?"),
    ]
    resolution = resolve_reference("Tell me more about it.", "en", conversation)
    assert resolution.needs_clarification is True


# --- Une question qui porte déjà son propre sujet n'a besoin d'aucune mémoire ---

def test_resolve_reference_never_captures_ordinary_word_as_entity_name():
    # Bug réel corrigé (découvert via un VRAI texte généré par Gemini, voir
    # rapport) : "...is a web development project completed in July 2024..."
    # capturait "completed" comme nom de projet ("project completed" matche
    # le motif "project NAME" si NAME n'est pas contraint à être capitalisé),
    # empêchant la boucle d'atteindre le tour plus ancien où "AGEP" est
    # réellement mentionné. Tous les vrais noms de projets sont capitalisés.
    conversation = [
        Turn(role="user", content="Tell me about the AGEP project."),
        Turn(role="assistant", content=(
            "AGEP is a web development project completed in July 2024 by Issalmou Adaaiche. "
            "It automates HR workflows for paramedical teams."
        )),
    ]
    resolution = resolve_reference("What technologies does it use?", "en", conversation)
    assert resolution.focus_entity_name == "AGEP"
    assert resolution.focus_entity_name != "completed"


def test_resolve_reference_self_sufficient_question_is_left_unchanged():
    resolution = resolve_reference("Tell me about the AGEP project.", "en", conversation=[])
    assert resolution.retrieval_query == "Tell me about the AGEP project."
    assert resolution.applied is False
    assert resolution.needs_clarification is False


# --- Fenêtre bornée : un focus trop ancien (au-delà de MAX_TURNS) est ignoré ---

def test_resolve_reference_ignores_focus_outside_the_short_window():
    from app.rag.memory import MAX_TURNS

    old_focus = [
        Turn(role="user", content="Tell me about the AGEP project."),
        Turn(role="assistant", content="AGEP is a management platform."),
    ]
    filler = [Turn(role="user", content=f"Hello {i}") for i in range(MAX_TURNS * 2)] + \
             [Turn(role="assistant", content=f"Hi {i}") for i in range(MAX_TURNS * 2)]
    conversation = old_focus + filler
    resolution = resolve_reference("What technologies does it use?", "en", conversation)
    assert resolution.needs_clarification is True  # le focus AGEP est hors fenêtre


# --- End-to-end (pipeline complet) : la conversation résout bien la question ---

def test_end_to_end_follow_up_resolves_to_agep_and_uses_context_in_prompt(seeded_store, fake_embedder):
    class CapturingProvider(FakeLLMProvider):
        def __init__(self):
            super().__init__("gemini", reply="AGEP uses Laravel and MySQL.")
            self.captured = []

        def generate(self, messages):
            self.captured.extend(messages)
            return super().generate(messages)

    provider = CapturingProvider()
    manager = LLMProviderManager([provider], cooldown_seconds=60)
    conversation = [
        Turn(role="user", content="Tell me about the AGEP project."),
        Turn(role="assistant", content="AGEP is a platform for paramedical teams."),
    ]
    result = answer_question(
        "What technologies does it use?", embedder=fake_embedder, store=seeded_store,
        manager=manager, conversation=conversation,
    )
    assert result.metrics["scope"] == "portfolio_related"
    user_message = next(m.content for m in provider.captured if m.role == "user")
    assert "CONVERSATION_FOCUS = AGEP" in user_message


def test_follow_up_disambiguates_correct_project_among_several_similar_ones(tmp_store, fake_embedder):
    # Bug réel corrigé (voir rapport) : avec plusieurs projets aux chunks
    # sémantiquement proches, un simple indice ajouté à l'embedding ne
    # suffisait pas toujours à faire remonter la bonne entité en tête du
    # top_k. Le lookup DIRECT par entity_id (ChromaStore.get_by_entity_id)
    # garantit que le bon projet — et UNIQUEMENT lui — est utilisé.
    ingest_portfolio(JS_FIVE_PROJECTS.encode("utf-8"), client=fake_embedder, store=tmp_store)

    class CapturingProvider(FakeLLMProvider):
        def __init__(self):
            super().__init__("gemini", reply="AGEP uses Python.")
            self.captured = []

        def generate(self, messages):
            self.captured.extend(messages)
            return super().generate(messages)

    provider = CapturingProvider()
    manager = LLMProviderManager([provider], cooldown_seconds=60)
    conversation = [
        Turn(role="user", content="Tell me about the AGEP project."),
        Turn(role="assistant", content="AGEP is one of his projects."),
    ]
    result = answer_question(
        "What technologies does it use?", embedder=fake_embedder, store=tmp_store,
        manager=manager, conversation=conversation,
    )
    assert result.metrics["retrieved_chunks"] == 1  # lookup direct : une seule entité, pas un top_k mélangé
    user_message = next(m.content for m in provider.captured if m.role == "user")
    context_part = user_message.split("QUESTION")[0]
    assert "AGEP" in context_part
    for other in ("ESTICAR", "WIREDWAVE", "SPEECHLY", "DESCRIPTOAI"):
        assert other not in context_part


def test_end_to_end_no_context_and_referential_question_asks_for_clarification(seeded_store, fake_embedder):
    manager, provider = _manager(reply="should never be called")
    result = answer_question(
        "Tell me more about it.", embedder=fake_embedder, store=seeded_store, manager=manager, conversation=[],
    )
    assert provider.call_count == 0
    assert result.metrics["scope"] == "ambiguous"


# --- Changement de langue : la mémoire ne doit JAMAIS déterminer la langue ---

def test_language_switch_english_then_french_follow_up_answers_in_french(seeded_store, fake_embedder):
    manager, provider = _manager(reply="réponse en français sur AGEP et ses technologies utilisées.")
    conversation = [
        Turn(role="user", content="Tell me about the AGEP project."),
        Turn(role="assistant", content="AGEP is a platform for paramedical teams."),
    ]
    result = answer_question(
        "Quelles technologies utilise-t-il ?", embedder=fake_embedder, store=seeded_store,
        manager=manager, conversation=conversation,
    )
    assert result.lang == "fr"


def test_language_switch_french_then_arabic_follow_up_answers_in_arabic(seeded_store, fake_embedder):
    manager, provider = _manager(reply="نص طويل بالعربية عن هذا الموضوع وعن مهاراته وتقنياته المختلفة.")
    conversation = [
        Turn(role="user", content="Parle-moi de son expérience."),
        Turn(role="assistant", content="Il a fait un stage chez ACME en tant que développeur."),
    ]
    result = answer_question(
        "ما هي مهاراته؟", embedder=fake_embedder, store=seeded_store,
        manager=manager, conversation=conversation,
    )
    assert result.lang == "ar"


# --- Cache : deux conversations différentes ne doivent jamais partager le cache ---

def test_different_conversation_focus_yields_different_cache_key(seeded_store, fake_embedder):
    conversation_agep = [
        Turn(role="user", content="Tell me about the AGEP project."),
        Turn(role="assistant", content="AGEP is a platform."),
    ]
    conversation_other = [
        Turn(role="user", content="Tell me about the WiredWave project."),
        Turn(role="assistant", content="WiredWave is a platform."),
    ]
    resolution_a = resolve_reference("What technologies does it use?", "en", conversation_agep)
    resolution_b = resolve_reference("What technologies does it use?", "en", conversation_other)
    assert resolution_a.retrieval_query != resolution_b.retrieval_query


def test_empty_conversation_does_not_change_cache_key_or_behavior(seeded_store, fake_embedder):
    # Non-régression explicite : `conversation=None`/`[]` doit produire un
    # texte de retrieval strictement identique à la question brute.
    resolution = resolve_reference("Quels projets ?", "fr", conversation=None)
    assert resolution.retrieval_query == "Quels projets ?"
    assert resolution.applied is False
