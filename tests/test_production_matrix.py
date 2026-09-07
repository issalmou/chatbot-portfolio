"""Matrice de tests combinés pour la dernière phase de durcissement avant
production : changements de langue directionnels complets, combinaisons
intent/scope/mémoire, cas "React", et le bug réel découvert dans cette
session (تقنياته ne déclenchait pas la résolution de référence en arabe).
Complète (sans remplacer) test_memory.py / test_scope.py / test_hardening.py.
"""

import pytest

from app.cache.caches import embedding_cache, response_cache, retrieval_cache
from app.ingestion.pipeline import ingest_portfolio
from app.language import detect_language
from app.llm.manager import LLMProviderManager
from app.rag.intent import detect_list_intent
from app.rag.memory import Turn, resolve_reference
from app.rag.retrieval import answer_question
from app.rag.scope import Scope, classify_scope
from app.rag.sections import detect_topic
from tests.conftest import SAMPLE_TRANSLATIONS_JS, FakeLLMProvider
from tests.test_retrieval import JS_FIVE_PROJECTS


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


@pytest.fixture()
def five_projects_store(tmp_store, fake_embedder):
    ingest_portfolio(JS_FIVE_PROJECTS.encode("utf-8"), client=fake_embedder, store=tmp_store)
    return tmp_store


def _manager(reply="réponse suffisamment longue pour une détection de langue fiable"):
    return LLMProviderManager([FakeLLMProvider("gemini", reply=reply)], cooldown_seconds=60)


def _scope_of(q, lang, conv=None):
    res = resolve_reference(q, lang, conv)
    topic = detect_topic(q, lang) or res.focus_topic
    is_exh = detect_list_intent(q, lang)
    return classify_scope(q, lang, topic, is_exh, res)


# --- Bug réel corrigé cette session : "تقنياته" ne déclenchait pas la
# résolution de référence en arabe (seulement la classification de scope) ---

def test_arabic_technologies_possessive_resolves_to_single_project_focus():
    conversation = [
        Turn(role="user", content="حدثني عن مشروع AGEP."),
        Turn(role="assistant", content="AGEP منصة جيدة."),
    ]
    resolution = resolve_reference("ما هي تقنياته؟", "ar", conversation)
    assert resolution.applied is True
    assert resolution.focus_entity_name == "AGEP"


def test_arabic_technologies_possessive_ambiguous_with_two_projects():
    conversation = [
        Turn(role="user", content="حدثني عن مشروع AGEP و مشروع EstiCar."),
        Turn(role="assistant", content="كلاهما مشروعان جيدان."),
    ]
    resolution = resolve_reference("ما هي تقنياته؟", "ar", conversation)
    assert resolution.needs_clarification is True


def test_arabic_technologies_possessive_end_to_end_isolates_correct_project(five_projects_store, fake_embedder):
    class CapturingProvider(FakeLLMProvider):
        def __init__(self):
            super().__init__("gemini", reply="AGEP يستخدم Python بشكل أساسي.")
            self.captured = []

        def generate(self, messages):
            self.captured.extend(messages)
            return super().generate(messages)

    provider = CapturingProvider()
    manager = LLMProviderManager([provider], cooldown_seconds=60)
    conversation = [
        Turn(role="user", content="حدثني عن مشروع AGEP."),
        Turn(role="assistant", content="AGEP هي منصة لإدارة الفرق."),
    ]
    result = answer_question(
        "ما هي تقنياته؟", embedder=fake_embedder, store=five_projects_store,
        manager=manager, conversation=conversation,
    )
    assert result.metrics["retrieved_chunks"] == 1
    user_message = next(m.content for m in provider.captured if m.role == "user")
    context_part = user_message.split("QUESTION")[0]
    assert "AGEP" in context_part
    for other in ("ESTICAR", "WIREDWAVE", "SPEECHLY", "DESCRIPTOAI"):
        assert other not in context_part


# --- Mots référentiels arabes additionnels (ذلك/عنه/منه/فيه) ---

@pytest.mark.parametrize("word", ["ذلك", "عنه", "منه", "فيه"])
def test_additional_arabic_referential_words_trigger_resolution(word):
    conversation = [
        Turn(role="user", content="حدثني عن مشروع AGEP."),
        Turn(role="assistant", content="AGEP منصة جيدة."),
    ]
    resolution = resolve_reference(f"أخبرني أكثر {word}", "ar", conversation)
    assert resolution.applied is True
    assert resolution.focus_entity_name == "AGEP"


# --- Les 6 paires directionnelles de changement de langue (section 2/21) ---

@pytest.mark.parametrize("first,first_lang,second,second_lang", [
    ("Tell me about AGEP.", "en", "Quelles technologies utilise-t-il ?", "fr"),
    ("Parle-moi du projet AGEP.", "fr", "What technologies does it use?", "en"),
    ("ما هي مهاراته؟", "ar", "Quelles sont ses compétences ?", "fr"),
    ("Quelles sont ses compétences ?", "fr", "ما هي مهاراته؟", "ar"),
    ("What is his experience?", "en", "ما هي خبرته؟", "ar"),
    ("ما هي خبرته؟", "ar", "What is his experience?", "en"),
])
def test_directional_language_pair_transition(seeded_store, fake_embedder, first, first_lang, second, second_lang):
    manager = _manager()
    r1 = answer_question(first, embedder=fake_embedder, store=seeded_store, manager=manager)
    assert r1.lang == first_lang

    conversation = [Turn(role="user", content=first), Turn(role="assistant", content=r1.response)]
    r2 = answer_question(second, embedder=fake_embedder, store=seeded_store, manager=manager, conversation=conversation)
    assert r2.lang == second_lang


# --- Combinaisons intent / scope / mémoire ---

def test_specific_then_ambiguous_resolves_via_memory_not_clarification(five_projects_store, fake_embedder):
    manager = _manager()
    r1 = answer_question("Tell me about AGEP.", embedder=fake_embedder, store=five_projects_store, manager=manager)
    conversation = [Turn(role="user", content="Tell me about AGEP."), Turn(role="assistant", content=r1.response)]
    r2 = answer_question("Tell me more.", embedder=fake_embedder, store=five_projects_store, manager=manager, conversation=conversation)
    assert r2.metrics["scope"] != "ambiguous"


def test_specific_then_exhaustive_returns_all_projects_not_just_prior_focus(five_projects_store, fake_embedder):
    manager = _manager()
    r1 = answer_question("Tell me about AGEP.", embedder=fake_embedder, store=five_projects_store, manager=manager)
    conversation = [Turn(role="user", content="Tell me about AGEP."), Turn(role="assistant", content=r1.response)]
    r2 = answer_question(
        "Give me all his projects.", embedder=fake_embedder, store=five_projects_store,
        manager=manager, conversation=conversation,
    )
    assert r2.metrics["unique_entities_count"] == 5


def test_memory_focus_then_exhaustive_different_topic_is_not_stuck_on_prior_project(five_projects_store, fake_embedder):
    manager = _manager()
    r1 = answer_question("Tell me about AGEP.", embedder=fake_embedder, store=five_projects_store, manager=manager)
    conversation = [Turn(role="user", content="Tell me about AGEP."), Turn(role="assistant", content=r1.response)]
    r2 = answer_question(
        "What are all his skills?", embedder=fake_embedder, store=five_projects_store,
        manager=manager, conversation=conversation,
    )
    assert r2.metrics["intent"] == "list_entities"
    assert r2.metrics["entity_type"] != "project"


def test_memory_focus_then_out_of_scope_question_stays_out_of_scope(five_projects_store, fake_embedder):
    manager = _manager()
    r1 = answer_question("Tell me about AGEP.", embedder=fake_embedder, store=five_projects_store, manager=manager)
    conversation = [Turn(role="user", content="Tell me about AGEP."), Turn(role="assistant", content=r1.response)]
    r2 = answer_question(
        "What is the capital of Japan?", embedder=fake_embedder, store=five_projects_store,
        manager=manager, conversation=conversation,
    )
    assert r2.metrics["scope"] == "out_of_scope"
    assert "Japan" not in r2.response


# --- Cas "React" : quatre formulations, classifications différenciées ---

def test_react_general_knowledge_is_out_of_scope():
    assert _scope_of("What is React?", "en") == Scope.OUT_OF_SCOPE


def test_react_project_filter_is_portfolio_related_and_exhaustive():
    assert _scope_of("Which of his projects use React?", "en") == Scope.PORTFOLIO_RELATED
    assert detect_list_intent("Which of his projects use React?", "en") is True


def test_react_yes_no_question_is_portfolio_related_via_personal_reference():
    assert _scope_of("Does he use React?", "en") == Scope.PORTFOLIO_RELATED
    assert detect_list_intent("Does he use React?", "en") is False


def test_react_experience_question_is_portfolio_related_and_exhaustive():
    assert _scope_of("Tell me about his React experience.", "en") == Scope.PORTFOLIO_RELATED
    assert detect_list_intent("Tell me about his React experience.", "en") is True


# --- Formulations exhaustives supplémentaires (section 10) ---

@pytest.mark.parametrize("query,lang", [
    ("What are all the projects he has worked on?", "en"),
    ("Give me the complete list of projects.", "en"),
    ("Présente-moi l'ensemble de ses projets.", "fr"),
    ("أعطني قائمة كاملة بمشاريعه.", "ar"),
    ("اذكر كل مشاريعه.", "ar"),
    ("ما هي كافة مشاريعه؟", "ar"),
])
def test_additional_exhaustive_phrasings_detected(query, lang):
    assert detect_list_intent(query, lang) is True


def test_additional_exhaustive_phrasings_return_all_five_real_projects(five_projects_store, fake_embedder):
    manager = _manager()
    for query in [
        "What are all the projects he has worked on?",
        "Give me the complete list of projects.",
    ]:
        result = answer_question(query, embedder=fake_embedder, store=five_projects_store, manager=manager)
        assert result.metrics["unique_entities_count"] == 5
