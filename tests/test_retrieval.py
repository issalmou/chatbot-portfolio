import pytest

from app.cache.caches import embedding_cache, response_cache, retrieval_cache
from app.ingestion.pipeline import ingest_portfolio
from app.llm.base import LLMMessage, LLMProviderError, ERROR_AUTH
from app.llm.manager import LLMProviderManager
from app.rag.retrieval import AllProvidersUnavailableError, answer_question
from tests.conftest import FakeLLMProvider

JS_SAMPLE = """
const translations = {
    fr: {
        about: { title: "À propos", headline: "Issalmou Adaaiche, développeur du portfolio AGEP.", description: "Dev.", quote: "Q." },
        projects: { title: "Projets", projects_details: [
            { title: "AGEP", shortName: "agep", date: "2024", type: "Web",
              description: "Description du projet AGEP en français.", theChallenge: "x", theSolution: "y",
              keyFeatures: ["f"], technologies: ["Laravel"] }
        ] },
        projectDetails: { theChallenge: "Défi", theSolution: "Solution", keyFeatures: "Fonctionnalités" },
    },
    en: {
        about: { title: "About", headline: "I am Issalmou Adaaiche.", description: "Dev.", quote: "Q." },
    },
    ar: {
        about: { title: "حول", headline: "أنا Issalmou Adaaiche.", description: "مطور.", quote: "ق." },
    },
};
"""


def _project_block(short_name: str, title: str) -> str:
    return (
        '{ title: "%s", shortName: "%s", date: "2024", type: "Web", '
        'description: "Description du projet %s.", theChallenge: "x", theSolution: "y", '
        'keyFeatures: ["f"], technologies: ["Python"] }' % (title, short_name, title)
    )


JS_FIVE_PROJECTS = """
const translations = {
    en: {
        about: { title: "About", headline: "Issalmou Adaaiche is a developer.", description: "Dev.", quote: "Q." },
        projects: { title: "Projects", projects_details: [%s] },
        projectDetails: { theChallenge: "Challenge", theSolution: "Solution", keyFeatures: "Key Features" },
    },
};
""" % ",".join(
    _project_block(name, name.upper())
    for name in ["agep", "esticar", "wiredwave", "speechly", "descriptoai"]
)


@pytest.fixture()
def five_projects_store(tmp_store, fake_embedder):
    ingest_portfolio(JS_FIVE_PROJECTS.encode("utf-8"), client=fake_embedder, store=tmp_store)
    return tmp_store


@pytest.mark.parametrize(
    "query,lang",
    [
        ("Give me all projects realized by the developer", "en"),
        ("List all his projects", "en"),
        ("What projects has he developed?", "en"),
        ("Show me every project", "en"),
    ],
)
def test_exhaustive_query_retrieves_all_five_projects_en(five_projects_store, fake_embedder, query, lang):
    manager, gemini = _manager(reply="All five projects listed.")
    result = answer_question(query, embedder=fake_embedder, store=five_projects_store, manager=manager)
    assert result.lang == lang
    assert result.metrics["intent"] == "list_entities"
    assert result.metrics["unique_entities_count"] == 5
    assert set(result.metrics["entity_ids"]) == {"agep", "esticar", "wiredwave", "speechly", "descriptoai"}


def test_exhaustive_query_french(five_projects_store, fake_embedder):
    manager, _ = _manager()
    result = answer_question(
        "Quels sont tous les projets réalisés par le développeur ?",
        embedder=fake_embedder, store=five_projects_store, manager=manager,
    )
    assert result.lang == "fr"
    assert result.metrics["unique_entities_count"] == 5


def test_exhaustive_query_arabic(five_projects_store, fake_embedder):
    manager, _ = _manager()
    result = answer_question(
        "ما هي جميع المشاريع التي أنجزها المطور؟",
        embedder=fake_embedder, store=five_projects_store, manager=manager,
    )
    assert result.lang == "ar"
    assert result.metrics["unique_entities_count"] == 5


def test_exhaustive_query_context_contains_all_five_project_names(five_projects_store, fake_embedder):
    captured_messages = []

    class CapturingProvider(FakeLLMProvider):
        def generate(self, messages):
            captured_messages.extend(messages)
            return super().generate(messages)

    gemini = CapturingProvider("gemini", reply="ok")
    manager = LLMProviderManager([gemini], cooldown_seconds=60)
    answer_question("List all his projects", embedder=fake_embedder, store=five_projects_store, manager=manager)

    user_message = next(m.content for m in captured_messages if m.role == "user")
    for name in ["AGEP", "ESTICAR", "WIREDWAVE", "SPEECHLY", "DESCRIPTOAI"]:
        assert name in user_message


def test_specific_query_about_one_project_is_not_exhaustive(five_projects_store, fake_embedder):
    manager, _ = _manager()
    result = answer_question(
        "Tell me about project AGEP.", embedder=fake_embedder, store=five_projects_store, manager=manager
    )
    assert result.metrics["intent"] == "specific"
    assert result.metrics["unique_entities_count"] is None
    # Retrieval classique : borné par retrieval_top_k, pas les 5 projets d'office.
    assert result.metrics["retrieved_chunks"] <= 8


def test_exhaustive_result_is_cached(five_projects_store, fake_embedder):
    # Réponse en anglais pour matcher la langue de la question : ce test
    # vérifie le cache, pas la validation de langue (couverte séparément par
    # test_language_guard.py) ; un mismatch déclencherait un retry légitime
    # qui fausserait le comptage d'appels ici.
    manager, gemini = _manager(reply="All five projects listed here.")
    q = "List all his projects"
    first = answer_question(q, embedder=fake_embedder, store=five_projects_store, manager=manager)
    assert first.metrics["cache_hit"] == "none"
    assert gemini.call_count == 1

    second = answer_question(q, embedder=fake_embedder, store=five_projects_store, manager=manager)
    assert second.metrics["cache_hit"] == "response"
    assert gemini.call_count == 1


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
    ingest_portfolio(JS_SAMPLE.encode("utf-8"), client=fake_embedder, store=tmp_store)
    return tmp_store


def _manager(reply="réponse générée"):
    gemini = FakeLLMProvider("gemini", reply=reply)
    return LLMProviderManager([gemini], cooldown_seconds=60), gemini


def test_answers_in_detected_language_fr(seeded_store, fake_embedder):
    manager, _ = _manager()
    result = answer_question("Quels projets ?", embedder=fake_embedder, store=seeded_store, manager=manager)
    assert result.lang == "fr"


def test_answers_in_detected_language_en(seeded_store, fake_embedder):
    manager, _ = _manager()
    result = answer_question("What is his experience?", embedder=fake_embedder, store=seeded_store, manager=manager)
    assert result.lang == "en"


def test_answers_in_detected_language_ar(seeded_store, fake_embedder):
    manager, _ = _manager()
    result = answer_question("ما هي مهارات إسالمو؟", embedder=fake_embedder, store=seeded_store, manager=manager)
    assert result.lang == "ar"


def test_cross_language_retrieval_finds_french_chunk_for_shared_topic(seeded_store, fake_embedder):
    # La question (n'importe quelle langue) contient "AGEP" : le chunk FR
    # "AGEP" doit être retrouvé même si la question est traitée comme EN,
    # car le retrieval ne filtre jamais par langue.
    manager, _ = _manager()
    result = answer_question("Tell me about AGEP project", embedder=fake_embedder, store=seeded_store, manager=manager)
    assert result.metrics["retrieved_chunks"] > 0


def test_protected_name_is_preserved_through_generation(seeded_store, fake_embedder):
    manager, _ = _manager(reply="⟦ISSALMOU_ADAAICHE⟧ est développeur.")
    result = answer_question("Qui est Issalmou Adaaiche ?", embedder=fake_embedder, store=seeded_store, manager=manager)
    assert "Issalmou Adaaiche" in result.response
    assert "⟦" not in result.response


def test_name_transliteration_is_corrected_even_if_llm_ignores_instruction(seeded_store, fake_embedder):
    # Simule un LLM qui, malgré la règle 6 du SYSTEM_INSTRUCTION, reproduit
    # quand même la translittération arabe présente dans le contexte source.
    # Le filet de sécurité déterministe (sanitize_name_transliterations) doit
    # corriger cela indépendamment de tout comportement LLM.
    manager, _ = _manager(reply="اسلمو إيدعيش هو مطور Full-Stack.")
    result = answer_question("من هو إسالمو؟", embedder=fake_embedder, store=seeded_store, manager=manager)
    assert "Issalmou Adaaiche" in result.response
    assert "اسلمو إيدعيش" not in result.response


def test_no_retrieved_chunks_returns_canned_message_without_calling_llm(tmp_store, fake_embedder):
    # Question clairement portfolio-liée (sujet "services" reconnu) mais sur
    # une base VIDE : doit tomber sur le message "information manquante",
    # pas sur le message hors-sujet (deux fallbacks distincts, voir
    # app/rag/scope.py) — d'où le choix d'une question topique ici plutôt
    # qu'une phrase sans sujet identifiable.
    manager, gemini = _manager()
    result = answer_question("Quels sont ses services ?", embedder=fake_embedder, store=tmp_store, manager=manager)
    assert gemini.call_count == 0
    assert "disponible" in result.response.lower()


def test_response_cache_hit_skips_embedding_retrieval_and_generation(seeded_store, fake_embedder):
    manager, gemini = _manager()
    q = "Quels projets ?"

    first = answer_question(q, embedder=fake_embedder, store=seeded_store, manager=manager)
    assert first.metrics["cache_hit"] == "none"
    calls_after_first = fake_embedder.call_count
    assert gemini.call_count == 1

    second = answer_question(q, embedder=fake_embedder, store=seeded_store, manager=manager)
    assert second.metrics["cache_hit"] == "response"
    assert second.response == first.response
    assert fake_embedder.call_count == calls_after_first  # aucun nouvel embedding
    assert gemini.call_count == 1  # aucun nouvel appel LLM


def test_new_question_after_cached_one_is_a_cache_miss(seeded_store, fake_embedder):
    manager, gemini = _manager()
    answer_question("Quels projets ?", embedder=fake_embedder, store=seeded_store, manager=manager)
    result = answer_question("Quelle est son expérience professionnelle ?", embedder=fake_embedder, store=seeded_store, manager=manager)
    assert result.metrics["cache_hit"] == "none"
    assert gemini.call_count == 2


def test_portfolio_update_invalidates_response_cache(seeded_store, fake_embedder):
    manager, gemini = _manager()
    q = "Quels projets ?"
    answer_question(q, embedder=fake_embedder, store=seeded_store, manager=manager)
    assert gemini.call_count == 1

    # Nouvelle version du portfolio -> le cache versionné doit être invalidé.
    updated_md = JS_SAMPLE.replace("Description du projet AGEP en français.", "Nouvelle description du projet AGEP en français.")
    ingest_portfolio(updated_md.encode("utf-8"), client=fake_embedder, store=seeded_store)

    result = answer_question(q, embedder=fake_embedder, store=seeded_store, manager=manager)
    # Le cache de réponse (versionné) est invalidé : pas de hit "response" sur
    # l'ancienne réponse. Le cache d'embedding, lui, n'est PAS versionné (c'est
    # voulu : l'embedding d'une question ne dépend pas du contenu du
    # portfolio) donc un hit "embedding" est normal et attendu ici.
    assert result.metrics["cache_hit"] != "response"
    assert gemini.call_count == 2  # la génération a bien été refaite (nouveau contenu)


def test_all_providers_unavailable_raises_dedicated_error(seeded_store, fake_embedder):
    gemini = FakeLLMProvider("gemini", error=LLMProviderError("down", ERROR_AUTH))
    manager = LLMProviderManager([gemini], cooldown_seconds=60)

    with pytest.raises(AllProvidersUnavailableError):
        answer_question("Quels projets ?", embedder=fake_embedder, store=seeded_store, manager=manager)
