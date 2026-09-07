"""Tests de la classification LOCALE de périmètre (app/rag/scope.py) :
PORTFOLIO_RELATED / AMBIGUOUS / OUT_OF_SCOPE — aucun appel LLM.
"""

import pytest

from app.language import detect_language
from app.rag.intent import detect_list_intent
from app.rag.memory import resolve_reference
from app.rag.scope import Scope, classify_scope, detect_identity_question
from app.rag.sections import detect_topic


def _classify(query: str, lang: str | None = None, conversation=None) -> Scope:
    lang = lang or detect_language(query)
    resolution = resolve_reference(query, lang, conversation)
    is_exhaustive = detect_list_intent(query, lang)
    topic = detect_topic(query, lang) or resolution.focus_topic
    return classify_scope(query, lang, topic, is_exhaustive, resolution)


# --- PORTFOLIO_RELATED (exemples fournis) ---

PORTFOLIO_RELATED_EN = [
    "What projects has he built?",
    "What technologies does he know?",
    "Tell me about his experience.",
    "What did he study?",
    "How can I contact him?",
    "What services does he offer?",
    "What are his skills?",
]

PORTFOLIO_RELATED_FR = [
    "Quels sont ses projets ?",
    "Quelles technologies utilise-t-il ?",
    "Parle-moi de son expérience.",
    "Quelles études a-t-il faites ?",
    "Comment puis-je le contacter ?",
    "Quels services propose-t-il ?",
]

PORTFOLIO_RELATED_AR = [
    "ما هي مشاريعه؟",
    "ما هي التقنيات التي يعرفها؟",
    "حدثني عن خبرته.",
    "ماذا درس؟",
    "كيف يمكنني التواصل معه؟",
    "ما هي الخدمات التي يقدمها؟",
]


@pytest.mark.parametrize("query", PORTFOLIO_RELATED_EN)
def test_portfolio_related_english(query):
    assert _classify(query, "en") == Scope.PORTFOLIO_RELATED


@pytest.mark.parametrize("query", PORTFOLIO_RELATED_FR)
def test_portfolio_related_french(query):
    assert _classify(query, "fr") == Scope.PORTFOLIO_RELATED


@pytest.mark.parametrize("query", PORTFOLIO_RELATED_AR)
def test_portfolio_related_arabic(query):
    assert _classify(query, "ar") == Scope.PORTFOLIO_RELATED


# --- OUT_OF_SCOPE (exemples fournis) ---

OUT_OF_SCOPE_EN = [
    "What is the capital of Japan?",
    "Write me a Python game.",
    "What's the weather today?",
    "Who won the World Cup?",
    "Explain quantum physics.",
    "Tell me a joke.",
]


@pytest.mark.parametrize("query", OUT_OF_SCOPE_EN)
def test_out_of_scope_english(query):
    assert _classify(query, "en") == Scope.OUT_OF_SCOPE


# --- Questions courtes : ne doivent JAMAIS être hors-sujet ---

SHORT_PORTFOLIO_EN = ["Projects?", "Skills?", "Experience?", "Education?", "Contact?"]
SHORT_PORTFOLIO_FR = ["Projets ?", "Compétences ?", "Expérience ?", "Formation ?", "Contact ?"]
SHORT_PORTFOLIO_AR = ["المشاريع؟", "المهارات؟", "الخبرة؟", "الدراسة؟", "التواصل؟"]


@pytest.mark.parametrize("query", SHORT_PORTFOLIO_EN)
def test_short_portfolio_query_english(query):
    assert _classify(query, "en") == Scope.PORTFOLIO_RELATED


@pytest.mark.parametrize("query", SHORT_PORTFOLIO_FR)
def test_short_portfolio_query_french(query):
    assert _classify(query, "fr") == Scope.PORTFOLIO_RELATED


@pytest.mark.parametrize("query", SHORT_PORTFOLIO_AR)
def test_short_portfolio_query_arabic(query):
    assert _classify(query, "ar") == Scope.PORTFOLIO_RELATED


# --- Piège de culture générale : un terme technique du portfolio dans une
# question générale ne la rend PAS portfolio-liée ---

def test_general_knowledge_trap_react():
    assert _classify("What is React?", "en") == Scope.OUT_OF_SCOPE


def test_general_knowledge_trap_python():
    assert _classify("What is Python?", "en") == Scope.OUT_OF_SCOPE


# --- Mais une question CONTEXTUALISÉE sur ce même terme reste portfolio-liée ---

def test_contextual_react_question_is_portfolio_related():
    assert _classify("What projects use React?", "en") == Scope.PORTFOLIO_RELATED


def test_contextual_react_question_with_possessive_is_portfolio_related():
    assert _classify("Which of his projects use React?", "en") == Scope.PORTFOLIO_RELATED


# --- Ambiguïté : reste AMBIGUOUS uniquement en l'absence de toute conversation ---

def test_ambiguous_without_any_context():
    assert _classify("Tell me more about it", "en", conversation=[]) == Scope.AMBIGUOUS


# --- Question sur l'identité de l'assistant lui-même (pas sur Issalmou) ---

IDENTITY_QUESTIONS_EN = [
    "Who am I talking to?",
    "Who are you?",
    "What's your name?",
    "What is your name?",
]

IDENTITY_QUESTIONS_FR = [
    "Avec qui je parle ?",
    "Qui es-tu ?",
    "Qui êtes-vous ?",
    "Comment tu t'appelles ?",
    "Quel est ton nom ?",
]

IDENTITY_QUESTIONS_AR = [
    "من أنت؟",
    "مع من أتحدث؟",
    "ما اسمك؟",
]


@pytest.mark.parametrize("query", IDENTITY_QUESTIONS_EN)
def test_identity_question_detected_english(query):
    assert detect_identity_question(query, "en") is True


@pytest.mark.parametrize("query", IDENTITY_QUESTIONS_FR)
def test_identity_question_detected_french(query):
    assert detect_identity_question(query, "fr") is True


@pytest.mark.parametrize("query", IDENTITY_QUESTIONS_AR)
def test_identity_question_detected_arabic(query):
    assert detect_identity_question(query, "ar") is True


# --- Piège : une question sur ISSALMOU (3e personne) n'est jamais une
# question d'identité de l'assistant (2e personne) ---

def test_question_about_issalmou_is_not_an_identity_question_english():
    assert detect_identity_question("Who is Issalmou?", "en") is False


def test_question_about_issalmou_is_not_an_identity_question_french():
    assert detect_identity_question("Qui est Issalmou ?", "fr") is False
    assert detect_identity_question("Quel est son nom ?", "fr") is False


def test_question_about_issalmou_is_not_an_identity_question_arabic():
    assert detect_identity_question("من هو إسلمو؟", "ar") is False
