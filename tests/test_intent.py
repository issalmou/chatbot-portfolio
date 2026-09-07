import pytest

from app.rag.intent import detect_list_intent, resolve_entity_type_keyword

# --- Questions EXHAUSTIVES attendues (intent = list_entities) ---

EXHAUSTIVE_EN = [
    "Give me all projects realized by the developer",
    "List all his projects",
    "What projects has he developed?",
    "Show me every project",
    "all projects",
    "all his projects",
    "list all projects",
    "projects he developed",
    "projects he realized",
    "What are all his skills?",
    "List all technologies he knows",
]

EXHAUSTIVE_FR = [
    "Donne-moi tous les projets réalisés par le développeur",
    "Quels sont tous ses projets ?",
    "Liste tous les projets qu'il a réalisés",
    "Quels projets a-t-il développés ?",
    "tous les projets",
    "tous ses projets",
    "liste des projets",
    "projets réalisés",
    "projets développés",
]

EXHAUSTIVE_AR = [
    "ما هي جميع المشاريع التي أنجزها المطور؟",
    "اذكر لي كل المشاريع التي قام بها",
    "ما هي كل مشاريعه؟",
    "جميع المشاريع",
    "كل المشاريع",
    "كل مشاريعه",
    "المشاريع التي أنجزها",
    "المشاريع التي طورها",
]

# --- Questions SPÉCIFIQUES attendues (intent = specific), non-régression ---

SPECIFIC_EN = [
    "Tell me about project X.",
    "What technologies were used in project X?",
    "What is project AGEP?",
    "Tell me about this project.",
]

SPECIFIC_FR = [
    "Parle-moi du projet X.",
    "Quelles technologies ont été utilisées ?",
    "Parle-moi de ce projet.",
]

SPECIFIC_AR = [
    "حدثني عن المشروع X.",
    "ما هي التقنيات المستخدمة؟",
]


@pytest.mark.parametrize("query", EXHAUSTIVE_EN)
def test_detects_exhaustive_intent_english(query):
    assert detect_list_intent(query, "en") is True


@pytest.mark.parametrize("query", EXHAUSTIVE_FR)
def test_detects_exhaustive_intent_french(query):
    assert detect_list_intent(query, "fr") is True


@pytest.mark.parametrize("query", EXHAUSTIVE_AR)
def test_detects_exhaustive_intent_arabic(query):
    assert detect_list_intent(query, "ar") is True


@pytest.mark.parametrize("query", SPECIFIC_EN)
def test_detects_specific_intent_english(query):
    assert detect_list_intent(query, "en") is False


@pytest.mark.parametrize("query", SPECIFIC_FR)
def test_detects_specific_intent_french(query):
    assert detect_list_intent(query, "fr") is False


@pytest.mark.parametrize("query", SPECIFIC_AR)
def test_detects_specific_intent_arabic(query):
    assert detect_list_intent(query, "ar") is False


def test_specific_reference_wins_over_bare_plural_noun_in_same_question():
    # "technologies" est pluriel mais la question cible un projet précis :
    # ne doit jamais devenir exhaustive à cause du pluriel seul.
    assert detect_list_intent("What technologies were used in project X?", "en") is False


def test_resolve_entity_type_keyword_projects():
    assert resolve_entity_type_keyword("Give me all projects", "en") == "project"
    assert resolve_entity_type_keyword("tous les projets", "fr") == "project"
    assert resolve_entity_type_keyword("جميع المشاريع", "ar") == "project"


def test_resolve_entity_type_keyword_skills():
    assert resolve_entity_type_keyword("What are all his skills?", "en") == "skills_data"
    assert resolve_entity_type_keyword("toutes ses compétences", "fr") == "skills_data"


def test_resolve_entity_type_keyword_ambiguous_returns_none():
    assert resolve_entity_type_keyword("Tell me everything about him", "en") is None


def test_resolve_entity_type_keyword_technologies_not_a_listable_entity():
    # "technologies" n'est pas indexé comme entité séparée (vit à l'intérieur
    # des chunks projet) : ne doit jamais être résolu comme un entity_type.
    assert resolve_entity_type_keyword("What technologies were used?", "en") is None
