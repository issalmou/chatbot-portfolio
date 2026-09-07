from app.rag.sections import (
    SECTION_ROUTES,
    detect_ambiguous,
    detect_topic,
    route_for_entity_type,
    route_for_topic,
)


def test_no_certification_route_exists():
    # Aucune page/route "Certifications" n'existe dans le frontend réel :
    # le sujet doit être reconnaissable (pour un fallback naturel) mais ne
    # doit JAMAIS résoudre vers une destination.
    assert "certification" not in SECTION_ROUTES
    assert detect_topic("Does he have any certifications?", "en") == "certification"
    assert route_for_topic("certification") is None
    assert route_for_topic(detect_topic("A-t-il des certifications ?", "fr")) is None
    assert route_for_topic(detect_topic("هل لديه شهادات؟", "ar")) is None


def test_skills_route_to_about_page_not_resume():
    # Constat vérifié dans le frontend réel : Skills vit sur /about, pas /resume.
    assert detect_topic("What are his skills?", "en") == "skills"
    assert route_for_topic("skills") == "/about#skills"


def test_projects_topic_routes_to_projects_page():
    assert detect_topic("Tell me about his projects", "en") == "projects"
    assert route_for_topic("projects") == "/projects"


def test_projects_topic_with_specific_entity_routes_to_its_detail_page():
    # Un projet précis identifié (entity_id) doit pointer vers sa page
    # dédiée (/project/<id>, route React Router réelle) plutôt que la liste
    # générale — sinon le LLM n'a aucune URL précise à proposer.
    assert route_for_topic("projects", project_entity_id="agep") == "/project/agep"


def test_projects_topic_without_entity_still_routes_to_general_list():
    assert route_for_topic("projects", project_entity_id=None) == "/projects"


def test_project_entity_id_ignored_for_other_topics():
    # project_entity_id n'a de sens que pour le topic "projects" : ignoré
    # ailleurs pour ne jamais produire une route incohérente.
    assert route_for_topic("skills", project_entity_id="agep") == "/about#skills"


def test_experience_and_education_both_route_to_resume():
    assert route_for_topic(detect_topic("What is his work experience?", "en")) == "/resume"
    assert route_for_topic(detect_topic("Where did he study?", "en")) == "/resume"
    assert route_for_topic(detect_topic("Quelle est sa formation ?", "fr")) == "/resume"


def test_contact_topic_routes_to_contact_page():
    assert route_for_topic(detect_topic("How can I contact him?", "en")) == "/contact"
    assert route_for_topic(detect_topic("Comment le contacter ?", "fr")) == "/contact"
    assert route_for_topic(detect_topic("كيف يمكنني التواصل معه؟", "ar")) == "/contact"


def test_unknown_topic_has_no_route():
    assert detect_topic("What's the weather like?", "en") is None
    assert route_for_topic(None) is None


def test_route_for_entity_type_matches_route_for_topic():
    assert route_for_entity_type("project") == "/projects"
    assert route_for_entity_type("skills_data") == "/about#skills"
    assert route_for_entity_type("internship") == "/resume"
    assert route_for_entity_type("education") == "/resume"
    assert route_for_entity_type(None) is None
    assert route_for_entity_type("unknown_type") is None


def test_ambiguous_anaphoric_question_without_topic():
    assert detect_ambiguous("Tell me more about it", "en") is True
    assert detect_ambiguous("Dis-m'en plus", "fr") is True
    assert detect_ambiguous("أخبرني أكثر", "ar") is True


def test_not_ambiguous_when_a_topic_is_identifiable():
    # Même formulée de façon vague, une question qui nomme un sujet réel
    # n'est jamais ambiguë : on peut répondre directement.
    assert detect_ambiguous("Tell me more about his projects", "en") is False


def test_not_ambiguous_when_no_anaphoric_marker_present():
    assert detect_ambiguous("What's the capital of Japan?", "en") is False
