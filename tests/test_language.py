import pytest

from app.language import (
    detect_language,
    protect_name,
    restore_name,
    sanitize_name_transliterations,
)


def test_detect_arabic_is_deterministic():
    assert detect_language("ما هي مهارات إسالمو؟") == "ar"


def test_detect_french_short_question():
    assert detect_language("Quels projets ?") == "fr"


def test_detect_english_short_question():
    assert detect_language("What are his skills?") == "en"


def test_detect_empty_defaults_to_fr():
    assert detect_language("") == "fr"
    assert detect_language("   ") == "fr"


# --- Hardening : questions à un seul mot-sujet (bug réel corrigé — voir
# rapport, langdetect n'est pas fiable sur un texte aussi court) ---

@pytest.mark.parametrize("query", ["Projects?", "Skills?", "Experience?", "Education?", "Contact?"])
def test_short_english_topic_words(query):
    assert detect_language(query) == "en"


@pytest.mark.parametrize("query", ["Projets ?", "Compétences ?", "Expérience ?", "Formation ?", "Contact ?"])
def test_short_french_topic_words(query):
    assert detect_language(query) == "fr"


def test_identical_spelling_word_disambiguated_by_french_typography():
    # "Contact" est orthographié à l'identique en FR/EN : seule la
    # typographie (espace avant "?", convention française) les distingue.
    assert detect_language("Contact?") == "en"
    assert detect_language("Contact ?") == "fr"


def test_mixed_sentence_with_english_tech_loanwords_stays_french():
    # Bug réel corrigé : langdetect seul classait cette phrase "en" à tort à
    # cause des emprunts techniques anglais, malgré une grammaire française
    # non ambiguë (quels/et/utilise-t-il).
    assert detect_language("Quels frameworks React et technologies backend utilise-t-il ?") == "fr"


def test_mixed_sentence_french_dominant_with_english_fragment():
    assert detect_language("Quels projects has he developed?") == "fr"


def test_arabic_stays_arabic_even_with_english_technical_terms():
    assert detect_language("ما هي technologies المستخدمة في مشروع React؟") == "ar"
    assert detect_language("ما هي technologies المستخدمة في his projects?") == "ar"


def test_protect_and_restore_roundtrip():
    text = "Issalmou Adaaiche est développeur."
    protected = protect_name(text)
    assert "Issalmou Adaaiche" not in protected
    assert restore_name(protected) == text


def test_sanitize_replaces_known_transliterations():
    for variant in ("اسلمو إيدعيش", "إسلمو إيدعيش", "إسلمو", "اسلمو"):
        result = sanitize_name_transliterations(f"{variant} هو مطور.")
        assert "Issalmou Adaaiche" in result
        assert variant not in result


def test_sanitize_longer_form_not_left_with_residue():
    result = sanitize_name_transliterations("إسلمو إيدعيش يعمل في الذكاء الاصطناعي.")
    assert result.count("Issalmou Adaaiche") == 1
    assert "إيدعيش" not in result


def test_sanitize_is_noop_on_already_correct_text():
    text = "Issalmou Adaaiche is a Full-Stack developer."
    assert sanitize_name_transliterations(text) == text
