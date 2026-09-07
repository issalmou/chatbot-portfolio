import os

import pytest

from app.config import settings
from app.ingestion.parser import PortfolioParseError, parse_translations_js
from tests.conftest import SAMPLE_TRANSLATIONS_JS


def test_parses_all_three_languages():
    blocks = parse_translations_js(SAMPLE_TRANSLATIONS_JS)
    langs = {b.lang for b in blocks}
    assert langs == {"fr", "en", "ar"}


def test_extracts_real_content_sections():
    blocks = parse_translations_js(SAMPLE_TRANSLATIONS_JS)
    sections = {(b.section, b.subsection) for b in blocks if b.lang == "en"}
    assert ("about", "intro") in sections
    assert ("resume", "education.0") in sections
    assert ("resume", "skills_data") in sections
    assert ("resume", "internship.0") in sections
    assert ("services", "service.0") in sections
    assert ("services", "process") in sections
    assert ("projects", "agep") in sections  # shortName utilisé comme clé stable
    assert ("contact", "info") in sections
    assert ("home", "expertise") in sections


def test_excludes_ui_chrome_and_seo_noise():
    blocks = parse_translations_js(SAMPLE_TRANSLATIONS_JS)
    all_text = " ".join(b.text for b in blocks)
    # Ces chaînes n'existent QUE dans seo/nav/chatbot/notFound/formFields/images,
    # qui doivent être exclus du contenu indexé.
    assert "seomarker" not in all_text
    assert "Type your message" not in all_text
    assert "Not found" not in all_text
    assert "/img/agep.png" not in all_text
    for b in blocks:
        assert b.section not in {"seo", "nav", "chatbot", "notFound", "projectDetails"}


def test_project_uses_shortname_as_stable_subsection():
    blocks = parse_translations_js(SAMPLE_TRANSLATIONS_JS)
    agep = [b for b in blocks if b.section == "projects" and b.subsection == "agep"]
    assert len(agep) == 3  # une par langue


def test_project_details_labels_are_reused_per_language_not_hardcoded():
    blocks = parse_translations_js(SAMPLE_TRANSLATIONS_JS)
    en_project = next(b for b in blocks if b.lang == "en" and b.subsection == "agep")
    fr_project = next(b for b in blocks if b.lang == "fr" and b.subsection == "agep")
    assert "The Challenge" in en_project.text
    assert "Le défi" in fr_project.text


def test_topic_group_is_shared_across_languages_for_the_same_content():
    blocks = parse_translations_js(SAMPLE_TRANSLATIONS_JS)
    en_agep = next(b for b in blocks if b.lang == "en" and b.subsection == "agep")
    fr_agep = next(b for b in blocks if b.lang == "fr" and b.subsection == "agep")
    ar_agep = next(b for b in blocks if b.lang == "ar" and b.subsection == "agep")
    assert en_agep.topic_group == fr_agep.topic_group == ar_agep.topic_group == "projects:agep"


def test_protected_name_present_verbatim_in_every_language():
    blocks = parse_translations_js(SAMPLE_TRANSLATIONS_JS)
    for lang in ("fr", "en", "ar"):
        combined = " ".join(b.text for b in blocks if b.lang == lang)
        assert "Issalmou Adaaiche" in combined


def test_empty_file_raises():
    with pytest.raises(PortfolioParseError):
        parse_translations_js("")


def test_malformed_js_raises():
    with pytest.raises(PortfolioParseError):
        parse_translations_js("const translations = { en: { about: [ } };")


def test_file_without_recognized_language_raises():
    with pytest.raises(PortfolioParseError):
        parse_translations_js("const translations = { xx: { about: { title: 'hi' } } };")


def test_project_without_shortname_is_skipped_not_crashed():
    js = """
    const translations = { en: {
        about: { title: "About", headline: "H", description: "D", quote: "Q" },
        projects: { title: "Projects", projects_details: [
            { title: "No Slug", description: "desc without shortName" }
        ] }
    } };
    """
    blocks = parse_translations_js(js)
    assert not any(b.section == "projects" for b in blocks)
    assert any(b.section == "about" for b in blocks)


@pytest.mark.skipif(
    not settings.translations_js_path or not os.path.exists(settings.translations_js_path),
    reason="translations.js réel non accessible sur cette machine",
)
def test_parses_the_real_translations_js_file():
    with open(settings.translations_js_path, "r", encoding="utf-8") as f:
        raw = f.read()
    blocks = parse_translations_js(raw)
    assert {b.lang for b in blocks} == {"fr", "en", "ar"}
    short_names = {b.subsection for b in blocks if b.section == "projects"}
    assert {"agep", "esticar", "wiredwave", "speechly", "descriptoai"}.issubset(short_names)
    for lang in ("fr", "en"):
        intro = next(b for b in blocks if b.lang == lang and b.subsection == "intro")
        assert "Issalmou Adaaiche" in intro.text
    # Constat réel (voir rapport) : le bloc AR de translations.js translittère
    # le nom en arabe (ex. "اسلمو إيدعيش") au lieu du latin "Issalmou Adaaiche"
    # dans son propre contenu narratif — c'est un choix déjà présent dans le
    # fichier frontend, que ce parser ne modifie pas (il ne fait qu'extraire).
    # La protection du nom (app/language.py) agit sur les occurrences LATINES
    # existantes ; voir aussi la règle 6 du SYSTEM_INSTRUCTION dans
    # app/rag/retrieval.py qui demande explicitement au LLM de restituer la
    # forme latine si le contexte contient une translittération.


@pytest.mark.skipif(
    not settings.translations_js_path or not os.path.exists(settings.translations_js_path),
    reason="translations.js réel non accessible sur cette machine",
)
def test_real_file_exhaustive_project_retrieval_matches_actual_project_count():
    """Test CRITIQUE (données réelles, pas un nombre codé en dur) : le nombre
    de project_id uniques retrouvés pour une question exhaustive doit être
    EXACTEMENT le nombre réel de projets présents dans translations.js — que
    ce nombre soit 5 aujourd'hui ou différent demain."""
    from app.ingestion.chunker import chunk_content_blocks
    from tests.conftest import FakeChromaStore, FakeEmbedder

    with open(settings.translations_js_path, "r", encoding="utf-8") as f:
        raw = f.read()
    blocks = parse_translations_js(raw)

    # Nombre réel de projets, dérivé des données sources elles-mêmes (une
    # entrée par shortName distinct dans le bloc EN, langue toujours présente).
    real_project_ids = {b.entity_id for b in blocks if b.entity_type == "project" and b.lang == "en"}
    assert len(real_project_ids) > 0, "aucun projet détecté dans le fichier réel : le test n'a plus de sens"

    chunks = chunk_content_blocks(blocks)
    store = FakeChromaStore()
    embedder = FakeEmbedder()
    to_upsert = [
        {
            "id": c.chunk_id, "document": c.text, "embedding": embedder.embed_documents([c.text])[0],
            "metadata": {
                "lang": c.lang, "section": c.section, "subsection": c.subsection, "part_index": c.part_index,
                "chunk_hash": c.chunk_hash, "topic_group": c.topic_group,
                "entity_type": c.entity_type, "entity_id": c.entity_id,
            },
        }
        for c in chunks
    ]
    store.sync(to_upsert=to_upsert, ids_to_delete=[])

    merged, _raw_count = store.get_all_by_entity_type("project", preferred_lang="en")
    found_ids = {item["entity_id"] for item in merged}

    assert found_ids == real_project_ids
    assert len(found_ids) == len(real_project_ids)  # aucun projet écrasé par un autre
