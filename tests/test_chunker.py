from app.ingestion.chunker import chunk_content_blocks
from app.ingestion.parser import ContentBlock, derive_entity


def _block(lang="fr", section="about", subsection="intro", label="À propos", text="Contenu court.", topic_group=None):
    entity_type, entity_id = derive_entity(section, subsection)
    return ContentBlock(
        lang=lang, section=section, subsection=subsection, label=label, text=text,
        topic_group=topic_group or f"{section}:{subsection}",
        entity_type=entity_type, entity_id=entity_id,
    )


def test_chunk_id_stable_when_unrelated_block_is_added():
    blocks_before = [_block(section="about", subsection="intro"), _block(section="projects", subsection="agep", text="AGEP.")]
    blocks_after = [
        _block(section="resume", subsection="education.0", text="Nouvelle section."),
        _block(section="about", subsection="intro"),
        _block(section="projects", subsection="agep", text="AGEP."),
    ]
    id_before = next(c.chunk_id for c in chunk_content_blocks(blocks_before) if c.section == "projects")
    id_after = next(c.chunk_id for c in chunk_content_blocks(blocks_after) if c.section == "projects")
    assert id_before == id_after


def test_chunk_id_changes_when_subsection_key_changes():
    old = chunk_content_blocks([_block(section="projects", subsection="agep", text="AGEP.")])[0]
    new = chunk_content_blocks([_block(section="projects", subsection="agep-v2", text="AGEP.")])[0]
    assert old.chunk_id != new.chunk_id


def test_topic_group_is_carried_from_block_to_chunk():
    block = _block(section="projects", subsection="agep", topic_group="projects:agep")
    chunk = chunk_content_blocks([block])[0]
    assert chunk.topic_group == "projects:agep"


def test_breadcrumb_uses_human_label_not_machine_keys():
    block = _block(section="resume", subsection="education.0", label="Resume > Education", text="BSc, 2020-2023.")
    chunk = chunk_content_blocks([block])[0]
    assert chunk.text.startswith("[FR > Resume > Education]")
    assert "education.0" not in chunk.text


def test_long_paragraph_is_split_by_sentence_not_arbitrary_cutoff():
    long_text = ("Phrase numéro un très longue. " * 40).strip()
    block = _block(text=long_text)
    chunks = chunk_content_blocks([block])
    assert len(chunks) > 1
    for c in chunks:
        assert c.text.rstrip().endswith(".")


def test_tiny_trailing_chunk_is_merged_into_previous():
    text = ("Un paragraphe de taille normale qui décrit correctement le sujet en détail. " * 8) + "\n\nOk."
    block = _block(text=text)
    chunks = chunk_content_blocks([block])
    assert all(len(c.text) > 20 for c in chunks)


def test_chunk_hash_changes_when_content_changes():
    a = chunk_content_blocks([_block(text="Version A du contenu.")])[0]
    b = chunk_content_blocks([_block(text="Version B, différente.")])[0]
    assert a.chunk_hash != b.chunk_hash
    assert a.chunk_id == b.chunk_id  # même identité logique, contenu modifié
