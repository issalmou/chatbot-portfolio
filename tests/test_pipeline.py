import pytest

from app.config import settings
from app.ingestion.pipeline import IngestionError, ingest_portfolio
from app.vectorstore.chroma_store import ChromaStore

JS_V1 = """
const translations = { en: {
    about: { title: "About", headline: "Issalmou Adaaiche.", description: "Developer.", quote: "Quote." },
    projects: { title: "Projects", projects_details: [
        { title: "AGEP", shortName: "agep", date: "2024", type: "Web",
          description: "Application de gestion.", theChallenge: "x", theSolution: "y",
          keyFeatures: ["f1"], technologies: ["Laravel"] }
    ] },
    projectDetails: { theChallenge: "Challenge", theSolution: "Solution", keyFeatures: "Key Features" },
} };
"""

JS_V1_MODIFIED_AGEP = JS_V1.replace("Application de gestion.", "Application de gestion des équipes, v2.")

JS_V1_AGEP_REMOVED = """
const translations = { en: {
    about: { title: "About", headline: "Issalmou Adaaiche.", description: "Developer.", quote: "Quote." },
} };
"""


def test_valid_file_creates_chunks(tmp_store, fake_embedder):
    result = ingest_portfolio(JS_V1.encode("utf-8"), client=fake_embedder, store=tmp_store)
    assert result.status == "updated"
    assert result.chunk_count == 2  # about + projects.agep
    assert result.new_chunks == 2
    assert tmp_store.count() == 2


def test_empty_file_is_rejected(tmp_store, fake_embedder):
    with pytest.raises(IngestionError):
        ingest_portfolio(b"", client=fake_embedder, store=tmp_store)
    assert tmp_store.count() == 0


def test_invalid_file_without_language_sections_is_rejected(tmp_store, fake_embedder):
    with pytest.raises(IngestionError):
        ingest_portfolio(b"const x = { xx: { about: {} } };", client=fake_embedder, store=tmp_store)
    assert tmp_store.count() == 0


def test_malformed_js_is_rejected(tmp_store, fake_embedder):
    with pytest.raises(IngestionError):
        ingest_portfolio(b"const translations = { en: { about: [ } };", client=fake_embedder, store=tmp_store)


def test_bad_encoding_is_rejected(tmp_store, fake_embedder):
    invalid_bytes = "café".encode("latin-1")  # invalide en UTF-8 strict
    with pytest.raises(IngestionError, match="[Ee]ncodage"):
        ingest_portfolio(invalid_bytes, client=fake_embedder, store=tmp_store)


def test_reingesting_identical_file_is_a_noop(tmp_store, fake_embedder):
    ingest_portfolio(JS_V1.encode("utf-8"), client=fake_embedder, store=tmp_store)
    assert fake_embedder.call_count == 1

    result = ingest_portfolio(JS_V1.encode("utf-8"), client=fake_embedder, store=tmp_store)
    assert result.status == "unchanged"
    assert fake_embedder.call_count == 1  # aucun nouvel appel d'embedding


def test_force_refreshes_metadata_of_unchanged_content_without_reembedding(tmp_store, fake_embedder):
    # Bug réel corrigé : un changement de SCHÉMA de métadonnées (ex. ajout de
    # entity_type/entity_id) sur un translations.js dont le CONTENU texte n'a
    # pas changé était auparavant totalement ignoré par le court-circuit
    # "fichier source identique" (le hash source ne change pas). `force=True`
    # doit rafraîchir les métadonnées de chaque chunk sans aucun nouvel appel
    # d'embedding (le texte étant inchangé, l'embedding déjà stocké est réutilisé).
    ingest_portfolio(JS_V1.encode("utf-8"), client=fake_embedder, store=tmp_store)
    assert fake_embedder.call_count == 1

    without_force = ingest_portfolio(JS_V1.encode("utf-8"), client=fake_embedder, store=tmp_store)
    assert without_force.status == "unchanged"

    result = ingest_portfolio(JS_V1.encode("utf-8"), client=fake_embedder, store=tmp_store, force=True)
    assert result.status == "updated"
    assert result.new_chunks == 0
    assert result.updated_chunks == 0
    assert result.unchanged_chunks == 2  # about + projects.agep, métadonnées rafraîchies
    assert fake_embedder.call_count == 1  # aucun nouvel appel d'embedding


def test_modified_chunk_is_reembedded_others_are_not(tmp_store, fake_embedder):
    ingest_portfolio(JS_V1.encode("utf-8"), client=fake_embedder, store=tmp_store)
    result = ingest_portfolio(JS_V1_MODIFIED_AGEP.encode("utf-8"), client=fake_embedder, store=tmp_store)
    assert result.status == "updated"
    assert result.updated_chunks == 1  # projects.agep modifié
    assert result.new_chunks == 0
    assert result.unchanged_chunks == 1  # about inchangé


def test_removed_section_deletes_its_chunk(tmp_store, fake_embedder):
    ingest_portfolio(JS_V1.encode("utf-8"), client=fake_embedder, store=tmp_store)
    assert tmp_store.count() == 2

    result = ingest_portfolio(JS_V1_AGEP_REMOVED.encode("utf-8"), client=fake_embedder, store=tmp_store)
    assert result.deleted_chunks == 1
    assert tmp_store.count() == 1


def test_embedding_failure_rolls_back_and_keeps_old_version(tmp_store, fake_embedder):
    ingest_portfolio(JS_V1.encode("utf-8"), client=fake_embedder, store=tmp_store)
    version_before = tmp_store.get_content_version()
    count_before = tmp_store.count()

    class FailingEmbedder:
        def embed_documents(self, texts):
            raise ConnectionError("Modèle d'embedding indisponible (simulation de test).")

    with pytest.raises(IngestionError):
        ingest_portfolio(JS_V1_MODIFIED_AGEP.encode("utf-8"), client=FailingEmbedder(), store=tmp_store)

    assert tmp_store.get_content_version() == version_before
    assert tmp_store.count() == count_before


@pytest.mark.skipif(
    not settings.chroma_api_key,
    reason="Test d'intégration réel : nécessite CHROMA_API_KEY (Chroma Cloud).",
)
def test_real_chroma_cloud_roundtrip():
    """TEST RÉEL (aucun mock) : connexion Chroma Cloud + embeddings E5
    réels (modèle local, chargé une fois). Utilise une collection DÉDIÉE aux
    tests (jamais `portfolio_rag_e5`, la collection de production), pour ne
    jamais écraser les vraies données — les chunk_id sont déterministes et
    collisionneraient avec les chunks réels si on utilisait la même
    collection."""
    from app.embeddings.e5_provider import e5_embedding_provider

    store = ChromaStore(collection_name="portfolio_rag_e5_test_integration")
    assert store.is_available

    result = ingest_portfolio(JS_V1.encode("utf-8"), client=e5_embedding_provider, store=store)
    assert result.status in ("updated", "unchanged")
    assert store.count() >= 1

    # Retrieval réel de bout en bout (embedding E5 réel + Chroma Cloud query).
    query_embedding = e5_embedding_provider.embed_query("AGEP project")
    retrieved = store.query(query_embedding, top_k=5)
    assert len(retrieved) >= 1

    # Ré-ingestion identique : doit être un no-op (aucun nouvel embedding).
    class FailIfCalled:
        def embed_documents(self, texts):
            raise AssertionError("ne devrait pas être appelé : hash inchangé")

    unchanged_result = ingest_portfolio(JS_V1.encode("utf-8"), client=FailIfCalled(), store=store)
    assert unchanged_result.status == "unchanged"
