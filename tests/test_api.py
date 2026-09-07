import pytest
from fastapi.testclient import TestClient

import main
from app.config import settings
from app.ingestion.pipeline import IngestionError, IngestionResult
from app.rag.retrieval import AllProvidersUnavailableError, AnswerResult
from app.vectorstore.chroma_store import ChromaUnavailableError

client = TestClient(main.app)


@pytest.fixture(autouse=True)
def _no_upload_token_by_default(monkeypatch):
    # La suite ne doit pas dépendre de la valeur réelle de CONTENT_UPLOAD_TOKEN
    # dans .env (protection ajoutée pour la production) : les tests qui
    # exercent spécifiquement l'authentification la reconfigurent eux-mêmes.
    monkeypatch.setattr(settings, "content_upload_token", None)


def test_chatbot_get_hello():
    response = client.get("/chatbot")
    assert response.status_code == 200
    assert "response" in response.json()


def test_chatbot_post_rejects_empty_query():
    response = client.post("/chatbot", json={"query": "   "})
    assert response.status_code == 400


def test_chatbot_post_returns_answer(monkeypatch):
    fake_result = AnswerResult(response="réponse test", lang="fr", metrics={"provider": "gemini", "total_ms": 1.0})
    monkeypatch.setattr(main, "answer_question", lambda query, conversation=None: fake_result)

    response = client.post("/chatbot", json={"query": "Quels projets ?"})

    assert response.status_code == 200
    body = response.json()
    assert body["response"] == "réponse test"
    assert body["lang"] == "fr"


def test_chatbot_post_returns_503_when_all_providers_down(monkeypatch):
    def _raise(query, conversation=None):
        raise AllProvidersUnavailableError("tous indisponibles")

    monkeypatch.setattr(main, "answer_question", _raise)

    response = client.post("/chatbot", json={"query": "Quels projets ?"})

    assert response.status_code == 503
    assert "clé" not in response.text.lower()  # aucune fuite de détail technique/clé


def test_upload_rejects_wrong_extension():
    response = client.post(
        "/upload-content",
        files={"file": ("portfolio.exe", b"contenu", "application/octet-stream")},
    )
    assert response.status_code == 400


def test_upload_rejects_oversized_file():
    huge = b"a" * (6 * 1024 * 1024)
    response = client.post("/upload-content", files={"file": ("translations.js", huge, "application/javascript")})
    assert response.status_code == 400


def test_upload_success_returns_ingestion_summary(monkeypatch):
    fake_result = IngestionResult(
        status="updated", source_hash="abc123", chunk_count=10,
        new_chunks=10, updated_chunks=0, unchanged_chunks=0, deleted_chunks=0,
    )
    monkeypatch.setattr(main, "ingest_portfolio", lambda raw, force=False: fake_result)

    response = client.post(
        "/upload-content",
        files={"file": ("translations.js", b"const translations = { en: { about: {} } };", "application/javascript")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "updated"
    assert body["chunk_count"] == 10


def test_upload_invalid_content_returns_422(monkeypatch):
    def _raise(raw, force=False):
        raise IngestionError("fichier invalide")

    monkeypatch.setattr(main, "ingest_portfolio", _raise)

    response = client.post("/upload-content", files={"file": ("translations.js", b"invalide", "application/javascript")})

    assert response.status_code == 422


def test_upload_returns_503_when_chroma_cloud_unavailable(monkeypatch):
    def _raise(raw, force=False):
        raise ChromaUnavailableError("Chroma Cloud injoignable")

    monkeypatch.setattr(main, "ingest_portfolio", _raise)

    response = client.post("/upload-content", files={"file": ("translations.js", b"const translations = {};", "application/javascript")})

    assert response.status_code == 503
    assert "api" not in response.text.lower() or "api_key" not in response.text.lower()


def test_upload_open_when_no_token_configured(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "content_upload_token", None)
    fake_result = IngestionResult(status="unchanged", source_hash="x", chunk_count=1, new_chunks=0, updated_chunks=0, unchanged_chunks=1, deleted_chunks=0)
    monkeypatch.setattr(main, "ingest_portfolio", lambda raw, force=False: fake_result)

    response = client.post("/upload-content", files={"file": ("translations.js", b"x", "application/javascript")})
    assert response.status_code == 200


def test_upload_rejects_missing_token_when_configured(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "content_upload_token", "secret-token")
    response = client.post("/upload-content", files={"file": ("translations.js", b"x", "application/javascript")})
    assert response.status_code == 401
    assert "secret-token" not in response.text


def test_upload_rejects_wrong_token(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "content_upload_token", "secret-token")
    response = client.post(
        "/upload-content",
        files={"file": ("translations.js", b"x", "application/javascript")},
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert response.status_code == 401


def test_upload_accepts_correct_token(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "content_upload_token", "secret-token")
    fake_result = IngestionResult(status="unchanged", source_hash="x", chunk_count=1, new_chunks=0, updated_chunks=0, unchanged_chunks=1, deleted_chunks=0)
    monkeypatch.setattr(main, "ingest_portfolio", lambda raw, force=False: fake_result)

    response = client.post(
        "/upload-content",
        files={"file": ("translations.js", b"x", "application/javascript")},
        headers={"Authorization": "Bearer secret-token"},
    )
    assert response.status_code == 200


def test_health_endpoint_never_leaks_api_keys():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert "llm" in body and "rag" in body
    assert "api_key" not in response.text.lower()
    from app.config import settings
    for key in (settings.gemini_api_key, settings.mistral_api_key, settings.groq_api_key, settings.chroma_api_key, settings.content_upload_token):
        if key:
            assert key not in response.text


def test_health_reports_chroma_connectivity_without_crashing():
    # Sans credentials Chroma Cloud configurés dans cet environnement de test,
    # /health doit répondre proprement avec chroma_connected=false plutôt que
    # de lever une exception.
    response = client.get("/health")
    assert response.status_code == 200
    assert "chroma_connected" in response.json()["rag"]
