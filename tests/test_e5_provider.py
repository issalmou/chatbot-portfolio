"""Tests du provider d'embedding E5 (via l'API Hugging Face Inference).

Deux catégories, explicitement séparées :
- TESTS MOCK (rapides, aucun appel réseau) : vérifient le format des préfixes
  E5 ("query: "/"passage: "), la forme du payload envoyé, et la gestion
  d'erreur HTTP.
- TESTS RÉELS (appellent la vraie API HF — nécessitent HF_TOKEN) : vérifient
  la dimension réelle, la normalisation réelle, et la qualité sémantique
  réelle en FR/EN/AR.
"""

from __future__ import annotations

import math

import httpx
import pytest

from app.config import settings
from app.embeddings.e5_provider import E5EmbeddingProvider, EmbeddingProviderError


class _FakeResponse:
    def __init__(self, status_code: int, json_data=None, text: str = ""):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text

    def json(self):
        if self._json_data is None:
            raise ValueError("no json")
        return self._json_data


# --- TESTS MOCK ---

def test_document_texts_get_passage_prefix(monkeypatch):
    captured = {}

    def fake_post(url, json, headers, timeout):
        captured["payload"] = json
        return _FakeResponse(200, json_data=[[1.0], [2.0]])

    monkeypatch.setattr(httpx, "post", fake_post)
    provider = E5EmbeddingProvider(api_token="fake-token")
    provider.embed_documents(["Issalmou Adaaiche est développeur.", "AGEP est un projet."])
    assert captured["payload"]["inputs"] == [
        "passage: Issalmou Adaaiche est développeur.",
        "passage: AGEP est un projet.",
    ]


def test_query_text_gets_query_prefix(monkeypatch):
    captured = {}

    def fake_post(url, json, headers, timeout):
        captured["payload"] = json
        return _FakeResponse(200, json_data=[[1.0]])

    monkeypatch.setattr(httpx, "post", fake_post)
    provider = E5EmbeddingProvider(api_token="fake-token")
    provider.embed_query("Quels sont ses projets ?")
    assert captured["payload"]["inputs"] == ["query: Quels sont ses projets ?"]


def test_url_uses_configured_model_id(monkeypatch):
    captured = {}

    def fake_post(url, json, headers, timeout):
        captured["url"] = url
        captured["headers"] = headers
        return _FakeResponse(200, json_data=[[1.0]])

    monkeypatch.setattr(httpx, "post", fake_post)
    provider = E5EmbeddingProvider(model_id="some/model", api_token="fake-token")
    provider.embed_query("x")
    assert captured["url"] == "https://router.huggingface.co/hf-inference/models/some/model/pipeline/feature-extraction"
    assert captured["headers"]["Authorization"] == "Bearer fake-token"


def test_missing_token_raises_without_network_call(monkeypatch):
    def fake_post(*args, **kwargs):
        raise AssertionError("ne devrait jamais être appelé sans token")

    monkeypatch.setattr(httpx, "post", fake_post)
    # "" (chaîne vide) explicite, distinct de None qui retomberait sur le
    # HF_TOKEN réel configuré dans l'environnement de test.
    provider = E5EmbeddingProvider(api_token="")
    with pytest.raises(EmbeddingProviderError):
        provider.embed_query("x")


def test_http_error_status_raises(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _FakeResponse(401, text="Invalid token"))
    provider = E5EmbeddingProvider(api_token="fake-token")
    with pytest.raises(EmbeddingProviderError):
        provider.embed_query("x")


def test_timeout_raises_embedding_error(monkeypatch):
    def fake_post(*a, **k):
        raise httpx.TimeoutException("timed out")

    monkeypatch.setattr(httpx, "post", fake_post)
    provider = E5EmbeddingProvider(api_token="fake-token")
    with pytest.raises(EmbeddingProviderError):
        provider.embed_query("x")


def test_network_error_raises_embedding_error(monkeypatch):
    def fake_post(*a, **k):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "post", fake_post)
    provider = E5EmbeddingProvider(api_token="fake-token")
    with pytest.raises(EmbeddingProviderError):
        provider.embed_query("x")


def test_response_shape_mismatch_raises(monkeypatch):
    # 2 textes envoyés mais 1 seul vecteur reçu : forme incohérente.
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _FakeResponse(200, json_data=[[1.0]]))
    provider = E5EmbeddingProvider(api_token="fake-token")
    with pytest.raises(EmbeddingProviderError):
        provider.embed_documents(["a", "b"])


def test_dimension_comes_from_settings_not_a_network_call(monkeypatch):
    def fake_post(*args, **kwargs):
        raise AssertionError("dimension ne doit jamais déclencher d'appel réseau")

    monkeypatch.setattr(httpx, "post", fake_post)
    provider = E5EmbeddingProvider(api_token="fake-token", dimension=768)
    assert provider.dimension == 768


def test_model_id_and_version_default_from_settings():
    provider = E5EmbeddingProvider()
    assert provider.model_id == settings.embedding_model_id
    assert provider.model_version == settings.embedding_model_version


# --- TESTS RÉELS (vraie API HF, aucun mock) ---

@pytest.fixture(scope="module")
def real_provider() -> E5EmbeddingProvider:
    return E5EmbeddingProvider()


@pytest.mark.skipif(not settings.hf_token, reason="Test réel : nécessite HF_TOKEN.")
def test_real_dimension_is_768(real_provider):
    vec = real_provider.embed_query("test")
    assert len(vec) == 768


@pytest.mark.skipif(not settings.hf_token, reason="Test réel : nécessite HF_TOKEN.")
def test_real_embeddings_are_normalized(real_provider):
    vec = real_provider.embed_query("Qui est Issalmou Adaaiche ?")
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-3


@pytest.mark.skipif(not settings.hf_token, reason="Test réel : nécessite HF_TOKEN.")
def test_real_multilingual_semantic_match_fr_en_ar(real_provider):
    """Une question FR/EN/AR sur la même personne doit être sémantiquement
    plus proche d'un passage pertinent que d'un passage sans rapport."""

    def cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb)

    relevant = real_provider.embed_documents(
        ["Issalmou Adaaiche est un développeur Full-Stack et Data Scientist basé au Maroc."]
    )[0]
    unrelated = real_provider.embed_documents(
        ["La recette de la tarte aux pommes nécessite des pommes, du sucre et de la pâte brisée."]
    )[0]

    for question in [
        "Qui est Issalmou Adaaiche ?",
        "Who is Issalmou Adaaiche?",
        "من هو اسلمو إيدعيش؟",
    ]:
        q_vec = real_provider.embed_query(question)
        assert cosine(q_vec, relevant) > cosine(q_vec, unrelated)
