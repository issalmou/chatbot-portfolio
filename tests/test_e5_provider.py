"""Tests du provider d'embedding E5.

Deux catégories, explicitement séparées :
- TESTS MOCK (rapides, aucun chargement de modèle) : vérifient le format des
  préfixes E5 ("query: "/"passage: ") et le cache singleton par model_id.
- TESTS RÉELS (chargent le vrai modèle intfloat/multilingual-e5-base — mis
  en cache par sentence-transformers après le premier téléchargement, donc
  rapides à partir de la 2e exécution) : vérifient la dimension réelle, la
  normalisation réelle, et la qualité sémantique réelle en FR/EN/AR.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.embeddings.e5_provider import E5EmbeddingProvider, _models


class _FakeSentenceTransformer:
    """Double minimal de sentence_transformers.SentenceTransformer."""

    def __init__(self, model_id, device=None):
        self.model_id = model_id
        self.device = device
        self.calls: list[list[str]] = []

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
        self.calls.append(list(texts))
        return np.array([[float(len(t))] for t in texts])

    def get_embedding_dimension(self):
        return 1


@pytest.fixture(autouse=True)
def _isolate_model_cache(monkeypatch):
    # Le cache de modèles est un dict de module partagé par tout le process
    # (voulu, pour éviter de recharger le vrai modèle à chaque instance) :
    # on l'isole pour les tests MOCK afin de ne jamais réutiliser un vrai
    # modèle déjà chargé par un test réel exécuté avant.
    saved = dict(_models)
    yield
    _models.clear()
    _models.update(saved)


# --- TESTS MOCK ---

def test_document_texts_get_passage_prefix(monkeypatch):
    fake = _FakeSentenceTransformer("fake-model")
    monkeypatch.setattr("app.embeddings.e5_provider._models", {"fake-model": fake})
    provider = E5EmbeddingProvider(model_id="fake-model")
    provider.embed_documents(["Issalmou Adaaiche est développeur.", "AGEP est un projet."])
    assert fake.calls[-1] == [
        "passage: Issalmou Adaaiche est développeur.",
        "passage: AGEP est un projet.",
    ]


def test_query_text_gets_query_prefix(monkeypatch):
    fake = _FakeSentenceTransformer("fake-model")
    monkeypatch.setattr("app.embeddings.e5_provider._models", {"fake-model": fake})
    provider = E5EmbeddingProvider(model_id="fake-model")
    provider.embed_query("Quels sont ses projets ?")
    assert fake.calls[-1] == ["query: Quels sont ses projets ?"]


def test_model_is_loaded_lazily_and_cached_per_model_id(monkeypatch):
    load_calls = []

    def _fake_ctor(model_id, device=None):
        load_calls.append(model_id)
        return _FakeSentenceTransformer(model_id, device)

    monkeypatch.setattr("app.embeddings.e5_provider._models", {})
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _fake_ctor)

    provider_a = E5EmbeddingProvider(model_id="model-a")
    provider_a.embed_query("x")
    provider_a.embed_query("y")
    provider_b = E5EmbeddingProvider(model_id="model-b")
    provider_b.embed_query("z")

    assert load_calls == ["model-a", "model-b"]  # un seul chargement par model_id


def test_dimension_is_measured_not_assumed(monkeypatch):
    fake = _FakeSentenceTransformer("fake-model")
    monkeypatch.setattr("app.embeddings.e5_provider._models", {"fake-model": fake})
    provider = E5EmbeddingProvider(model_id="fake-model")
    assert provider.dimension == 1  # valeur renvoyée par le faux modèle, pas une constante codée en dur


def test_model_id_and_version_default_from_settings():
    provider = E5EmbeddingProvider()
    from app.config import settings

    assert provider.model_id == settings.embedding_model_id
    assert provider.model_version == settings.embedding_model_version


# --- TESTS RÉELS (vrai modèle E5, aucun mock) ---

@pytest.fixture(scope="module")
def real_provider() -> E5EmbeddingProvider:
    return E5EmbeddingProvider()


def test_real_dimension_is_768(real_provider):
    assert real_provider.dimension == 768


def test_real_embeddings_are_normalized(real_provider):
    vec = real_provider.embed_query("Qui est Issalmou Adaaiche ?")
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-3


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
