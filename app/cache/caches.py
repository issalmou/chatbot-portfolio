"""Instances de cache partagées par l'application + construction des clés.

Trois caches, avec des clés et des durées de vie volontairement différentes :

- embedding_cache : clé = texte normalisé + langue. Indépendant de la
  version du portfolio (l'embedding d'une question ne dépend pas du
  contenu du RAG), donc PAS invalidé lors d'une mise à jour du portfolio.
- retrieval_cache / response_cache : clé = texte normalisé + langue +
  version du contenu (source_hash). Une mise à jour du portfolio change la
  version -> les clés changent -> les anciennes entrées deviennent
  inatteignables (et sont explicitement purgées pour libérer la mémoire
  immédiatement, sans attendre le TTL).
"""

from __future__ import annotations

import hashlib
import re

from app.cache.local_cache import TTLLRUCache
from app.config import settings

embedding_cache: TTLLRUCache[list[float]] = TTLLRUCache(
    max_size=settings.embedding_cache_max_size,
    ttl_seconds=settings.embedding_cache_ttl_seconds,
)
retrieval_cache: TTLLRUCache[list[dict]] = TTLLRUCache(
    max_size=settings.retrieval_cache_max_size,
    ttl_seconds=settings.retrieval_cache_ttl_seconds,
)
response_cache: TTLLRUCache[str] = TTLLRUCache(
    max_size=settings.response_cache_max_size,
    ttl_seconds=settings.response_cache_ttl_seconds,
)


def normalize_question(question: str) -> str:
    """Normalisation légère : espaces, casse -> maximise les cache hits
    sur des questions équivalentes (ex. "Quels projets ?" / "quels projets?")."""
    return re.sub(r"\s+", " ", question.strip().lower())


def embedding_cache_key(
    text: str, lang: str, model_id: str, model_version: str, kind: str = "query"
) -> str:
    """Clé de cache d'embedding : modèle + version + type (query/passage) +
    langue + texte normalisé. Un changement de modèle d'embedding (ex.
    Gemini -> E5) ou de logique d'encodage (model_version) produit donc
    automatiquement des clés totalement différentes — aucune collision
    possible entre deux espaces vectoriels incompatibles."""
    normalized = normalize_question(text)
    raw = f"{model_id}|{model_version}|{kind}|{lang}|{normalized}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def versioned_cache_key(question: str, lang: str, content_version: str) -> str:
    """Clé pour le retrieval : dépend du contenu (source_version) mais pas du
    provider LLM, puisque le retrieval ne dépend que de ChromaDB + l'embedding."""
    normalized = normalize_question(question)
    return hashlib.sha256(
        f"{normalized}|{lang}|{content_version}".encode("utf-8")
    ).hexdigest()


def response_cache_key(question: str, lang: str, content_version: str, provider: str, model: str) -> str:
    """Clé pour la réponse générée : inclut provider+model en plus de la
    version du contenu. Deux providers différents (ou deux modèles) peuvent
    produire une formulation différente à partir du même contexte RAG ; on
    évite donc de servir la réponse d'un provider à la place d'un autre, par
    fiabilité, plutôt que de supposer qu'elles sont interchangeables."""
    normalized = normalize_question(question)
    return hashlib.sha256(
        f"{normalized}|{lang}|{content_version}|{provider}|{model}".encode("utf-8")
    ).hexdigest()


def invalidate_versioned_caches() -> None:
    """Appelé après une ingestion réussie : les anciennes réponses/retrievals
    ne correspondent plus au nouveau contenu, donc on les purge immédiatement
    plutôt que d'attendre leur expiration naturelle (TTL)."""
    retrieval_cache.clear()
    response_cache.clear()
