import time

from app.cache.caches import (
    embedding_cache_key,
    response_cache_key,
    versioned_cache_key,
)
from app.cache.local_cache import TTLLRUCache


def test_lru_eviction_when_max_size_exceeded():
    cache: TTLLRUCache[str] = TTLLRUCache(max_size=2, ttl_seconds=60)
    cache.set("a", "1")
    cache.set("b", "2")
    cache.set("c", "3")  # doit évincer "a" (le plus ancien)
    assert cache.get("a") is None
    assert cache.get("b") == "2"
    assert cache.get("c") == "3"


def test_ttl_expiration():
    cache: TTLLRUCache[str] = TTLLRUCache(max_size=10, ttl_seconds=0.05)
    cache.set("a", "1")
    assert cache.get("a") == "1"
    time.sleep(0.1)
    assert cache.get("a") is None


def test_clear_empties_cache():
    cache: TTLLRUCache[str] = TTLLRUCache(max_size=10, ttl_seconds=60)
    cache.set("a", "1")
    cache.clear()
    assert cache.get("a") is None
    assert len(cache) == 0


def test_hit_miss_stats_tracked():
    cache: TTLLRUCache[str] = TTLLRUCache(max_size=10, ttl_seconds=60)
    cache.set("a", "1")
    cache.get("a")
    cache.get("missing")
    assert cache.stats["hits"] == 1
    assert cache.stats["misses"] == 1


def test_versioned_key_changes_when_content_version_changes():
    k1 = versioned_cache_key("Quels projets ?", "fr", "hash-v1")
    k2 = versioned_cache_key("Quels projets ?", "fr", "hash-v2")
    assert k1 != k2


def test_versioned_key_is_stable_for_equivalent_questions():
    k1 = versioned_cache_key("Quels projets ?", "fr", "hash-v1")
    k2 = versioned_cache_key("  quels projets ?  ", "fr", "hash-v1")
    assert k1 == k2


def test_embedding_key_independent_of_content_version():
    # L'embedding d'une question ne dépend pas du contenu du portfolio.
    k1 = embedding_cache_key("Quels projets ?", "fr", "e5-base", "1", kind="query")
    k2 = embedding_cache_key("Quels projets ?", "fr", "e5-base", "1", kind="query")
    assert k1 == k2


def test_embedding_key_changes_with_model():
    k1 = embedding_cache_key("Quels projets ?", "fr", "gemini-embedding-001", "1", kind="query")
    k2 = embedding_cache_key("Quels projets ?", "fr", "intfloat/multilingual-e5-base", "1", kind="query")
    assert k1 != k2  # jamais de collision entre deux espaces vectoriels différents


def test_embedding_key_changes_with_kind():
    k1 = embedding_cache_key("Issalmou Adaaiche", "fr", "e5-base", "1", kind="query")
    k2 = embedding_cache_key("Issalmou Adaaiche", "fr", "e5-base", "1", kind="passage")
    assert k1 != k2  # "query: " et "passage: " ne doivent jamais partager un embedding


def test_response_key_differs_per_provider_and_model():
    base = ("Quels projets ?", "fr", "hash-v1")
    k1 = response_cache_key(*base, provider="gemini", model="gemini-flash-latest")
    k2 = response_cache_key(*base, provider="mistral", model="mistral-small-latest")
    assert k1 != k2
