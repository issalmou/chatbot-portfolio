"""Wrapper Chroma Cloud : synchronisation par diff, recherche, versioning.

Le reste de l'application ne manipule jamais directement le client Chroma :
tout passe par cette classe. Aucune persistance locale — la version courante
du contenu (source_hash) est stockée dans la collection elle-même, sous un
document réservé (`__source_meta__`), pour survivre à un redéploiement sans
disque persistant."""

from __future__ import annotations

from datetime import datetime, timezone

import chromadb

from app.config import settings

_META_DOC_ID = "__source_meta__"
_KIND_CHUNK = "chunk"
_KIND_META = "meta"


class ChromaUnavailableError(Exception):
    """Chroma Cloud n'est pas configuré ou injoignable."""


class ChromaStore:
    def __init__(
        self,
        api_key: str | None = None,
        tenant: str | None = None,
        database: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        self._client = None
        self._collection = None
        try:
            self._client = chromadb.CloudClient(
                tenant=tenant or settings.chroma_tenant,
                database=database or settings.chroma_database,
                api_key=api_key or settings.chroma_api_key,
            )
            self._collection = self._client.get_or_create_collection(
                name=collection_name or settings.chroma_collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            # Ne doit jamais faire planter le démarrage : /chatbot dégrade en 503,
            # /health signale l'indisponibilité, jamais la clé API dans le message.
            print(f"ATTENTION: connexion Chroma Cloud impossible ({type(exc).__name__}).")

    @property
    def is_available(self) -> bool:
        return self._collection is not None

    def _require_collection(self):
        if self._collection is None:
            raise ChromaUnavailableError(
                "Chroma Cloud non configuré ou injoignable (vérifier CHROMA_API_KEY / "
                "CHROMA_TENANT / CHROMA_DATABASE)."
            )
        return self._collection

    # --- Versioning (document réservé dans la collection elle-même) ---

    def get_meta(self) -> dict:
        collection = self._require_collection()
        result = collection.get(ids=[_META_DOC_ID], include=["metadatas"])
        metadatas = result.get("metadatas") or []
        return dict(metadatas[0]) if metadatas else {}

    def get_content_version(self) -> str | None:
        return self.get_meta().get("source_hash")

    def commit_new_version(self, source_hash: str, chunk_count: int) -> None:
        collection = self._require_collection()
        dim = settings.embedding_dimension
        filler_embedding = [1.0] + [0.0] * (dim - 1)  # non nul, jamais utilisé en recherche
        collection.upsert(
            ids=[_META_DOC_ID],
            documents=["source_meta"],
            embeddings=[filler_embedding],
            metadatas=[
                {
                    "kind": _KIND_META,
                    "source_hash": source_hash,
                    "chunk_count": chunk_count,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            ],
        )

    # --- Diff / synchronisation ---

    def existing_chunk_hashes(self) -> dict[str, str]:
        """id -> chunk_hash pour tous les chunks réels (exclut le document de méta)."""
        collection = self._require_collection()
        result = collection.get(where={"kind": _KIND_CHUNK}, include=["metadatas"])
        hashes: dict[str, str] = {}
        for doc_id, metadata in zip(result["ids"], result["metadatas"] or []):
            if metadata and "chunk_hash" in metadata:
                hashes[doc_id] = metadata["chunk_hash"]
        return hashes

    def get_embeddings(self, ids: list[str]) -> dict[str, list[float]]:
        """Récupère les embeddings déjà calculés pour des chunks inchangés,
        afin de rafraîchir leurs métadonnées dans Chroma Cloud sans
        réappeler Gemini."""
        if not ids:
            return {}
        collection = self._require_collection()
        result = collection.get(ids=ids, include=["embeddings"])
        return {doc_id: list(emb) for doc_id, emb in zip(result["ids"], result["embeddings"])}

    def sync(self, to_upsert: list[dict], ids_to_delete: list[str]) -> None:
        """Applique le diff. Appelé uniquement après que tous les embeddings
        nécessaires ont été obtenus (voir ingestion/pipeline.py) : seule
        étape qui mute Chroma Cloud."""
        collection = self._require_collection()
        if to_upsert:
            collection.upsert(
                ids=[u["id"] for u in to_upsert],
                documents=[u["document"] for u in to_upsert],
                embeddings=[u["embedding"] for u in to_upsert],
                metadatas=[{**u["metadata"], "kind": _KIND_CHUNK} for u in to_upsert],
            )
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)

    # --- Recherche ---

    def query(self, embedding: list[float], top_k: int) -> list[dict]:
        """Recherche multilingue avec déduplication par sujet (topic_group).
        Chaque sujet existe en 3 exemplaires (fr/en/ar) à l'embedding très
        proche : sans dédup, un même sujet saturerait le top_k. On élargit
        donc la recherche brute puis on ne garde que le meilleur chunk par
        sujet, indépendamment de sa langue."""
        collection = self._require_collection()
        count = self.count()
        if count == 0:
            return []

        candidate_pool = min(max(top_k * 4, top_k), count)
        result = collection.query(
            query_embeddings=[embedding],
            n_results=candidate_pool,
            where={"kind": _KIND_CHUNK},
            include=["documents", "metadatas", "distances"],
        )
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        deduped: list[dict] = []
        seen_topics: set[tuple] = set()
        for doc, meta, dist in zip(documents, metadatas, distances):
            topic = meta.get("topic_group")
            if topic in seen_topics:
                continue
            seen_topics.add(topic)
            deduped.append({"document": doc, "metadata": meta, "distance": dist})
            if len(deduped) >= top_k:
                break

        return deduped

    def get_all_by_entity_type(self, entity_type: str, preferred_lang: str | None = None) -> tuple[list[dict], int]:
        """Récupère TOUTES les entités d'un type (filtrage metadata exact,
        pas de similarité) — pour les questions exhaustives, où l'objectif
        est la complétude. Chaque entité existe en plusieurs langues/parties :
        on ne garde qu'une version par entité (langue demandée si disponible),
        parties recollées dans l'ordre, pour ne jamais envoyer le même projet
        plusieurs fois au LLM."""
        collection = self._require_collection()
        result = collection.get(
            where={"$and": [{"kind": {"$eq": _KIND_CHUNK}}, {"entity_type": {"$eq": entity_type}}]},
            include=["documents", "metadatas"],
        )
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []

        # entity_id -> lang -> {part_index: (document, metadata)}
        by_entity: dict[str, dict[str, dict[int, tuple[str, dict]]]] = {}
        for doc, meta in zip(documents, metadatas):
            entity_id = meta.get("entity_id")
            if not entity_id:
                continue
            lang = meta.get("lang", "")
            part_index = meta.get("part_index", 0)
            by_entity.setdefault(entity_id, {}).setdefault(lang, {})[part_index] = (doc, meta)

        lang_priority = [preferred_lang] if preferred_lang else []
        lang_priority += [l for l in ("en", "fr", "ar") if l != preferred_lang]

        merged: list[dict] = []
        for entity_id, by_lang in by_entity.items():
            chosen_lang = next((l for l in lang_priority if l in by_lang), next(iter(by_lang)))
            parts = by_lang[chosen_lang]
            ordered_docs = [parts[idx][0] for idx in sorted(parts)]
            representative_meta = parts[min(parts)][1]
            merged.append(
                {
                    "document": "\n".join(ordered_docs),
                    "metadata": representative_meta,
                    "entity_id": entity_id,
                }
            )
        return merged, len(documents)

    def get_by_entity_id(self, entity_id: str, preferred_lang: str | None = None) -> dict | None:
        """Récupère UNE entité précise par son identifiant exact (filtrage
        metadata, pas de similarité). Utilisé par la résolution de référence
        conversationnelle (app/rag/memory.py) : un lookup direct est plus
        fiable qu'un embedding enrichi d'un mot-clé pour faire remonter la
        bonne entité parmi plusieurs projets sémantiquement proches. None si
        absente (le retrieval retombe alors sur la recherche classique).
        Même logique de fusion multi-langue/partie que get_all_by_entity_type."""
        collection = self._require_collection()
        result = collection.get(
            where={"$and": [{"kind": {"$eq": _KIND_CHUNK}}, {"entity_id": {"$eq": entity_id}}]},
            include=["documents", "metadatas"],
        )
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        if not documents:
            return None

        by_lang: dict[str, dict[int, tuple[str, dict]]] = {}
        for doc, meta in zip(documents, metadatas):
            lang = meta.get("lang", "")
            part_index = meta.get("part_index", 0)
            by_lang.setdefault(lang, {})[part_index] = (doc, meta)

        lang_priority = [preferred_lang] if preferred_lang else []
        lang_priority += [l for l in ("en", "fr", "ar") if l != preferred_lang]
        chosen_lang = next((l for l in lang_priority if l in by_lang), next(iter(by_lang)))
        parts = by_lang[chosen_lang]
        ordered_docs = [parts[idx][0] for idx in sorted(parts)]
        representative_meta = parts[min(parts)][1]
        return {"document": "\n".join(ordered_docs), "metadata": representative_meta, "entity_id": entity_id}

    def count(self) -> int:
        """Nombre de chunks réels (exclut le document de métadonnées interne)."""
        collection = self._require_collection()
        result = collection.get(where={"kind": _KIND_CHUNK}, include=[])
        return len(result.get("ids", []))


chroma_store = ChromaStore()
