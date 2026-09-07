"""Orchestration de l'ingestion : parse -> chunk -> diff -> embed -> sync.

Garantie de rollback : l'appel au modèle d'embedding (seule étape pouvant
réellement échouer) s'exécute intégralement avant toute mutation de Chroma
Cloud. S'il échoue, IngestionError est levée sans avoir touché la base ;
la nouvelle version n'est marquée "active" qu'après la synchronisation réussie.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from app.cache.caches import invalidate_versioned_caches
from app.embeddings.base import EmbeddingProvider
from app.embeddings.e5_provider import e5_embedding_provider
from app.ingestion.chunker import chunk_content_blocks
from app.ingestion.parser import PortfolioParseError, parse_translations_js
from app.vectorstore.chroma_store import ChromaStore, chroma_store


class IngestionError(Exception):
    """Erreur fonctionnelle d'ingestion (fichier invalide ou échec d'embedding).

    Ne jamais lever cette exception après le début de `store.sync(...)`."""


@dataclass
class IngestionResult:
    status: str  # "unchanged" | "updated"
    source_hash: str
    chunk_count: int
    new_chunks: int
    updated_chunks: int
    unchanged_chunks: int
    deleted_chunks: int


def compute_source_hash(raw_bytes: bytes) -> str:
    return hashlib.sha256(raw_bytes).hexdigest()


def ingest_portfolio(
    raw_bytes: bytes,
    client: EmbeddingProvider | None = None,
    store: ChromaStore | None = None,
    force: bool = False,
) -> IngestionResult:
    """`force=True` contourne le raccourci "fichier source identique" et force
    le rafraîchissement des métadonnées de tous les chunks (branche
    `to_refresh`, sans nouvel appel d'embedding) — utile après une évolution
    du schéma de métadonnées alors que translations.js n'a pas changé."""
    client = client or e5_embedding_provider
    store = store or chroma_store

    if not raw_bytes:
        raise IngestionError("Le fichier envoyé est vide.")

    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise IngestionError(f"Encodage invalide : le fichier doit être encodé en UTF-8 ({exc}).") from exc

    source_hash = compute_source_hash(raw_bytes)
    if not force and store.get_content_version() == source_hash:
        current_count = store.count()
        return IngestionResult(
            status="unchanged",
            source_hash=source_hash,
            chunk_count=current_count,
            new_chunks=0,
            updated_chunks=0,
            unchanged_chunks=current_count,
            deleted_chunks=0,
        )

    try:
        blocks = parse_translations_js(text)
    except PortfolioParseError as exc:
        raise IngestionError(str(exc)) from exc

    chunks = chunk_content_blocks(blocks)
    if not chunks:
        raise IngestionError("Aucun chunk n'a pu être généré à partir du fichier (contenu insuffisant).")

    existing_hashes = store.existing_chunk_hashes()
    new_ids = {c.chunk_id for c in chunks}

    to_embed = [c for c in chunks if existing_hashes.get(c.chunk_id) != c.chunk_hash]
    to_refresh = [c for c in chunks if c not in to_embed]  # inchangés, métadonnées à rafraîchir
    ids_to_delete = [doc_id for doc_id in existing_hashes if doc_id not in new_ids]

    def _metadata(chunk) -> dict:
        return {
            "lang": chunk.lang,
            "section": chunk.section,
            "subsection": chunk.subsection,
            "part_index": chunk.part_index,
            "chunk_hash": chunk.chunk_hash,
            "topic_group": chunk.topic_group,
            "entity_type": chunk.entity_type,
            "entity_id": chunk.entity_id,
        }

    to_upsert: list[dict] = []
    if to_embed:
        try:
            embeddings = client.embed_documents([c.text for c in to_embed])
        except Exception as exc:
            raise IngestionError(f"Échec de génération des embeddings : {exc}") from exc

        if len(embeddings) != len(to_embed):
            raise IngestionError(
                "Réponse d'embedding incohérente : nombre de vecteurs différent du nombre de chunks."
            )

        for chunk, embedding in zip(to_embed, embeddings):
            to_upsert.append(
                {"id": chunk.chunk_id, "document": chunk.text, "embedding": embedding, "metadata": _metadata(chunk)}
            )

    if to_refresh:
        # Texte inchangé : on réutilise l'embedding déjà stocké, juste pour
        # rafraîchir les métadonnées (pas de nouvel appel d'embedding).
        cached_embeddings = store.get_embeddings([c.chunk_id for c in to_refresh])
        for chunk in to_refresh:
            embedding = cached_embeddings.get(chunk.chunk_id)
            if embedding is None:
                continue  # ne devrait pas arriver : chunk_hash matché => déjà en base
            to_upsert.append(
                {"id": chunk.chunk_id, "document": chunk.text, "embedding": embedding, "metadata": _metadata(chunk)}
            )

    # Point de non-retour : uniquement des opérations disque locales à partir d'ici.
    store.sync(to_upsert=to_upsert, ids_to_delete=ids_to_delete)
    store.commit_new_version(source_hash=source_hash, chunk_count=len(chunks))
    invalidate_versioned_caches()

    new_count = sum(1 for c in to_embed if c.chunk_id not in existing_hashes)
    updated_count = len(to_embed) - new_count

    return IngestionResult(
        status="updated",
        source_hash=source_hash,
        chunk_count=len(chunks),
        new_chunks=new_count,
        updated_chunks=updated_count,
        unchanged_chunks=len(to_refresh),
        deleted_chunks=len(ids_to_delete),
    )
