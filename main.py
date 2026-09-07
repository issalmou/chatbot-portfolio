# main.py
"""Point d'entrée FastAPI — routes uniquement.

Toute la logique métier vit dans app/ : ingestion (app/ingestion), stockage
vectoriel Chroma Cloud (app/vectorstore), retrieval + génération multi-provider
(app/rag, app/llm), cache (app/cache). translations.js est la source de
vérité unique du contenu RAG ; le backend n'en garde jamais de copie
permanente (voir app/ingestion/pipeline.py et /upload-content ci-dessous).
"""

import logging
import os
import secrets
import time
from contextlib import asynccontextmanager

from typing import Literal

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from app.config import settings
from app.embeddings.e5_provider import EmbeddingProviderError, e5_embedding_provider
from app.ingestion.pipeline import IngestionError, ingest_portfolio
from app.llm.manager import llm_manager
from app.rag.memory import Turn
from app.rag.retrieval import AllProvidersUnavailableError, answer_question
from app.vectorstore.chroma_store import ChromaUnavailableError, chroma_store

# Bornes défensives côté API (coût/latence, anti-abus payload) — distinctes de
# settings.conversation_memory_turns (combien de tours la mémoire utilise
# réellement, voir app/rag/memory.py::MAX_TURNS) : l'API peut accepter un
# historique plus long que ce que la mémoire regarde ensuite.
MAX_MESSAGE_LENGTH = 2000  # caractères, par message de conversation
MAX_CONVERSATION_MESSAGES_ACCEPTED = 12  # messages (pas "tours"), acceptés par l'API

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issalmou_assistant")

# Métriques réservées aux logs serveur, jamais renvoyées dans la réponse HTTP publique.
DEBUG_ONLY_METRIC_KEYS = {
    "intent", "entity_type", "unique_entities_count", "entity_ids", "duplicates_removed",
    "topic", "suggested_route", "scope", "reference_resolved", "conversation_turns_used",
}

MAX_UPLOAD_SIZE_BYTES = 5 * 1024 * 1024  # 5 Mo, largement suffisant pour un portfolio texte


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Initialisation depuis translations.js si le fichier est accessible
    depuis ce backend. Ne réindexe que si le hash a changé (ingest_portfolio).
    Si le fichier ou Chroma Cloud est indisponible, on log et on continue
    sans casser le démarrage ni l'index existant."""
    if not settings.content_upload_token:
        logger.warning(
            "CONTENT_UPLOAD_TOKEN non configuré : /upload-content est accessible "
            "sans authentification. À éviter en production."
        )

    # Vérifie l'API d'embedding (HF_TOKEN valide, modèle joignable) au démarrage
    # plutôt qu'à la première requête utilisateur.
    try:
        t0 = time.perf_counter()
        e5_embedding_provider.embed_query("warmup")
        logger.info("API d'embedding vérifiée en %.2fs.", time.perf_counter() - t0)
    except Exception as exc:
        logger.warning("Vérification de l'API d'embedding échouée (%s: %s) — sera retentée à la demande.", type(exc).__name__, exc)

    path = settings.translations_js_path
    if not path or not os.path.exists(path):
        logger.info("translations.js non accessible depuis ce backend (%s) : index Chroma Cloud existant conservé tel quel.", path)
    else:
        try:
            with open(path, "rb") as f:
                raw_bytes = f.read()
            result = ingest_portfolio(raw_bytes)
            logger.info("Initialisation depuis translations.js : status=%s chunks=%s", result.status, result.chunk_count)
        except Exception as exc:
            logger.warning("Initialisation depuis translations.js ignorée (%s: %s).", type(exc).__name__, exc)
    yield


app = FastAPI(
    title="Issalmou Assistant AI",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConversationTurnModel(BaseModel):
    """Un tour de conversation fourni par le client (mémoire courte, sans
    stockage serveur — voir app/rag/memory.py). `role` restreint à
    user/assistant : impossible d'injecter un faux message "system"."""

    role: Literal["user", "assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def _truncate_content(cls, value: str) -> str:
        return value[:MAX_MESSAGE_LENGTH]  # tronque plutôt que de rejeter la requête


class ChatRequest(BaseModel):
    """`conversation` est optionnel et rétrocompatible : {"query": "..."}
    seul continue de fonctionner à l'identique (conversation vide = comportement inchangé)."""

    query: str
    conversation: list[ConversationTurnModel] = Field(default_factory=list, max_length=MAX_CONVERSATION_MESSAGES_ACCEPTED)


@app.get("/chatbot")
async def chat_hello():
    return {"response": "hello world i'am assistant AI of Issalmou"}


@app.post("/chatbot")
async def chat_endpoint(request: ChatRequest):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide.")

    conversation = [Turn(role=turn.role, content=turn.content) for turn in request.conversation]

    try:
        # answer_question() est synchrone et bloquante (embedding CPU, appels réseau
        # Chroma/LLM) : l'exécuter directement bloquerait la boucle asyncio entière,
        # sérialisant toutes les requêtes sur ce worker. run_in_threadpool() la délègue
        # à un thread du pool, permettant plusieurs requêtes en parallèle — sûr car
        # `request`/`conversation`/`result` sont locales à cet appel (les seuls états
        # partagés, les caches, sont déjà protégés par verrou, voir app/cache/local_cache.py).
        result = await run_in_threadpool(answer_question, request.query, conversation=conversation)
    except AllProvidersUnavailableError as exc:
        logger.error("Tous les providers LLM sont indisponibles: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Le service de génération n'est temporairement disponible sur aucun fournisseur.",
        )
    except ChromaUnavailableError:
        logger.error("Chroma Cloud injoignable pendant /chatbot.")
        raise HTTPException(status_code=503, detail="La base documentaire est temporairement indisponible.")
    except EmbeddingProviderError as exc:
        logger.error("API d'embedding indisponible pendant /chatbot: %s", exc)
        raise HTTPException(status_code=503, detail="Le service d'embedding est temporairement indisponible.")

    logger.info(
        "chat lang=%s scope=%s intent=%s topic=%s suggested_route=%s entity_type=%s "
        "reference_resolved=%s conversation_turns=%s provider=%s model=%s "
        "fallback=%s language_retry=%s cache_hit=%s retrieved=%s unique_entities=%s "
        "duplicates_removed=%s total_ms=%s",
        result.lang,
        result.metrics.get("scope"),
        result.metrics.get("intent"),
        result.metrics.get("topic"),
        result.metrics.get("suggested_route"),
        result.metrics.get("entity_type"),
        result.metrics.get("reference_resolved"),
        result.metrics.get("conversation_turns_used"),
        result.metrics.get("provider"),
        result.metrics.get("model"),
        result.metrics.get("fallback_used"),
        result.metrics.get("language_retry"),
        result.metrics.get("cache_hit"),
        result.metrics.get("retrieved_chunks"),
        result.metrics.get("unique_entities_count"),
        result.metrics.get("duplicates_removed"),
        result.metrics.get("total_ms"),
    )

    public_metrics = {k: v for k, v in result.metrics.items() if k not in DEBUG_ONLY_METRIC_KEYS}
    return {"response": result.response, "lang": result.lang, "metrics": public_metrics}


def _verify_upload_token(authorization: str | None = Header(default=None)) -> None:
    """Exige "Authorization: Bearer <CONTENT_UPLOAD_TOKEN>" si configuré.
    Comparaison en temps constant (secrets.compare_digest) contre le timing attack."""
    if not settings.content_upload_token:
        return  # non configuré : accès ouvert (dev local), déjà signalé au démarrage
    expected = f"Bearer {settings.content_upload_token}"
    if not authorization or not secrets.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="Token d'upload manquant ou invalide.")


@app.post("/upload-content", dependencies=[Depends(_verify_upload_token)])
async def upload_content(file: UploadFile = File(...), force: bool = False):
    """Upload d'une nouvelle version de translations.js. Traité entièrement en
    mémoire : jamais écrit sur disque ni conservé côté backend. Protégé par
    CONTENT_UPLOAD_TOKEN si configuré.

    `force=true` contourne la détection "contenu inchangé" pour rafraîchir les
    métadonnées de tous les chunks (après une évolution de schéma), sans
    nouvel appel d'embedding puisque le texte n'a pas changé."""
    if file.filename and not file.filename.lower().endswith(".js"):
        raise HTTPException(status_code=400, detail="Le fichier doit être translations.js (extension .js).")

    raw_bytes = await file.read()
    if len(raw_bytes) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="Fichier trop volumineux (5 Mo maximum).")

    try:
        result = ingest_portfolio(raw_bytes, force=force)
    except IngestionError as exc:
        logger.warning("Échec d'ingestion du portfolio: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except ChromaUnavailableError:
        logger.error("Chroma Cloud injoignable pendant /upload-content.")
        raise HTTPException(status_code=503, detail="Chroma Cloud est temporairement indisponible.")

    logger.info(
        "upload-content status=%s chunks=%s new=%s updated=%s unchanged=%s deleted=%s",
        result.status,
        result.chunk_count,
        result.new_chunks,
        result.updated_chunks,
        result.unchanged_chunks,
        result.deleted_chunks,
    )

    return {
        "status": result.status,
        "source_hash": result.source_hash,
        "chunk_count": result.chunk_count,
        "new_chunks": result.new_chunks,
        "updated_chunks": result.updated_chunks,
        "unchanged_chunks": result.unchanged_chunks,
        "deleted_chunks": result.deleted_chunks,
    }


@app.get("/health")
async def health():
    """État des providers LLM et de Chroma Cloud. Ne retourne jamais de clé API,
    quel que soit l'état de la connexion (y compris en cas d'erreur)."""
    llm_status = llm_manager.status()
    active_provider = next(
        (name for name, info in llm_status.items() if info["available"]), None
    )

    try:
        rag_status = {
            "chroma_connected": True,
            "content_version": chroma_store.get_content_version(),
            "chunk_count": chroma_store.count(),
        }
    except ChromaUnavailableError:
        rag_status = {"chroma_connected": False, "content_version": None, "chunk_count": None}

    return {
        "status": "ok",
        "llm": {
            "active_provider": active_provider,
            "configured": [name for name, info in llm_status.items() if info["configured"]],
            "providers": llm_status,
        },
        "rag": rag_status,
    }
