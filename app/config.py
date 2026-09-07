"""Configuration centralisée de l'application (variables d'environnement + constantes)."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Embeddings : E5 multilingue local (MIT, dim. réelle 768, 100 langues dont
    # fr/en/ar), indépendant du fallback LLM de génération — aucun appel réseau à l'inférence.
    embedding_model_id: str = "intfloat/multilingual-e5-base"
    # À incrémenter si la logique d'encodage change sans changer de modèle (clé de cache).
    embedding_model_version: str = "1"
    embedding_device: str = "cpu"
    # Dimension mesurée du modèle, utilisée pour le vecteur de remplissage des
    # métadonnées internes à Chroma (voir chroma_store.py) — pas pour valider les vecteurs réels.
    embedding_dimension: int = 768

    # Fournisseurs LLM de génération, ordre de fallback strict : Gemini -> Mistral
    # -> Groq -> OpenAI. Un provider sans clé API est simplement ignoré (app/llm/manager.py).
    gemini_api_key: str | None = None
    # Alias "-latest" maintenu par Google : une version figée ("gemini-2.5-flash")
    # a déjà fini en 404 "no longer available to new users".
    gemini_model: str = "gemini-flash-latest"

    mistral_api_key: str | None = None
    mistral_base_url: str = "https://api.mistral.ai/v1"
    mistral_model: str = "mistral-small-latest"

    groq_api_key: str | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    # "llama-3.3-70b-versatile" a été décommissionné (404 model_not_found) ;
    # gpt-oss-120b vérifié disponible, rapide et respecte les instructions de langue.
    groq_model: str = "openai/gpt-oss-120b"

    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"

    llm_request_timeout_seconds: float = 12.0
    llm_circuit_breaker_cooldown_seconds: float = 60.0

    # Chroma Cloud (chromadb.CloudClient)
    chroma_api_key: str | None = None
    chroma_tenant: str | None = None
    chroma_database: str | None = None
    chroma_collection_name: str = "portfolio_rag"

    # Source d'initialisation optionnelle au démarrage si accessible depuis le
    # backend (voir main.py) ; la mise à jour normale passe par POST /upload-content.
    translations_js_path: str | None = (
        r"C:\Users\isalm\Pictures\issalmou-portfolio\src\data\translations.js"
    )

    # Nombre de derniers échanges considérés pour la résolution de référence
    # (app/rag/memory.py). Borné en dur côté code : ne doit jamais transformer
    # la mémoire courte en historique permanent, quelle que soit cette valeur.
    conversation_memory_turns: int = 3

    # 8 vérifié empiriquement nécessaire pour qu'une question sur "les projets"
    # retrouve bien tous les projets distincts, pas seulement le premier.
    retrieval_top_k: int = 8

    chunk_max_chars: int = 900
    chunk_min_chars: int = 40

    embedding_cache_max_size: int = 512
    embedding_cache_ttl_seconds: int = 86400  # 24h : l'embedding d'un texte donné ne change pas
    retrieval_cache_max_size: int = 256
    retrieval_cache_ttl_seconds: int = 3600
    response_cache_max_size: int = 256
    response_cache_ttl_seconds: int = 3600

    allowed_origins: list[str] = ["https://issalmouad.com"]

    protected_name: str = "Issalmou Adaaiche"

    # Si vide, /upload-content reste accessible sans authentification (dev local,
    # avertissement loggé au démarrage). En production, définir cette variable pour
    # exiger "Authorization: Bearer <token>". Ne jamais exposer côté frontend.
    content_upload_token: str | None = None


settings = Settings()
