# Python 3.12 : version stable largement supportée par les dépendances
# (chromadb/onnxruntime). L'environnement de développement local tournait en
# 3.14, où chromadb==1.3.5 s'est révélé incompatible (voir rapport final) ;
# 3.12 évite ce type de risque de compatibilité de dépendances en production.
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# torch CPU installé explicitement en premier (index dédié) : les embeddings
# E5 tournent en local sur CPU pour ce projet, la build CUDA par défaut sur
# Linux serait plusieurs fois plus volumineuse pour aucun bénéfice ici.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY main.py .

# Pré-télécharge les poids du modèle E5 dans l'image : le conteneur démarre
# alors sans dépendre du réseau vers huggingface.co (cold start plus rapide
# et plus fiable qu'un téléchargement à la première requête).
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-base')"

# Vérifié en direct : même avec les poids déjà en cache local,
# sentence-transformers/huggingface_hub fait par défaut des requêtes HEAD/GET
# vers huggingface.co à CHAQUE démarrage pour vérifier les métadonnées. En
# production, le modèle étant déjà figé dans cette image, on force le mode
# 100% hors-ligne : aucune dépendance réseau à huggingface.co au runtime.
ENV HF_HUB_OFFLINE=1

# Aucun répertoire de données local : la persistance RAG (chunks, embeddings,
# métadonnées) vit entièrement dans Chroma Cloud (voir app/vectorstore).
# translations.js n'est pas copié dans l'image : le contenu initial est
# fourni via POST /upload-content, ou via TRANSLATIONS_JS_PATH si ce chemin
# est monté explicitement dans le conteneur.

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
