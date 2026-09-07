# 3.12 : chromadb==1.3.5 s'est révélé incompatible avec 3.14 (dev local).
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# pip/setuptools/wheel outils de build uniquement (jamais importés au runtime),
# mais corrige des CVE connues (setuptools path traversal, wheel path traversal).
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements.txt .
# torch CPU installé en premier, version fixée : le wheel par défaut sur
# PyPI déclare cuda-toolkit/nvidia-* comme dépendances obligatoires sur
# Linux (même sans GPU) — E5 tourne entièrement sur CPU (app/config.py).
# Une fois installé, requirements.txt le trouve déjà satisfaisant et ne le
# retélécharge jamais depuis l'index par défaut.
RUN pip install --no-cache-dir torch==2.14.0+cpu --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY main.py .

# Pré-télécharge les poids E5 dans l'image : démarrage sans dépendre du
# réseau vers huggingface.co (cold start plus rapide et fiable).
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-base')"

# Sans ça, huggingface_hub vérifie les métadonnées du modèle en ligne à
# chaque démarrage même avec les poids déjà en cache local.
ENV HF_HUB_OFFLINE=1

# Aucune donnée locale : la persistance RAG vit entièrement dans Chroma
# Cloud. translations.js n'est pas copié ici, le contenu passe par
# POST /upload-content (ou TRANSLATIONS_JS_PATH si monté explicitement).

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
