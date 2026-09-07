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
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY main.py .

# Aucune donnée locale : la persistance RAG vit entièrement dans Chroma
# Cloud. translations.js n'est pas copié ici, le contenu passe par
# POST /upload-content (ou TRANSLATIONS_JS_PATH si monté explicitement).

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
