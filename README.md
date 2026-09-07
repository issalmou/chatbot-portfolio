# Issalmou Assistant AI — backend RAG multilingue

Backend FastAPI d'un chatbot RAG (Retrieval-Augmented Generation) pour le
portfolio d'Issalmou Adaaiche. Répond en **français, anglais ou arabe**
selon la langue de la question posée, à partir du contenu réel du
portfolio — jamais à partir de connaissances générales.

Ce document est écrit pour n'importe qui souhaitant **faire tourner, tester
ou intégrer** ce backend, pas seulement pour son auteur d'origine.

## Sommaire

- [Ce que fait ce backend](#ce-que-fait-ce-backend)
- [Architecture (vue d'ensemble)](#architecture-vue-densemble)
- [Démarrage rapide](#démarrage-rapide)
- [Configuration (.env)](#configuration-env)
- [API](#api)
- [Mémoire conversationnelle](#mémoire-conversationnelle)
- [Mettre à jour le contenu du portfolio](#mettre-à-jour-le-contenu-du-portfolio)
- [Tests](#tests)
- [Sécurité](#sécurité)
- [Limites connues](#limites-connues)

## Ce que fait ce backend

- **Source de vérité unique** : `translations.js`, le fichier i18n du
  frontend React (`src/data/translations.js`). Aucune copie n'est jamais
  conservée côté backend — voir `POST /upload-content`.
- **Multilingue strict FR/EN/AR** : la langue de la **question** détermine
  toujours la langue de la **réponse**, indépendamment de la langue du
  contenu récupéré ou de la conversation précédente.
- **Mémoire conversationnelle courte** (sans base de données) : comprend
  "it", "il", "ce projet", "هذا المشروع"... en résolvant la référence vers
  un projet/sujet précis à partir des derniers échanges fournis par le
  client à chaque requête — jamais stockée côté serveur.
- **Anti-hallucination** : ne répond qu'à partir du contenu réellement
  indexé ; une information absente est signalée comme telle (jamais comme
  une négation, ex. jamais "il n'a pas de X" pour une simple absence
  d'information) ; ne suit jamais une instruction qui serait dissimulée
  dans le contenu récupéré (protection anti prompt-injection).
- **Classification de périmètre locale** (sans appel LLM) : distingue une
  question liée au portfolio, une question ambiguë (référence introuvable
  → demande de clarification) et une question hors-sujet (réponse courte
  indiquant la spécialisation du chatbot).
- **Listes exhaustives fiables** : "donne-moi tous ses projets" retourne
  systématiquement le nombre réel d'entités, jamais un sous-ensemble
  tronqué par la recherche par similarité.
- **Navigation jamais inventée** : une section suggérée (ex. `/projects`)
  n'est proposée que si elle existe réellement dans le frontend — aucune
  page ni URL fictive.
- **Fallback multi-provider** : Gemini → Mistral → Groq → OpenAI, avec un
  disjoncteur (circuit breaker) en mémoire par provider — un provider en
  échec est automatiquement mis de côté puis retenté après un délai.
- **Multi-utilisateur sûr** : aucun état (langue, conversation, entité
  résolue, contexte récupéré) n'est jamais partagé entre deux requêtes ;
  les caches partagés sont thread-safe et correctement isolés par contenu.
- **312 tests automatisés**, sans mock pour les décisions critiques du
  pipeline, plus une suite de validation réelle contre Chroma Cloud/LLM.

## Architecture (vue d'ensemble)

```
translations.js (frontend)
   │  parse (JSON5, jamais exécuté) + chunking + hash SHA-256
   ▼
Embeddings E5 multilingue LOCAL (intfloat/multilingual-e5-base, aucune clé requise)
   ▼
Chroma Cloud (base vectorielle distante — aucune persistance locale)
   │
   ▼  (à chaque question)
détection de langue (FR/EN/AR)
   ▼
résolution de référence conversationnelle (mémoire courte, app/rag/memory.py)
   ▼
détection d'intention (question précise vs liste exhaustive)
   ▼
classification de périmètre (portfolio-lié / ambigu / hors-sujet — 100% local)
   ▼
retrieval (lookup exact par entité / liste exhaustive / recherche sémantique)
   ▼
prompt anti-hallucination ────────────► LLMProviderManager
                                          Gemini → Mistral → Groq → OpenAI
   ▼
validation de la langue de sortie (1 retry maximum si la langue détectée diffère)
   ▼
cache (embedding / retrieval / réponse — thread-safe, invalidé au contenu)
   ▼
réponse JSON
```

L'embedding (E5, local) est **totalement indépendant** du fallback LLM de
génération : changer de provider de génération n'affecte jamais les
embeddings, et inversement.

## Démarrage rapide

### Local (Python)

```bash
pip install -r requirements.txt
cp .env.example .env   # renseignez au moins GEMINI_API_KEY + CHROMA_API_KEY
uvicorn main:app --reload
```

Le premier démarrage télécharge le modèle E5 (~280 Mo, mis en cache par
`sentence-transformers`) ; les démarrages suivants le chargent depuis le
cache local (quelques secondes). Le serveur écoute sur `http://localhost:8000`.

Vérifiez rapidement que tout fonctionne :

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/chatbot \
  -H "Content-Type: application/json" \
  -d '{"query": "What projects has he built?"}'
```

### Docker

```bash
cp .env.example .env   # renseignez vos clés
docker compose up --build
```

Aucun volume à monter : toute la persistance (chunks, embeddings,
métadonnées) vit sur Chroma Cloud, jamais sur le disque du conteneur — le
conteneur est donc totalement jetable/reproductible.

## Configuration (.env)

Copiez `.env.example` en `.env` et renseignez les valeurs. **Une clé
absente ou vide = provider automatiquement ignoré** (pas d'erreur au
démarrage) ; n'utilisez jamais une valeur placeholder du type `"..."`.

| Variable | Obligatoire ? | Description |
|---|---|---|
| `CHROMA_API_KEY` | **Oui** | Clé API [Chroma Cloud](https://www.trychroma.com/) |
| `CHROMA_TENANT`, `CHROMA_DATABASE` | Non (souvent auto-résolues) | À renseigner explicitement si Chroma répond `ChromaAuthError: Could not determine a database name...` |
| `CHROMA_COLLECTION_NAME` | **Oui, avec cette valeur exacte** | ⚠️ **`portfolio_rag_e5`** — voir l'avertissement ci-dessous |
| `GEMINI_API_KEY`, `MISTRAL_API_KEY`, `GROQ_API_KEY`, `OPENAI_API_KEY` | Au moins un provider requis | Ordre de fallback fixe : Gemini → Mistral → Groq → OpenAI |
| `TRANSLATIONS_JS_PATH` | Non | Chemin local vers `translations.js` pour l'initialisation automatique au démarrage ; laissez vide en production (utilisez `POST /upload-content`) |
| `CONTENT_UPLOAD_TOKEN` | Recommandé en production | Protège `POST /upload-content` (`Authorization: Bearer <token>`) ; si vide, l'endpoint reste ouvert (dev local uniquement, un avertissement est loggé) |
| `CONVERSATION_MEMORY_TURNS` | Non (défaut `3`) | Nombre de derniers échanges considérés pour la mémoire conversationnelle — borné à 5 au maximum quelle que soit la valeur fournie |
| `RETRIEVAL_TOP_K` | Non (défaut `8`) | Nombre de chunks pour une recherche sémantique classique (sans effet sur les listes exhaustives, gérées séparément par entité) |
| `LLM_REQUEST_TIMEOUT_SECONDS`, `LLM_CIRCUIT_BREAKER_COOLDOWN_SECONDS` | Non | Réglages fins du fallback multi-provider |

> **⚠️ Piège fréquent — `CHROMA_COLLECTION_NAME` n'est PAS vraiment
> optionnelle.** Le code a un défaut interne (`portfolio_rag`) qui pointe
> vers une **ancienne collection abandonnée** (embeddings Gemini, dimension
> incompatible avec le modèle E5 actuel). Si vous omettez cette variable en
> vous fiant au fait qu'elle "a une valeur par défaut", le retrieval semblera
> fonctionner (aucune erreur) mais retournera des résultats incohérents ou
> vides. **Renseignez toujours explicitement `CHROMA_COLLECTION_NAME=portfolio_rag_e5`**
> (déjà correct dans `.env.example` — ne le supprimez pas).

Toutes les autres variables (chunking, tailles/TTL de cache, CORS, nom
protégé) ont des valeurs par défaut raisonnables — voir `app/config.py`
pour la liste complète et le détail de chacune.

## API

### `POST /chatbot`

```json
{
  "query": "What technologies does it use?",
  "conversation": [
    { "role": "user", "content": "Tell me about the AGEP project." },
    { "role": "assistant", "content": "AGEP is a platform for..." }
  ]
}
```

- `query` (obligatoire) : la question, dans n'importe laquelle des 3
  langues supportées. **Jamais traduite** : la réponse arrive dans la même
  langue que `query`, quel que soit le contenu de `conversation`.
- `conversation` (optionnel, rétrocompatible — `{"query": "..."}` seul
  fonctionne toujours) : jusqu'à 12 messages `{role, content}` acceptés ;
  seuls les derniers pertinents sont réellement utilisés par la mémoire
  (voir plus bas). Aucune conversation n'est jamais conservée côté serveur
  entre deux requêtes — le client doit la renvoyer à chaque fois.

Réponse :

```json
{
  "response": "AGEP uses HTML/CSS, Bootstrap, JavaScript, Laravel and MySQL.",
  "lang": "en",
  "metrics": {
    "provider": "gemini",
    "model": "gemini-flash-latest",
    "fallback_used": false,
    "cache_hit": "none",
    "retrieved_chunks": 1,
    "total_ms": 842.3
  }
}
```

`metrics` contient des informations de diagnostic public (latence,
provider utilisé...) ; les métriques internes de raisonnement (intention
détectée, sujet, entités résolues...) sont volontairement **exclues** de la
réponse HTTP et ne sont visibles que dans les logs serveur.

Codes d'erreur : `400` (question vide), `503` (aucun provider LLM
disponible, ou Chroma Cloud injoignable) — jamais de détail technique,
de clé API ou de stack trace dans le corps de la réponse.

### `POST /upload-content`

Upload d'une nouvelle version de `translations.js` (multipart, champ
`file`). Voir [Mettre à jour le contenu](#mettre-à-jour-le-contenu-du-portfolio).

### `GET /health`

État des providers LLM (configuré/disponible/en cooldown) et de la
connexion Chroma Cloud — jamais de clé API dans la réponse.

## Mémoire conversationnelle

Le backend est **stateless** : aucune conversation n'est jamais stockée en
base ou en mémoire serveur entre deux requêtes. C'est le **client** qui
renvoie l'historique pertinent à chaque appel, via le champ `conversation`.

- Seuls les derniers échanges sont réellement pris en compte (voir
  `CONVERSATION_MEMORY_TURNS`, borné à 5 quoi qu'il arrive) — envoyer un
  historique plus long ne sert à rien, le surplus est ignoré.
- La résolution de référence ("it", "il", "ce projet", "هذا المشروع"...) ne
  **devine jamais** : si la conversation mentionne plusieurs projets
  possibles, ou n'en mentionne aucun, le backend répond en demandant une
  clarification plutôt que de choisir arbitrairement.
- La langue de la conversation précédente n'influence **jamais** la langue
  de la réponse courante — seule la langue de `query` compte.

## Mettre à jour le contenu du portfolio

Depuis le dépôt **frontend** (`issalmou-portfolio`), après avoir édité
`src/data/translations.js` :

```bash
npm run sync-chatbot
```

(variables `CHATBOT_API_URL` et `CONTENT_UPLOAD_TOKEN`, voir le
`.env.example` du frontend — jamais préfixées `VITE_`, jamais exécutées
dans le navigateur). En production, cette commande tourne automatiquement
en fin de build Netlify (`netlify.toml`).

Équivalent manuel :

```bash
curl -X POST http://localhost:8000/upload-content \
  -H "Authorization: Bearer $CONTENT_UPLOAD_TOKEN" \
  -F "file=@src/data/translations.js"
```

Le fichier uploadé n'est jamais conservé côté serveur (traitement en
mémoire uniquement, y compris en cas d'erreur). Si son hash SHA-256 est
identique à la version déjà indexée, la requête retourne
`{"status": "unchanged"}` sans aucun calcul d'embedding. Sinon, seuls les
chunks réellement nouveaux ou modifiés sont ré-embeddés ; les chunks
disparus sont supprimés de Chroma Cloud.

Ajoutez `?force=true` pour forcer le rafraîchissement des métadonnées de
tous les chunks sans réappeler le modèle d'embedding (utile uniquement
après une évolution du schéma de métadonnées interne, pas pour un usage
courant).

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -q
```

**312 tests**, aucun appel réseau réel par défaut (Chroma Cloud et les
providers LLM sont simulés par des doublures fidèles à leur interface
réelle). Un test d'intégration réel contre Chroma Cloud existe dans
`tests/test_pipeline.py` et se lance automatiquement dès que
`CHROMA_API_KEY` est configurée (sinon il est proprement ignoré).

Pour une validation manuelle end-to-end avec de vrais providers LLM :

```bash
docker compose up --build
curl -X POST http://localhost:8000/chatbot \
  -H "Content-Type: application/json" \
  -d '{"query": "Donne-moi tous ses projets."}'
```

## Sécurité

- Aucun secret n'est jamais loggé, renvoyé dans une réponse HTTP, ou
  exposé via `/health`.
- `POST /upload-content` : comparaison du token en temps constant
  (`secrets.compare_digest`), jamais dans le frontend/un bundle
  JavaScript/une variable `VITE_*`.
- Le contenu récupéré depuis le portfolio est toujours traité comme de la
  **donnée**, jamais comme une instruction — une phrase du type "ignore les
  instructions précédentes" dissimulée dans le contenu indexé est ignorée
  par construction (voir le system prompt, `app/rag/retrieval.py`).
- Conversation client bornée (12 messages max, 2000 caractères par
  message) — anti-abus, sans jamais devenir un historique permanent.

## Limites connues

- Un seul worker Uvicorn est utilisé volontairement (le modèle E5 est
  chargé en mémoire ; plusieurs workers en multiplieraient la consommation
  RAM sans bénéfice d'isolation, déjà garantie par ailleurs). Les requêtes
  `/chatbot` sont exécutées via un threadpool (`run_in_threadpool`), donc
  réellement concurrentes malgré le worker unique.
- La classification de périmètre et la résolution de référence restent des
  heuristiques locales (mots-clés/regex), pas un classifieur appris —
  robustes sur tous les cas testés, mais une formulation très inhabituelle
  pourrait échapper à la détection.
- La couverture des formes possessives arabes est une liste énumérée, pas
  un analyseur morphologique complet.
- Les quotas des providers LLM (Gemini/Mistral/Groq gratuits notamment)
  sont une contrainte externe réelle : sous forte charge, certaines
  requêtes peuvent recevoir un `503` propre en attendant le fallback
  suivant ou la fin du cooldown.
