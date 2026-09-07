"""Détection d'intention : question EXHAUSTIVE (« liste tous les X ») vs
question SPÉCIFIQUE (retrieval sémantique classique). Le retrieval classique
(top_k borné) est conçu pour la pertinence, pas l'exhaustivité d'une catégorie
à plusieurs entités — d'où cette détection dédiée, purement locale.

Ordre des signaux : (1) quantificateur de totalité explicite (all/tous/جميع)
→ exhaustif ; (2) référence à un projet précis ("project X", "المشروع")
→ force spécifique même si un pluriel apparaît ailleurs ; (3) sinon, un nom
d'entité pluriel/collectif connu sans référence précise → exhaustif implicite.

La résolution du type d'entité (project/skill/experience/...) utilise un
dictionnaire multilingue restreint aux types listables (voir
app/ingestion/parser.py::derive_entity) ; en cas d'ambiguïté,
app/rag/retrieval.py retombe sur un signal sémantique.
"""

from __future__ import annotations

import re

# 1. Quantificateurs de totalité
_TOTALITY_PATTERNS: dict[str, re.Pattern] = {
    "en": re.compile(r"\b(all|every|entire|complete|whole)\b", re.IGNORECASE),
    "fr": re.compile(r"\b(tous|toutes|tout|entière|complète|complet|ensemble des)\b", re.IGNORECASE),
    "ar": re.compile(r"(جميع|كل|كافة)"),
}

# 2. Références à une entité précise (force SPECIFIC)
_SPECIFIC_REFERENCE_PATTERNS: dict[str, re.Pattern] = {
    "en": re.compile(r"\b(this|that)\s+project\b|\bproject\s+[A-Z]", re.IGNORECASE),
    "fr": re.compile(r"\b(ce|cette|du|au)\s+projet\b|\bprojet\s+[A-Z]", re.IGNORECASE),
    # "المشروع" (défini singulier) est morphologiquement distinct de "المشاريع" (pluriel).
    "ar": re.compile(r"المشروع"),
}

# 3. Noms d'entité pluriels/collectifs connus, sans référence précise. Restreint
# aux entity_type listables (voir derive_entity) — pas de "technologies"/"تقنيات"
# ici, elles vivent dans les chunks projet, pas comme entités indexées séparément.
_ENTITY_KEYWORDS: dict[str, dict[str, str]] = {
    "en": {
        "projects": "project", "project": "project",
        "skills": "skills_data",
        "experiences": "internship", "experience": "internship", "internships": "internship",
        "education": "education",
        "services": "service",
        "certifications": "certification", "certification": "certification",
    },
    "fr": {
        "projets": "project", "projet": "project",
        "compétences": "skills_data", "competences": "skills_data",
        "expériences": "internship", "experiences": "internship", "stages": "internship",
        "formations": "education", "formation": "education",
        "services": "service",
        "certifications": "certification", "certification": "certification",
    },
    "ar": {
        "مشاريع": "project",
        "مهارات": "skills_data",
        "خبرات": "internship",
        "تعليم": "education", "دراسات": "education",
        "خدمات": "service",
        "شهادات": "certification",
    },
}

_BARE_PLURAL_PATTERNS: dict[str, re.Pattern] = {
    lang: re.compile("|".join(re.escape(k) for k in keywords), re.IGNORECASE if lang != "ar" else 0)
    for lang, keywords in _ENTITY_KEYWORDS.items()
}


def detect_list_intent(query: str, lang: str) -> bool:
    """True si la question demande une liste exhaustive plutôt qu'une
    réponse ciblée. Ordre des règles important (voir docstring du module)."""
    totality = _TOTALITY_PATTERNS.get(lang)
    if totality and totality.search(query):
        return True

    specific = _SPECIFIC_REFERENCE_PATTERNS.get(lang)
    if specific and specific.search(query):
        return False

    bare_plural = _BARE_PLURAL_PATTERNS.get(lang)
    if bare_plural and bare_plural.search(query):
        return True

    return False


def resolve_entity_type_keyword(query: str, lang: str) -> str | None:
    """Type d'entité explicitement nommé dans la question (project, skills_data,
    internship, education, service, certification), ou None si ambigu — dans
    ce cas app/rag/retrieval.py retombe sur un signal sémantique."""
    keywords = _ENTITY_KEYWORDS.get(lang, {})
    lowered = query if lang == "ar" else query.lower()
    for keyword, entity_type in keywords.items():
        if keyword in lowered:
            return entity_type
    return None
