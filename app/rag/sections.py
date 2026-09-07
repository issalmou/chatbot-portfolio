"""Sections RÉELLES du portfolio (issalmou-portfolio) et détection du sujet
d'une question, pour proposer — uniquement quand c'est pertinent — une
destination de navigation VALIDÉE par le backend, jamais inventée par le LLM.

Vérifié en direct dans le frontend (routes React Router, pas d'ancres
`/#section`) : skills vit sur `/about#skills`, `/resume` regroupe formation
ET expériences, et aucune page "Certifications" n'existe. SECTION_ROUTES ne
contient QUE des destinations vérifiées — à tenir à jour si le frontend change.
"""

from __future__ import annotations

import re

SECTION_ROUTES: dict[str, str] = {
    "about": "/about",
    "skills": "/about#skills",
    "resume": "/resume",       # formation ET expériences/stages
    "projects": "/projects",
    "services": "/services",
    "contact": "/contact",
}

# Sujets reconnus mais sans destination réelle : jamais de route proposée.
_NO_ROUTE_TOPICS = {"certification"}

# entity_type (app/ingestion/parser.py::derive_entity) -> topic de routage.
_ENTITY_TYPE_TO_TOPIC = {
    "project": "projects",
    "skills_data": "skills",
    "internship": "resume",
    "education": "resume",
    "service": "services",
    "intro": "about",
    "info": "contact",
}

_TOPIC_KEYWORDS: dict[str, dict[str, str]] = {
    "en": {
        "projects": "projects", "project": "projects",
        "skills": "skills", "skill": "skills",
        "experience": "resume", "experiences": "resume", "internship": "resume", "internships": "resume", "work history": "resume",
        "education": "resume", "degree": "resume", "studied": "resume", "study": "resume", "university": "resume",
        "services": "services", "service": "services",
        # Pas de simple mot "about" : collisionnerait avec "tell me more about it"
        # et empêcherait la détection d'ambiguïté sur ce cas.
        "who is": "about", "profile": "about", "about him": "about", "about the developer": "about",
        "contact": "contact", "email": "contact", "phone": "contact", "reach": "contact", "hire": "contact",
        "certification": "certification", "certifications": "certification", "certificate": "certification", "certified": "certification",
    },
    "fr": {
        "projets": "projects", "projet": "projects",
        "compétences": "skills", "competences": "skills", "compétence": "skills",
        "expérience": "resume", "expériences": "resume", "stage": "resume", "stages": "resume", "parcours professionnel": "resume",
        "formation": "resume", "études": "resume", "diplôme": "resume", "université": "resume",
        "services": "services", "service": "services",
        "à propos": "about", "profil": "about", "qui est": "about",
        "contact": "contact", "email": "contact", "téléphone": "contact", "joindre": "contact", "embaucher": "contact",
        "certification": "certification", "certifications": "certification", "certificat": "certification", "certifié": "certification",
    },
    "ar": {
        "مشاريع": "projects", "مشروع": "projects",
        "مهارات": "skills", "مهارة": "skills",
        "خبرة": "resume", "خبرات": "resume", "تدريب": "resume",
        "تعليم": "resume", "دراسة": "resume", "شهادة جامعية": "resume", "جامعة": "resume", "درس": "resume",
        "خدمات": "services", "خدمة": "services",
        "نبذة": "about", "من هو": "about", "الملف الشخصي": "about",
        "اتصل": "contact", "تواصل": "contact", "البريد": "contact", "الهاتف": "contact",
        "شهادة": "certification", "شهادات": "certification",
        # Formes possessives suffixées ("خبرته") : non attrapables par simple
        # sous-chaîne, ajoutées explicitement plutôt que via un analyseur morphologique.
        "خبرته": "resume", "تعليمه": "resume", "دراسته": "resume",
        "خدماته": "services", "مهاراته": "skills", "مشاريعه": "projects",
        "شهاداته": "certification",
    },
}

# Marqueurs d'une question purement anaphorique ("dis m'en plus"), utiles
# seulement en l'absence de tout sujet reconnu ci-dessus.
_AMBIGUOUS_PATTERNS: dict[str, re.Pattern] = {
    "en": re.compile(r"\b(more about it|tell me more|about that|about this|elaborate|go on|and then)\b", re.IGNORECASE),
    "fr": re.compile(r"\b(plus à ce sujet|dis[- ]?m'en plus|en dire plus|et ensuite|continue|développe)\b", re.IGNORECASE),
    "ar": re.compile(r"(أخبرني أكثر|المزيد عن ذلك|زيدني|كمل|أكمل)"),
}


def detect_topic(query: str, lang: str) -> str | None:
    """Sujet de la question parmi ceux de SECTION_ROUTES, ou "certification"
    (reconnu mais sans route), ou None si aucun sujet n'est identifiable."""
    keywords = _TOPIC_KEYWORDS.get(lang, {})
    lowered = query if lang == "ar" else query.lower()
    for keyword, topic in keywords.items():
        if keyword in lowered:
            return topic
    return None


def detect_ambiguous(query: str, lang: str) -> bool:
    """True si la question est une simple relance anaphorique sans sujet identifiable."""
    if detect_topic(query, lang) is not None:
        return False
    pattern = _AMBIGUOUS_PATTERNS.get(lang)
    return bool(pattern and pattern.search(query))


def route_for_topic(topic: str | None) -> str | None:
    """Route réelle pour un sujet, ou None si le sujet n'a délibérément pas
    de destination (ex. "certification") ou n'est pas reconnu."""
    if not topic or topic in _NO_ROUTE_TOPICS:
        return None
    return SECTION_ROUTES.get(topic)


def route_for_entity_type(entity_type: str | None) -> str | None:
    """Variante basée sur l'entity_type d'un chunk retrouvé (voir
    app/ingestion/parser.py::derive_entity), utile quand le sujet de la
    question n'est pas identifiable par mot-clé mais qu'un chunk pertinent
    a tout de même été retrouvé par similarité sémantique."""
    topic = _ENTITY_TYPE_TO_TOPIC.get(entity_type or "")
    return route_for_topic(topic)
