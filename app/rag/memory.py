"""Mémoire conversationnelle courte (stateless, fournie par le client à
chaque requête) et résolution de référence ("it", "ce projet", "هذا
المشروع") avant le retrieval. Ne devine jamais : sans référent identifiable,
`needs_clarification=True`. N'influence jamais la langue de réponse (calculée
avant tout accès à la conversation, voir app/rag/retrieval.py)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.config import settings
from app.language import detect_language
from app.rag.sections import detect_ambiguous, detect_topic

# Bornée en dur (1-5) : CONVERSATION_MEMORY_TURNS ne doit jamais transformer
# la mémoire courte en historique permanent.
_MAX_TURNS_CEILING = 5


def _effective_max_turns() -> int:
    return max(1, min(settings.conversation_memory_turns, _MAX_TURNS_CEILING))


MAX_TURNS = _effective_max_turns()


@dataclass(frozen=True)
class Turn:
    role: str  # "user" | "assistant"
    content: str


@dataclass(frozen=True)
class ReferenceResolution:
    retrieval_query: str           # texte pour l'embedding/retrieval si aucune entité précise trouvée
    needs_clarification: bool
    focus_topic: str | None        # sujet canonique (voir app/rag/sections.py::SECTION_ROUTES)
    focus_entity_name: str | None  # nom brut extrait (ex. "AGEP") -> permet un lookup direct par entity_id
    focus_hint: str | None         # étiquette injectée comme CONVERSATION_FOCUS dans le prompt
    applied: bool                  # True seulement si résolu via la conversation ce tour-ci


# Pronoms/tournures visant un référent externe à la question elle-même.
# Séparé de sections._AMBIGUOUS_PATTERNS (relances génériques type "tell me
# more", réutilisées ci-dessous via detect_ambiguous).
_REFERENTIAL_PATTERNS: dict[str, re.Pattern] = {
    "en": re.compile(r"\b(it|this project|that project|this one|these ones|which ones|the ones|them)\b", re.IGNORECASE),
    "fr": re.compile(r"\b(il|elle|celui-ci|celui-là|celle-ci|celle-là|ce projet|lesquell[eé]s|ceux|celles)\b", re.IGNORECASE),
    # "تقنياته" (ses technologies) : "تقنيات" est volontairement exclu des
    # mots-clés de sujet (app/rag/sections.py), donc doit être capté ici.
    "ar": re.compile(r"(هذا المشروع|هذا|هذه|ذلك|عنه|منه|فيه|منها|فيها|يستخدمها|يستخدمه|أي منها|تقنيات(?:ه|ها|هم))"),
}

# Extraction mécanique (jamais d'invention) d'un nom de projet près de
# "project"/"projet"/"مشروع". Le nom capturé DOIT être capitalisé ([A-Z]) :
# sans cette contrainte, un texte généré comme "...project completed in
# July..." capture "completed" comme s'il s'agissait d'un nom. Le mot-clé
# reste insensible à la casse via `(?i:...)`.
_NAME_AFTER_KEYWORD: dict[str, re.Pattern] = {
    "en": re.compile(r"(?i:project)\s+([A-Z][\w-]*)"),
    "fr": re.compile(r"(?i:projet)\s+([A-Z][\w-]*)"),
    "ar": re.compile(r"(?:مشروع|المشروع)\s+([A-Z][\w-]*)"),
}
_NAME_BEFORE_KEYWORD: dict[str, re.Pattern] = {
    "en": re.compile(r"\b([A-Z][\w-]*)\s+(?i:project)\b"),
    "fr": re.compile(r"\b([A-Z][\w-]*)\s+(?i:projet)\b"),
}
# Repli pour "Tell me about AGEP." (sans le mot "project") — sinon la
# question ne porte aucun sujet détectable et finit classée OUT_OF_SCOPE.
_NAME_AFTER_INTRO: dict[str, re.Pattern] = {
    "en": re.compile(r"(?i:tell me (?:more )?about\s+(?:the\s+)?)([A-Z][\w-]*)\b"),
    "fr": re.compile(r"(?i:parle-moi (?:plus )?d[eu]'?\s*(?:la\s+|le\s+|l[’'])?)([A-Z][\w-]*)\b"),
    "ar": re.compile(r"حدثني عن\s+([A-Z][\w-]*)"),
}

_NAME_STOPWORDS = {
    "this", "that", "the", "a", "an", "his", "her", "its", "it", "them", "these", "those", "he", "she",
    "ce", "cette", "cet", "le", "la", "un", "une", "du", "au", "son", "sa", "lui", "elle", "il", "eux",
}

# Deux entités coordonnées ("AGEP and Project B") dans le même tour : "it"
# n'a alors plus de référent unique -> clarification plutôt qu'un choix
# arbitraire. Limité à "X and/et/و Y" (pas de NER complet).
_MULTI_ENTITY_PATTERNS: dict[str, re.Pattern] = {
    "en": re.compile(r"\b([A-Z][A-Za-z0-9]{1,20})\b\s+and\s+(?:the\s+)?(?:project\s+)?\b([A-Z][A-Za-z0-9]{1,20})\b"),
    "fr": re.compile(r"\b([A-Z][A-Za-z0-9]{1,20})\b\s+et\s+(?:le\s+projet\s+)?\b([A-Z][A-Za-z0-9]{1,20})\b"),
    "ar": re.compile(r"\b([A-Za-z][A-Za-z0-9]{1,20})\b\s*و\s*(?:مشروع\s+)?\b([A-Za-z][A-Za-z0-9]{1,20})\b"),
}


def _has_multiple_entity_mentions(text: str, lang: str) -> bool:
    pattern = _MULTI_ENTITY_PATTERNS.get(lang)
    if not pattern:
        return False
    match = pattern.search(text)
    if not match:
        return False
    first, second = match.group(1).lower(), match.group(2).lower()
    return first not in _NAME_STOPWORDS and second not in _NAME_STOPWORDS and first != second


def _is_referential(query: str, lang: str) -> bool:
    pattern = _REFERENTIAL_PATTERNS.get(lang)
    return bool(pattern and pattern.search(query)) or detect_ambiguous(query, lang)


def _first_valid_name(pattern: re.Pattern | None, text: str) -> str | None:
    if not pattern:
        return None
    for match in pattern.finditer(text):
        candidate = match.group(1)
        if candidate and candidate.lower() not in _NAME_STOPWORDS:
            return candidate
    return None


def _extract_focus(text: str, lang: str) -> tuple[str | None, str | None]:
    """(topic, entity_name) extraits mécaniquement (jamais d'invention).
    Ordre du plus sûr au plus permissif : "project NAME", puis "NAME
    project", puis "tell me about NAME" en dernier repli."""
    topic = detect_topic(text, lang)
    entity_name = _first_valid_name(_NAME_AFTER_KEYWORD.get(lang), text)
    if entity_name is None:
        entity_name = _first_valid_name(_NAME_BEFORE_KEYWORD.get(lang), text)
    if entity_name is None:
        entity_name = _first_valid_name(_NAME_AFTER_INTRO.get(lang), text)
    return topic, entity_name


def resolve_reference(query: str, lang: str, conversation: list[Turn] | None) -> ReferenceResolution:
    """N'intervient que si la question seule ne porte aucun sujet
    identifiable et contient une marque référentielle claire. Sinon
    inchangée : app/rag/scope.py gère le reste (hors-sujet vs portfolio-lié)."""
    own_topic, own_entity = _extract_focus(query, lang)
    if own_topic is not None or own_entity is not None:
        return ReferenceResolution(query, False, own_topic, own_entity, own_entity or own_topic, False)

    if not _is_referential(query, lang):
        return ReferenceResolution(query, False, None, None, None, False)

    # Une entité précise (ex. "AGEP"), même dans un tour plus ancien, prime
    # sur un sujet générique trouvé dans un tour plus récent. Le sujet le
    # plus récent sert de repli si aucune entité n'est trouvée.
    best_topic: str | None = None
    for turn in reversed((conversation or [])[-MAX_TURNS * 2:]):
        turn_lang = detect_language(turn.content)
        if _has_multiple_entity_mentions(turn.content, turn_lang):
            return ReferenceResolution(query, True, None, None, None, False)
        topic, entity_name = _extract_focus(turn.content, turn_lang)
        if entity_name:
            return ReferenceResolution(f"{query} {entity_name}", False, topic or best_topic, entity_name, entity_name, True)
        if topic and best_topic is None:
            best_topic = topic

    if best_topic:
        return ReferenceResolution(f"{query} {best_topic}", False, best_topic, None, best_topic, True)

    return ReferenceResolution(query, True, None, None, None, False)
