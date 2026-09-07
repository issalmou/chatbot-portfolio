"""Détection locale de la langue (aucun appel LLM) et protection du nom propre."""

from __future__ import annotations

import re

from app.config import settings

_ARABIC_RANGE = re.compile(r"[؀-ۿ]")

# Mots-outils très fréquents, utilisés uniquement en repli lorsque langdetect
# échoue ou n'est pas fiable sur un texte très court.
_FR_STOPWORDS = {
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "est", "quels",
    "quelles", "quel", "quelle", "que", "qui", "pour", "avec", "sur", "dans",
    "quoi", "comment", "pourquoi", "tu", "vous", "il", "elle", "ses", "son",
}
_EN_STOPWORDS = {
    "the", "a", "an", "is", "are", "what", "which", "who", "how", "why",
    "for", "with", "on", "in", "of", "you", "his", "her", "does", "do",
}

# Questions à un seul mot-sujet dont l'orthographe distingue la langue sans
# langdetect (peu fiable sur un texte aussi court — "Projects?"/"Skills?"
# étaient mal classés "fr"). Dupliqué plutôt qu'importé depuis app/rag/sections.py :
# ce module reste une brique bas niveau sans dépendance vers app/rag/.
_EN_ONLY_SHORT_WORDS = {"projects", "project", "skills", "skill", "experience", "education"}
_FR_ONLY_SHORT_WORDS = {"projets", "projet", "compétences", "competences", "compétence", "expérience", "formation"}
# Orthographe identique dans les deux langues : seule la typographie de
# ponctuation distingue ("Contact ?" avec espace = fr ; "Contact?" sans = en).
_AMBIGUOUS_SHORT_WORDS = {"contact", "services", "service"}
_FRENCH_TERMINAL_SPACING = re.compile(r"\s[?!:;]")

LANG_LABELS = {"fr": "français", "en": "anglais", "ar": "arabe"}

SUPPORTED_LANGS = ("fr", "en", "ar")


def _short_word_language(stripped: str) -> str | None:
    """Cas des questions à un seul mot-sujet (voir docstring des ensembles
    ci-dessus). None si le texte n'est pas de cette forme (plusieurs mots) —
    on laisse alors la suite de detect_language() décider normalement."""
    word = re.sub(r"[?!.:;,]+$", "", stripped).strip().lower()
    if not word or " " in word:
        return None
    if word in _EN_ONLY_SHORT_WORDS:
        return "en"
    if word in _FR_ONLY_SHORT_WORDS:
        return "fr"
    if word in _AMBIGUOUS_SHORT_WORDS:
        return "fr" if _FRENCH_TERMINAL_SPACING.search(stripped) else "en"
    return None


def detect_language(text: str) -> str:
    """Détecte fr/en/ar localement, sans appel réseau : (1) arabe via la
    plage Unicode ; (2) mot-sujet court résolu par orthographe/typographie
    (_short_word_language), plus fiable que langdetect sur texte court ;
    (3) décompte de mots-outils invariants (quels/et/il vs what/does/his) s'il
    est asymétrique — langdetect seul se laisse tromper par des emprunts
    techniques anglais dans une phrase française par ailleurs sans ambiguïté ;
    (4) sinon langdetect ; (5) en dernier repli, le même décompte, tie-break vers fr."""
    stripped = text.strip()
    if not stripped:
        return "fr"

    if _ARABIC_RANGE.search(stripped):
        return "ar"

    short = _short_word_language(stripped)
    if short:
        return short

    tokens = set(re.findall(r"[a-zà-ÿ']+", stripped.lower()))
    fr_hits = len(tokens & _FR_STOPWORDS)
    en_hits = len(tokens & _EN_STOPWORDS)
    if fr_hits and not en_hits:
        return "fr"
    if en_hits and not fr_hits:
        return "en"

    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0  # résultats déterministes
        code = detect(stripped)
        if code == "fr":
            return "fr"
        if code == "en":
            return "en"
        if code == "ar":
            return "ar"
    except Exception:
        pass

    if en_hits > fr_hits:
        return "en"
    return "fr"


def language_label(lang_code: str) -> str:
    return LANG_LABELS.get(lang_code, "français")


_PLACEHOLDER = "⟦ISSALMOU_ADAAICHE⟧"


def protect_name(text: str) -> str:
    """Remplace le nom propre par un jeton neutre avant un appel LLM : un
    jeton fait de symboles/majuscules risque bien moins d'être traduit ou
    reformulé qu'un nom propre en toutes lettres."""
    return text.replace(settings.protected_name, _PLACEHOLDER)


def restore_name(text: str) -> str:
    return text.replace(_PLACEHOLDER, settings.protected_name)


# Formes translittérées du nom trouvées dans le contenu source arabe.
# protect_name() ne protège que la forme latine exacte ; ce filtre
# déterministe est une garantie supplémentaire indépendante du LLM, appliquée
# après restore_name() — formes les plus longues en premier pour éviter tout résidu partiel.
_KNOWN_TRANSLITERATIONS = (
    "اسلمو إيدعيش",
    "إسلمو إيدعيش",
    "إسلمو",
    "اسلمو",
)


def sanitize_name_transliterations(text: str) -> str:
    """Filet de sécurité déterministe : remplace toute translittération
    arabe connue du nom par la forme latine exacte, indépendamment de ce
    que le LLM a réellement produit."""
    for variant in _KNOWN_TRANSLITERATIONS:
        text = text.replace(variant, settings.protected_name)
    return text
