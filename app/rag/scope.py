"""Classification locale (aucun appel LLM) du périmètre d'une question :
PORTFOLIO_RELATED, AMBIGUOUS (référent introuvable) ou OUT_OF_SCOPE. Un terme
technique du portfolio ("React") dans une question générale ne rend pas la
question "portfolio-related" — il faut une référence explicite (mot-clé de
section, référence personnelle, ou résolution via la conversation). Décision
d'architecture : la distance d'embedding s'est révélée peu fiable pour cette
distinction (voir app/rag/retrieval.py), donc heuristique locale plutôt
qu'un appel LLM supplémentaire ; un cas non couvert est laissé au system
prompt (règle HORS SUJET) plutôt que deviné localement."""

from __future__ import annotations

import re
from enum import Enum

from app.rag.memory import ReferenceResolution


class Scope(str, Enum):
    PORTFOLIO_RELATED = "portfolio_related"
    AMBIGUOUS = "ambiguous"
    OUT_OF_SCOPE = "out_of_scope"


# Référence personnelle au développeur : signal qu'une question sans
# mot-clé de section porte quand même sur lui plutôt que sur une
# connaissance générale ("What technologies does he know?" vs "What is React?").
_PERSONAL_REFERENCE_PATTERNS: dict[str, re.Pattern] = {
    "en": re.compile(r"\b(he|his|him|issalmou|the developer)\b", re.IGNORECASE),
    "fr": re.compile(r"\b(il|son|sa|ses|lui|issalmou|le développeur)\b", re.IGNORECASE),
    # Le suffixe possessif arabe (ه/ها/هم) est collé au nom (pas un mot
    # séparé comme "he/il") : couvert explicitement pour "تقنيات".
    "ar": re.compile(r"(هو|له|لديه|عنده|يعرفها|يعرفه|يستخدمها|يستخدمه|إسلمو|اسلمو|تقنيات(ه|ها|هم))"),
}


def _has_personal_reference(query: str, lang: str) -> bool:
    pattern = _PERSONAL_REFERENCE_PATTERNS.get(lang)
    return bool(pattern and pattern.search(query))


def classify_scope(
    query: str,
    lang: str,
    topic: str | None,
    is_exhaustive: bool,
    resolution: ReferenceResolution,
) -> Scope:
    if is_exhaustive:
        return Scope.PORTFOLIO_RELATED
    if topic is not None:
        return Scope.PORTFOLIO_RELATED
    if resolution.applied:
        return Scope.PORTFOLIO_RELATED
    # Un focus_entity_name extrait de la question SEULE (resolution.applied
    # est False) n'est pas une preuve suffisante ici : le motif le plus
    # permissif capturerait aussi bien "React" que "AGEP". C'est
    # app/rag/retrieval.py qui vérifie via un lookup Chroma direct avant de
    # confirmer OUT_OF_SCOPE.
    if _has_personal_reference(query, lang):
        return Scope.PORTFOLIO_RELATED
    if resolution.needs_clarification:
        return Scope.AMBIGUOUS
    return Scope.OUT_OF_SCOPE
