"""Chunking sémantique : respecte la structure logique produite par le parser
(un ContentBlock par section/subsection), subdivisé par paragraphe seulement
si sa taille dépasse un seuil raisonnable. `chunk_id` dérive de (langue,
section, sous-section, index local) — des clés machine stables, pas du texte
traduit — donc indépendant de l'ordre du fichier et de la langue."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from app.config import settings
from app.ingestion.parser import ContentBlock

_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?؟।])\s+")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    lang: str
    section: str
    subsection: str
    part_index: int
    text: str
    chunk_hash: str
    topic_group: str
    entity_type: str
    entity_id: str


def _make_chunk_id(lang: str, section: str, subsection: str, part_index: int) -> str:
    digest = hashlib.sha1(f"{lang}|{section}|{subsection}|{part_index}".encode("utf-8")).hexdigest()
    return f"chk_{digest[:20]}"


def _breadcrumb(lang: str, label: str) -> str:
    return f"[{lang.upper()} > {label}]"


def _split_long_paragraph(paragraph: str, max_chars: int) -> list[str]:
    """Découpe un paragraphe trop long par phrase plutôt qu'à un nombre de
    caractères arbitraire, pour ne jamais couper une idée en plein milieu."""
    sentences = _SENTENCE_SPLIT_RE.split(paragraph)
    parts: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) > max_chars and current:
            parts.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts or [paragraph]


def _group_paragraphs(paragraphs: list[str], max_chars: int, min_chars: int) -> list[str]:
    groups: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            for sub in _split_long_paragraph(paragraph, max_chars):
                current = _append_or_flush(sub, current, max_chars, groups)
        else:
            current = _append_or_flush(paragraph, current, max_chars, groups)
    if current:
        groups.append(current)

    # Fusionne un dernier groupe trop petit avec le précédent pour éviter
    # des chunks orphelins de quelques mots.
    if len(groups) >= 2 and len(groups[-1]) < min_chars:
        groups[-2] = f"{groups[-2]}\n\n{groups[-1]}"
        groups.pop()
    return groups


def _append_or_flush(piece: str, current: str, max_chars: int, groups: list[str]) -> str:
    candidate = f"{current}\n\n{piece}" if current else piece
    if len(candidate) > max_chars and current:
        groups.append(current)
        return piece
    return candidate


def chunk_content_blocks(blocks: list[ContentBlock]) -> list[Chunk]:
    max_chars = settings.chunk_max_chars
    min_chars = settings.chunk_min_chars
    chunks: list[Chunk] = []

    for block in blocks:
        paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(block.text) if p.strip()]
        if not paragraphs:
            continue

        groups = _group_paragraphs(paragraphs, max_chars, min_chars)
        breadcrumb = _breadcrumb(block.lang, block.label)

        for part_index, group in enumerate(groups):
            full_text = f"{breadcrumb}\n{group}"
            chunk_id = _make_chunk_id(block.lang, block.section, block.subsection, part_index)
            chunk_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    lang=block.lang,
                    section=block.section,
                    subsection=block.subsection,
                    part_index=part_index,
                    topic_group=block.topic_group,
                    entity_type=block.entity_type,
                    entity_id=block.entity_id,
                    text=full_text,
                    chunk_hash=chunk_hash,
                )
            )

    return chunks
