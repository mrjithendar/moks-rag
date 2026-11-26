"""Markdown-aware chunking: split by section first, then by paragraph/sentence, with overlap inside
a section.

Every chunk keeps its heading path ("Install › Helm values"), and the text that gets embedded starts
with "<title> › <heading path>", so a chunk deep inside an article still says what it is about.
Chunks never exceed chunk_size characters, overlap included. Headings inside fenced code blocks are
code, not headings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SEPARATORS = ("\n\n", "\n", ". ", "? ", "! ", " ")
_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_FENCE_RE = re.compile(r"^[ \t]*(```|~~~)")
PATH_SEP = " › "


@dataclass(frozen=True)
class Chunk:
    """One piece of a document: its text and the headings it sits under."""

    text: str
    heading_path: str = ""


def _split_recursive(text: str, chunk_size: int, separators: tuple[str, ...]) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    if not separators:
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep = separators[0]
    if sep not in text:
        return _split_recursive(text, chunk_size, separators[1:])

    parts = text.split(sep)
    merged: list[str] = []
    current = ""

    for idx, part in enumerate(parts):
        piece = part if idx == len(parts) - 1 else part + sep
        if len(piece) > chunk_size:
            if current.strip():
                merged.append(current)
                current = ""
            merged.extend(_split_recursive(piece, chunk_size, separators[1:]))
            continue

        if len(current) + len(piece) <= chunk_size:
            current += piece
        else:
            if current.strip():
                merged.append(current)
            current = piece

    if current.strip():
        merged.append(current)

    return merged


def _clean_heading(title: str) -> str:
    # Drop Markdown emphasis/links from a heading so the path reads as plain text.
    title = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", title)
    return re.sub(r"[*_`]+", "", title).strip()


def split_sections(text: str) -> list[tuple[str, str]]:
    """(heading path, section text) in document order. The heading line stays in its section's
    text.
    """
    sections: list[tuple[str, str]] = []
    stack: list[str] = []
    lines: list[str] = []
    has_body = False
    in_fence = False

    def flush() -> None:
        nonlocal lines, has_body
        if lines and has_body:
            sections.append((PATH_SEP.join(stack), "\n".join(lines).strip()))
        lines, has_body = [], False

    for line in text.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
        heading = None if in_fence else _HEADING_RE.match(line)
        if heading:
            if has_body:
                flush()
            else:
                lines = []  # a heading directly under another heading: its path is enough
            level = len(heading.group(1))
            stack = [*stack[: level - 1], _clean_heading(heading.group(2))]
            lines.append(line)
            continue
        lines.append(line)
        if line.strip():
            has_body = True
    flush()
    return sections


def _with_overlap(pieces: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    if chunk_overlap == 0 or len(pieces) <= 1:
        return pieces
    out = [pieces[0]]
    for prev, curr in zip(pieces, pieces[1:], strict=False):
        room = min(chunk_overlap, chunk_size - len(curr) - 1)
        if room <= 0:
            out.append(curr)
            continue
        window = prev[-room:]
        tail = window.lstrip()
        # Avoid starting the overlap mid-word when possible.
        if len(prev) > room and " " in tail:
            tail = tail.split(" ", 1)[1].lstrip()
        if tail and not curr.startswith(tail):
            sep = "\n" if "\n" in window else " "
            out.append(f"{tail}{sep}{curr}")
        else:
            out.append(curr)
    return out


def chunk_markdown(text: str, chunk_size: int = 1500, chunk_overlap: int = 200) -> list[Chunk]:
    """Split a (Markdown or plain) document into chunks of at most chunk_size characters."""
    cleaned = text.strip()
    if not cleaned:
        return []
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be strictly smaller than chunk_size")

    body_size = max(1, chunk_size - chunk_overlap)
    chunks: list[Chunk] = []
    for path, section in split_sections(cleaned):
        pieces = [p.strip() for p in _split_recursive(section, body_size, _SEPARATORS) if p.strip()]
        chunks.extend(Chunk(t, path) for t in _with_overlap(pieces, chunk_size, chunk_overlap))
    return chunks


def chunk_text(text: str, chunk_size: int = 1500, chunk_overlap: int = 200) -> list[str]:
    """The chunks' text only."""
    return [c.text for c in chunk_markdown(text, chunk_size, chunk_overlap)]


def embedding_text(chunk: Chunk, title: str | None) -> str:
    """What gets embedded (dense and sparse): "<title> › <heading path>" above the chunk text."""
    parts: list[str] = []
    for part in ([title] if title else []) + (
        chunk.heading_path.split(PATH_SEP) if chunk.heading_path else []
    ):
        part = part.strip()
        if part and (not parts or parts[-1].casefold() != part.casefold()):
            parts.append(part)
    header = PATH_SEP.join(parts)
    return f"{header}\n\n{chunk.text}" if header else chunk.text
