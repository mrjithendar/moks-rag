"""Recursive text chunking with overlap tailored for bge-m3 2048-token context."""

from __future__ import annotations

_SEPARATORS = ("\n\n", "\n", ". ", "? ", "! ", " ")


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


def chunk_text(text: str, chunk_size: int = 1500, chunk_overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks while respecting paragraph and sentence boundaries."""
    cleaned = text.strip()
    if not cleaned:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be strictly smaller than chunk_size")

    raw_chunks = [
        c.strip() for c in _split_recursive(cleaned, chunk_size, _SEPARATORS) if c.strip()
    ]
    if chunk_overlap == 0 or len(raw_chunks) <= 1:
        return raw_chunks

    overlapped: list[str] = [raw_chunks[0]]
    for prev, curr in zip(raw_chunks, raw_chunks[1:], strict=False):
        tail = prev[-chunk_overlap:].lstrip()
        # Avoid cutting the first word of the overlap tail mid-word when possible
        if len(prev) > chunk_overlap and " " in tail:
            tail = tail.split(" ", 1)[1].lstrip()
        if tail and not curr.startswith(tail):
            sep = "\n" if "\n" in prev[-chunk_overlap:] else " "
            overlapped.append(f"{tail}{sep}{curr}")
        else:
            overlapped.append(curr)

    return overlapped
