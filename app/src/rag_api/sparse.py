"""Deterministic lexical BM25-style sparse vector encoder for Qdrant IDF sparse indexes.

Ollama's bge-m3 server returns dense 1024-dim vectors only. Pairing Qdrant's server-side
`Modifier.IDF` sparse index with deterministic hashed term-frequency vectors gives fast,
zero-extra-RAM lexical matching for hybrid RRF fusion on CPU-only ARM64 nodes.

Terms (index version v2): Unicode words (so "Größe" and Devanagari survive), case-folded, English
stopwords dropped, alphabetic words stemmed with the Snowball English stemmer (so "pods" matches
"pod"), and hyphenated or underscored terms indexed both whole and by part ("bge-m3" also gives
"bge" and "m3"). Changing any of this changes which terms match, so it is part of
config.INDEX_VERSION.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

import snowballstemmer

# A word: starts and ends with a letter or digit, may contain - or _ inside.
_TOKEN_RE = re.compile(r"[^\W_](?:[\w\-]*[^\W_])?")
_PART_RE = re.compile(r"[-_]+")

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "will",
        "with",
    }
)

_STEMMER = snowballstemmer.stemmer("english")


@dataclass(frozen=True)
class SparseVectorData:
    """Sparse vector representation compatible with Qdrant SparseVector."""

    indices: list[int]
    values: list[float]


def _token_to_index(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


@lru_cache(maxsize=65536)
def _normalize(word: str) -> str:
    return _STEMMER.stemWord(word) if word.isalpha() else word


def _keep(word: str) -> bool:
    # One-character ASCII terms ("a", "x", "1") carry no signal; one CJK character is a word.
    return word not in _STOPWORDS and (len(word) > 1 or not word.isascii())


def tokenize(text: str) -> list[str]:
    """The terms of a text, in order, as they are indexed and queried."""
    terms: list[str] = []
    for raw in _TOKEN_RE.findall(text.casefold()):
        parts = [p for p in _PART_RE.split(raw) if p]
        if len(parts) > 1:
            terms.append(raw)  # the whole term, unstemmed: "bge-m3", "max_tokens"
            terms.extend(_normalize(p) for p in parts if _keep(p))
        elif _keep(raw):
            terms.append(_normalize(raw))
    return terms


def encode_sparse_bm25(text: str, k1: float = 1.2) -> SparseVectorData:
    """Convert text into a sorted sparse vector of hashed term IDs and saturated TF weights."""
    tokens = tokenize(text)
    if not tokens:
        return SparseVectorData(indices=[], values=[])

    counts = Counter(tokens)
    weights: dict[int, float] = {}
    for token, tf in counts.items():
        idx = _token_to_index(token)
        # BM25 term-frequency saturation; Qdrant Modifier.IDF multiplies by IDF at query time.
        tf_weight = (tf * (k1 + 1.0)) / (tf + k1)
        weights[idx] = weights.get(idx, 0.0) + tf_weight

    sorted_items = sorted(weights.items(), key=lambda pair: pair[0])
    return SparseVectorData(
        indices=[idx for idx, _ in sorted_items],
        values=[round(val, 6) for _, val in sorted_items],
    )
