"""Deterministic lexical BM25-style sparse vector encoder for Qdrant IDF sparse indexes.

Ollama's bge-m3 server returns dense 1024-dim vectors only. Pairing Qdrant's server-side
`Modifier.IDF` sparse index with deterministic hashed term-frequency vectors gives fast,
zero-extra-RAM lexical matching for hybrid RRF fusion on CPU-only ARM64 nodes.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_\-]{1,}")

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


@dataclass(frozen=True)
class SparseVectorData:
    """Sparse vector representation compatible with Qdrant SparseVector."""

    indices: list[int]
    values: list[float]


def _token_to_index(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def encode_sparse_bm25(text: str, k1: float = 1.2) -> SparseVectorData:
    """Convert text into a sorted sparse vector of hashed token IDs and saturated TF weights."""
    tokens = [tok for tok in _TOKEN_RE.findall(text.lower()) if tok not in _STOPWORDS]
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
