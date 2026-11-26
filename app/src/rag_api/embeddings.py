"""Dense vectors from bge-m3 (Ollama) and lexical sparse vectors for hybrid RRF."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from rag_api import metrics
from rag_api.config import Settings
from rag_api.errors import EmbeddingInputTooLong, UpstreamError, from_response, upstream
from rag_api.logs import emit
from rag_api.sparse import SparseVectorData, encode_sparse_bm25

logger = logging.getLogger("rag_api.embeddings")

DEPENDENCY = "bge-m3"


def _json(resp: httpx.Response) -> dict[str, Any]:
    try:
        body = resp.json()
    except ValueError as exc:
        raise UpstreamError(DEPENDENCY, "answered with something that is not JSON") from exc
    if not isinstance(body, dict):
        raise UpstreamError(DEPENDENCY, "answered with JSON that is not an object")
    return body


class EmbeddingClient:
    """Generates 1024-dim dense vectors via bge-m3 and sparse lexical vectors for hybrid RRF."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self._base_url = settings.bge_m3_url.rstrip("/")
        self._model = settings.bge_m3_model
        self._api_style = settings.bge_m3_api_style
        self._dim = settings.embedding_dim
        self._batch_size = settings.embedding_batch_size
        self._client = client
        if self._api_style == "openai":
            emit(
                logger,
                "embedding_truncation_undetectable",
                level=logging.WARNING,
                detail=(
                    "BGE_M3_API_STYLE=openai: chunks longer than the model context are "
                    "embedded from their first part only"
                ),
            )

    @property
    def model(self) -> str:
        return self._model

    async def embed_dense(self, texts: list[str], *, truncate: bool) -> list[list[float]]:
        """Dense embeddings in bounded batches. truncate=False refuses inputs longer than the
        context.
        """
        if not texts:
            return []
        all_vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            vectors = await self._embed_dense_batch(batch, truncate=truncate)
            for vec in vectors:
                if not isinstance(vec, list) or len(vec) != self._dim:
                    size = len(vec) if isinstance(vec, list) else type(vec).__name__
                    raise UpstreamError(
                        DEPENDENCY, f"returned a {size}-dim embedding, expected {self._dim}"
                    )
            all_vectors.extend(vectors)
        return all_vectors

    async def _embed_dense_batch(self, batch: list[str], *, truncate: bool) -> list[list[float]]:
        if self._api_style == "ollama":
            body = {"model": self._model, "input": batch, "truncate": truncate}
            with upstream(DEPENDENCY):
                resp = await self._client.post(f"{self._base_url}/api/embed", json=body)
            if resp.status_code >= 400:
                text = resp.text
                if (
                    not truncate
                    and resp.status_code in (400, 500)
                    and "context length" in text.lower()
                ):
                    raise EmbeddingInputTooLong(
                        "A chunk is longer than the embedding model's context; send a smaller "
                        "chunk_size"
                    )
                raise from_response(DEPENDENCY, resp, text[:300])
            embeddings = _json(resp).get("embeddings")
            if not isinstance(embeddings, list) or len(embeddings) != len(batch):
                raise UpstreamError(
                    DEPENDENCY, "answered /api/embed without one embedding per input"
                )
            return embeddings

        with upstream(DEPENDENCY):
            resp = await self._client.post(
                f"{self._base_url}/v1/embeddings", json={"model": self._model, "input": batch}
            )
            resp.raise_for_status()
        data = _json(resp).get("data")
        if not isinstance(data, list) or len(data) != len(batch):
            raise UpstreamError(
                DEPENDENCY, "answered /v1/embeddings without one embedding per input"
            )
        try:
            ordered = sorted(data, key=lambda item: item.get("index", 0))
            return [item["embedding"] for item in ordered]
        except (AttributeError, KeyError, TypeError) as exc:
            raise UpstreamError(DEPENDENCY, "answered /v1/embeddings in an unknown shape") from exc

    @staticmethod
    def embed_sparse(texts: list[str]) -> list[SparseVectorData]:
        """Lexical BM25-style sparse vectors (Qdrant applies IDF)."""
        return [encode_sparse_bm25(text) for text in texts]

    async def embed_documents(
        self, texts: list[str]
    ) -> tuple[list[list[float]], list[SparseVectorData]]:
        """Aligned dense + sparse vectors for chunks; a chunk too long to embed whole is refused."""
        with metrics.stage("embed_documents"):
            dense = await self.embed_dense(texts, truncate=False)
        return dense, self.embed_sparse(texts)

    async def embed_query(self, text: str) -> tuple[list[float], SparseVectorData]:
        """Dense + sparse vectors for one query; a long query is cut to the context (its start
        matters).
        """
        with metrics.stage("embed_query"):
            dense = await self.embed_dense([text], truncate=True)
        return dense[0], self.embed_sparse([text])[0]
