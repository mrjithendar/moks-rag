"""Async clients for Docling Serve, BGE-M3 embeddings, Qdrant, and LiteLLM."""

from __future__ import annotations

import json
import logging
import os
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import httpx

from rag_api.config import COLLECTIONS, CollectionName, Settings
from rag_api.models import RetrievedChunk
from rag_api.sparse import SparseVectorData, encode_sparse_bm25

if TYPE_CHECKING:
    from qdrant_client import AsyncQdrantClient

logger = logging.getLogger("rag_api.clients")

_PLAIN_TEXT_EXTENSIONS = frozenset(
    {".txt", ".md", ".mdx", ".rst", ".csv", ".json", ".yaml", ".yml"}
)
_PLAIN_TEXT_MIMES = frozenset(
    {"text/plain", "text/markdown", "text/x-markdown", "application/json"}
)


class DoclingClient:
    """Converts binary/rich documents (PDF, DOCX, HTML, etc.) to Markdown via docling-serve."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self._base_url = settings.docling_url.rstrip("/")
        self._client = client

    @staticmethod
    def needs_docling(filename: str | None, content_type: str | None) -> bool:
        ext = os.path.splitext((filename or "").lower())[1]
        mime = (content_type or "").split(";")[0].strip().lower()
        return not (ext in _PLAIN_TEXT_EXTENSIONS or mime in _PLAIN_TEXT_MIMES)

    async def parse_file(
        self,
        filename: str,
        content: bytes,
        content_type: str | None = None,
    ) -> str:
        """Send file bytes to docling-serve /v1/convert/file and extract Markdown text."""
        url = f"{self._base_url}/v1/convert/file"
        files = {
            "files": (
                filename or "document",
                content,
                content_type or "application/octet-stream",
            )
        }
        data = {"to_formats": "md"}

        response = await self._client.post(url, files=files, data=data)
        response.raise_for_status()
        payload = response.json()

        # docling-serve returns {"document": {"md_content": ...}} or {"documents": [...]}
        document = payload.get("document")
        if isinstance(document, dict):
            extracted = (
                document.get("md_content")
                or document.get("text_content")
                or document.get("html_content")
            )
            if isinstance(extracted, str) and extracted.strip():
                return extracted.strip()

        documents = payload.get("documents")
        if isinstance(documents, list) and documents:
            first = documents[0]
            if isinstance(first, dict):
                extracted = first.get("md_content") or first.get("text_content")
                if isinstance(extracted, str) and extracted.strip():
                    return extracted.strip()

        raise ValueError("docling-serve returned an empty document payload")


class EmbeddingClient:
    """Generates 1024-dim dense vectors via bge-m3 and sparse lexical vectors for hybrid RRF."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self._base_url = settings.bge_m3_url.rstrip("/")
        self._model = settings.bge_m3_model
        self._api_style = settings.bge_m3_api_style
        self._dim = settings.embedding_dim
        self._batch_size = settings.embedding_batch_size
        self._sparse_mode = settings.sparse_mode
        self._client = client

    async def embed_dense(self, texts: list[str]) -> list[list[float]]:
        """Generate dense 1024-dim embeddings via bge-m3 in bounded batches."""
        if not texts:
            return []

        all_vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            vectors = await self._embed_dense_batch(batch)
            for vec in vectors:
                if len(vec) != self._dim:
                    raise ValueError(
                        f"Expected {self._dim}-dim embedding from bge-m3, got {len(vec)}"
                    )
            all_vectors.extend(vectors)
        return all_vectors

    async def _embed_dense_batch(self, batch: list[str]) -> list[list[float]]:
        if self._api_style == "ollama":
            url = f"{self._base_url}/api/embed"
            resp = await self._client.post(url, json={"model": self._model, "input": batch})
            resp.raise_for_status()
            payload = resp.json()
            embeddings = payload.get("embeddings")
            if not isinstance(embeddings, list) or len(embeddings) != len(batch):
                raise ValueError("Unexpected /api/embed response from bge-m3")
            return embeddings

        url = f"{self._base_url}/v1/embeddings"
        resp = await self._client.post(url, json={"model": self._model, "input": batch})
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data")
        if not isinstance(data, list) or len(data) != len(batch):
            raise ValueError("Unexpected /v1/embeddings response from bge-m3")
        ordered = sorted(data, key=lambda item: item.get("index", 0))
        return [item["embedding"] for item in ordered]

    async def embed_sparse(self, texts: list[str]) -> list[SparseVectorData]:
        """Generate lexical sparse vectors (BM25/IDF default, optional upstream sparse endpoint)."""
        if self._sparse_mode == "bge_m3":
            try:
                url = f"{self._base_url}/embed_sparse"
                resp = await self._client.post(url, json={"inputs": texts})
                resp.raise_for_status()
                payload = resp.json()
                if isinstance(payload, list) and len(payload) == len(texts):
                    results: list[SparseVectorData] = []
                    for item in payload:
                        if isinstance(item, dict) and "indices" in item and "values" in item:
                            results.append(
                                SparseVectorData(
                                    indices=[int(i) for i in item["indices"]],
                                    values=[float(v) for v in item["values"]],
                                )
                            )
                        else:
                            raise ValueError("Malformed sparse response item")
                    return results
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Upstream bge-m3 sparse endpoint unavailable (%s); falling back to BM25",
                    exc,
                )

        return [encode_sparse_bm25(text) for text in texts]

    async def embed_hybrid(
        self, texts: list[str]
    ) -> tuple[list[list[float]], list[SparseVectorData]]:
        """Produce aligned dense and sparse vectors for each input text."""
        dense = await self.embed_dense(texts)
        sparse = await self.embed_sparse(texts)
        return dense, sparse


class QdrantStore:
    """Manages isolated Qdrant collections with named dense + sparse vectors and RRF fusion."""

    def __init__(self, settings: Settings, client: AsyncQdrantClient | None = None) -> None:
        from qdrant_client import AsyncQdrantClient

        self._dim = settings.embedding_dim
        self._prefetch_limit = settings.prefetch_limit
        self._collections_ensured = False
        self._client = client or AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            timeout=settings.http_timeout_seconds,
        )

    async def close(self) -> None:
        await self._client.close()

    async def healthcheck(self) -> bool:
        await self._client.get_collections()
        if not self._collections_ensured:
            await self.ensure_collections()
        return True

    async def ensure_collections(self) -> None:
        """Create the 3 isolated hybrid collections and doc_id payload indexes if missing."""
        from qdrant_client import models

        existing_resp = await self._client.get_collections()
        existing = {c.name for c in existing_resp.collections}

        for name in COLLECTIONS:
            if name not in existing:
                await self._client.create_collection(
                    collection_name=name,
                    vectors_config={
                        "dense": models.VectorParams(
                            size=self._dim,
                            distance=models.Distance.COSINE,
                        )
                    },
                    sparse_vectors_config={
                        "sparse": models.SparseVectorParams(
                            modifier=models.Modifier.IDF,
                        )
                    },
                )
                logger.info("Created Qdrant hybrid collection: %s", name)

            await self._client.create_payload_index(
                collection_name=name,
                field_name="doc_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

        self._collections_ensured = True

    async def _ensure_ready(self) -> None:
        if not self._collections_ensured:
            await self.ensure_collections()

    async def upsert_chunks(
        self,
        collection: CollectionName,
        doc_id: str,
        chunks: list[str],
        dense_vectors: list[list[float]],
        sparse_vectors: list[SparseVectorData],
        metadata: dict[str, Any],
    ) -> int:
        """Replace existing chunks for doc_id and upsert fresh dense+sparse points."""
        from qdrant_client import models

        await self._ensure_ready()
        # Remove prior chunks for this doc_id so re-indexing never leaves orphan tail chunks.
        await self.delete_document(collection=collection, doc_id=doc_id)

        points: list[models.PointStruct] = []
        for idx, (chunk, dense_vec, sparse_vec) in enumerate(
            zip(chunks, dense_vectors, sparse_vectors, strict=True)
        ):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{collection}:{doc_id}:{idx}"))
            payload = {
                **metadata,
                "doc_id": doc_id,
                "chunk_index": idx,
                "chunk_count": len(chunks),
                "text": chunk,
            }
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense_vec,
                        "sparse": models.SparseVector(
                            indices=sparse_vec.indices,
                            values=sparse_vec.values,
                        ),
                    },
                    payload=payload,
                )
            )

        await self._client.upsert(collection_name=collection, points=points, wait=True)
        return len(points)

    async def delete_document(self, collection: CollectionName, doc_id: str) -> None:
        """Delete all chunks matching doc_id from the target collection."""
        from qdrant_client import models

        await self._ensure_ready()
        await self._client.delete(
            collection_name=collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id",
                            match=models.MatchValue(value=doc_id),
                        )
                    ]
                )
            ),
            wait=True,
        )

    async def hybrid_search(
        self,
        collection: CollectionName,
        dense_vector: list[float],
        sparse_vector: SparseVectorData,
        top_k: int = 6,
        filters: dict[str, str | int | float | bool] | None = None,
    ) -> list[RetrievedChunk]:
        """Execute Qdrant hybrid prefetch (dense + sparse) with Reciprocal Rank Fusion (RRF)."""
        from qdrant_client import models

        await self._ensure_ready()
        query_filter: models.Filter | None = None
        if filters:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(key=key, match=models.MatchValue(value=value))
                    for key, value in filters.items()
                ]
            )

        prefetch_k = max(self._prefetch_limit, top_k * 3)
        prefetches: list[models.Prefetch] = [
            models.Prefetch(
                query=dense_vector,
                using="dense",
                limit=prefetch_k,
                filter=query_filter,
            )
        ]

        if sparse_vector.indices:
            prefetches.append(
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_vector.indices,
                        values=sparse_vector.values,
                    ),
                    using="sparse",
                    limit=prefetch_k,
                    filter=query_filter,
                )
            )

        result = await self._client.query_points(
            collection_name=collection,
            prefetch=prefetches,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
            query_filter=query_filter,
        )

        chunks: list[RetrievedChunk] = []
        for point in result.points:
            payload = dict(point.payload or {})
            text = str(payload.pop("text", ""))
            doc_id = str(payload.get("doc_id", ""))
            chunks.append(
                RetrievedChunk(
                    id=str(point.id),
                    doc_id=doc_id,
                    score=round(float(point.score), 6),
                    text=text,
                    metadata=payload,
                )
            )
        return chunks


class LiteLLMClient:
    """OpenAI-compatible client targeting internal LiteLLM proxy."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self._base_url = settings.litellm_url.rstrip("/")
        self._api_key = settings.litellm_api_key
        self._client = client

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def chat_completions(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int | None = 1024,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        resp = await self._client.post(
            f"{self._base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()

    async def stream_chat_completions(
        self,
        model: str,
        virtual_model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int | None = 1024,
    ) -> AsyncIterator[str]:
        """Proxy SSE chunks from LiteLLM while presenting the selected virtual model name."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        async with self._client.stream(
            "POST",
            f"{self._base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                if line == "data: [DONE]":
                    yield "data: [DONE]\n\n"
                    continue
                if line.startswith("data: "):
                    raw_json = line.removeprefix("data: ").strip()
                    try:
                        chunk_obj = json.loads(raw_json)
                        if isinstance(chunk_obj, dict):
                            chunk_obj["model"] = virtual_model
                            yield f"data: {json.dumps(chunk_obj)}\n\n"
                            continue
                    except json.JSONDecodeError:
                        pass
                yield f"{line}\n\n"
