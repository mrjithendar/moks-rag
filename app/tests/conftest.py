"""Shared pytest fixtures and lightweight fakes for rag-api tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rag_api.config import CollectionName, Settings
from rag_api.main import create_app
from rag_api.models import RetrievedChunk
from rag_api.sparse import SparseVectorData, encode_sparse_bm25

TEST_TOKEN = "test-rag-bearer-token-123"


class FakeDoclingClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def parse_file(
        self,
        filename: str,
        content: bytes,
        content_type: str | None = None,
    ) -> str:
        self.calls.append(
            {"filename": filename, "size": len(content), "content_type": content_type}
        )
        return f"# Parsed {filename}\n\nExtracted PDF body content for hybrid indexing."


class FakeEmbeddingClient:
    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []

    async def embed_hybrid(
        self, texts: list[str]
    ) -> tuple[list[list[float]], list[SparseVectorData]]:
        self.calls.append(list(texts))
        dense = [[0.01 * (i + 1)] * self.dim for i in range(len(texts))]
        sparse = [encode_sparse_bm25(t) for t in texts]
        return dense, sparse


class FakeQdrantStore:
    def __init__(self) -> None:
        self.ensured = False
        self.docs: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self.deleted: list[tuple[str, str]] = []
        self.searches: list[dict[str, Any]] = []

    async def ensure_collections(self) -> None:
        self.ensured = True

    async def healthcheck(self) -> bool:
        return True

    async def close(self) -> None:
        return None

    async def upsert_chunks(
        self,
        collection: CollectionName,
        doc_id: str,
        chunks: list[str],
        dense_vectors: list[list[float]],
        sparse_vectors: list[SparseVectorData],
        metadata: dict[str, Any],
    ) -> int:
        assert len(chunks) == len(dense_vectors) == len(sparse_vectors)
        self.docs[(collection, doc_id)] = [{"chunk": c, "metadata": metadata} for c in chunks]
        return len(chunks)

    async def delete_document(self, collection: CollectionName, doc_id: str) -> None:
        self.deleted.append((collection, doc_id))
        self.docs.pop((collection, doc_id), None)

    async def hybrid_search(
        self,
        collection: CollectionName,
        dense_vector: list[float],
        sparse_vector: SparseVectorData,
        top_k: int = 6,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        self.searches.append(
            {
                "collection": collection,
                "dense_dim": len(dense_vector),
                "sparse_terms": len(sparse_vector.indices),
                "top_k": top_k,
                "filters": filters,
            }
        )
        return [
            RetrievedChunk(
                id="pt-1",
                doc_id="runbook-istio-503",
                score=0.9125,
                text=(
                    "Check PeerAuthentication STRICT mTLS and sidecar injection "
                    "on the rag namespace."
                ),
                metadata={"doc_id": "runbook-istio-503", "title": "Istio 503 Runbook"},
            )
        ]


class FakeLiteLLMClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def chat_completions(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int | None = 1024,
    ) -> dict[str, Any]:
        self.calls.append({"model": model, "messages": messages, "stream": False})
        return {
            "id": "chatcmpl-test-1",
            "created": 1764130000,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Grounded answer referencing [source: runbook-istio-503].",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 42, "completion_tokens": 12, "total_tokens": 54},
        }

    async def stream_chat_completions(
        self,
        model: str,
        virtual_model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int | None = 1024,
    ) -> AsyncIterator[str]:
        self.calls.append({"model": model, "virtual_model": virtual_model, "stream": True})
        yield (
            f'data: {{"id":"chatcmpl-stream-1","object":"chat.completion.chunk",'
            f'"model":"{virtual_model}","choices":[{{"index":0,"delta":'
            f'{{"role":"assistant","content":"Streamed RAG answer"}}}}]}}\n\n'
        )
        yield "data: [DONE]\n\n"


@pytest.fixture()
def fakes() -> dict[str, Any]:
    return {
        "docling": FakeDoclingClient(),
        "embeddings": FakeEmbeddingClient(),
        "qdrant": FakeQdrantStore(),
        "litellm": FakeLiteLLMClient(),
    }


@pytest.fixture()
def client(fakes: dict[str, Any]) -> TestClient:
    settings = Settings(
        rag_bearer_token=TEST_TOKEN,
        root_path="/rag",
        ensure_collections_on_startup=True,
    )
    app = create_app(
        settings,
        docling_client=fakes["docling"],
        embedding_client=fakes["embeddings"],
        qdrant_store=fakes["qdrant"],
        litellm_client=fakes["litellm"],
    )
    with TestClient(app) as test_client:
        yield test_client
