"""Shared fixtures: a real QdrantStore on in-memory Qdrant, a deterministic embedder, fakes for
docling-serve and LiteLLM.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from qdrant_client import AsyncQdrantClient

from rag_api.config import Settings
from rag_api.errors import EmbeddingInputTooLong
from rag_api.llm import DONE
from rag_api.main import create_app
from rag_api.sparse import SparseVectorData, encode_sparse_bm25, tokenize
from rag_api.store import QdrantStore

TEST_TOKEN = "test-rag-bearer-token-123"
DIM = 32
TOKENS = {
    "moks-cms": "cms-token-0123456789abcdef",
    "moks-seo-agent": "seo-token-0123456789abcdef",
    "moks-sre-agent": "sre-token-0123456789abcdef",
    "open-webui": "owui-token-0123456789abcdef",
    "rag-admin": "admin-token-0123456789abcdef",
    "no-policy": "nopolicy-token-0123456789ab",
}
POLICY = {
    "moks-cms": {"read": ["cms_content"], "write": ["cms_content"]},
    "moks-seo-agent": {"read": ["seo_intel", "cms_content"], "write": ["seo_intel"]},
    "moks-sre-agent": {"read": ["sre_knowledge"], "write": ["sre_knowledge"]},
    "open-webui": {"read": ["*"]},
    "rag-admin": {"read": ["*"], "write": ["*"], "admin": True},
}


def auth(token: str = TEST_TOKEN, service: str = "moks-cms") -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "X-Service-Name": service}


def svc(service: str) -> dict[str, str]:
    return auth(TOKENS[service], service)


def bag_vector(text: str, dim: int = DIM) -> list[float]:
    """A deterministic lexical 'embedding': stemmed terms hashed into buckets, L2-normalized."""
    vec = [0.0] * dim
    for term in tokenize(text):
        vec[int(hashlib.md5(term.encode()).hexdigest(), 16) % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec] if any(vec) else [1.0 / math.sqrt(dim)] * dim


class FakeEmbeddingClient:
    model = "fake-bag-of-words"

    def __init__(self, dim: int = DIM, max_chars: int | None = None) -> None:
        self.dim = dim
        self.max_chars = max_chars
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    async def embed_documents(
        self, texts: list[str]
    ) -> tuple[list[list[float]], list[SparseVectorData]]:
        if self.max_chars is not None and any(len(t) > self.max_chars for t in texts):
            raise EmbeddingInputTooLong("A chunk is longer than the embedding model's context")
        self.document_calls.append(list(texts))
        return [bag_vector(t, self.dim) for t in texts], [encode_sparse_bm25(t) for t in texts]

    async def embed_query(self, text: str) -> tuple[list[float], SparseVectorData]:
        self.query_calls.append(text)
        return bag_vector(text, self.dim), encode_sparse_bm25(text)


class FakeDoclingClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def parse_file(
        self, filename: str, content: bytes, content_type: str | None = None
    ) -> str:
        self.calls.append(
            {"filename": filename, "size": len(content), "content_type": content_type}
        )
        return f"# Parsed {filename}\n\nExtracted PDF body content for hybrid indexing."


class FakeStream:
    def __init__(self, events: list[Any], fail_after: int | None = None) -> None:
        self._events = events
        self._fail_after = fail_after
        self.usage: dict[str, Any] | None = None
        self.finish_reason: str | None = None
        self.closed = False

    async def events(self) -> AsyncIterator[dict[str, Any] | str]:
        from rag_api.errors import UpstreamError

        for i, event in enumerate(self._events):
            if self._fail_after is not None and i >= self._fail_after:
                raise UpstreamError("litellm", "connection dropped", kind="unavailable")
            if isinstance(event, dict) and isinstance(event.get("usage"), dict):
                self.usage = event["usage"]
            yield event

    async def aclose(self) -> None:
        self.closed = True


class FakeLiteLLMClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.answers: list[str] = []
        self.open_error: Exception | None = None
        self.stream_fail_after: int | None = None
        self.streams: list[FakeStream] = []

    def _answer(self, model: str, messages: list[dict[str, Any]]) -> str:
        if self.answers:
            return self.answers.pop(0)
        return "Grounded answer referencing [source: runbook-istio-503]."

    async def chat_completions(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int | None = 1024,
    ) -> dict[str, Any]:
        self.calls.append(
            {"model": model, "messages": messages, "stream": False, "max_tokens": max_tokens}
        )
        return {
            "id": "chatcmpl-test-1",
            "created": 1764130000,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": self._answer(model, messages)},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 42, "completion_tokens": 12, "total_tokens": 54},
        }

    async def open_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int | None = 1024,
    ) -> FakeStream:
        self.calls.append({"model": model, "messages": messages, "stream": True})
        if self.open_error is not None:
            raise self.open_error
        stream = FakeStream(
            [
                {
                    "id": "chatcmpl-stream-1",
                    "object": "chat.completion.chunk",
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": "Streamed RAG answer"}}],
                },
                {
                    "id": "chatcmpl-stream-1",
                    "choices": [],
                    "usage": {"prompt_tokens": 9, "completion_tokens": 3},
                },
                DONE,
            ],
            fail_after=self.stream_fail_after,
        )
        self.streams.append(stream)
        return stream


def make_settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "rag_bearer_token": TEST_TOKEN,
        "root_path": "/rag",
        "embedding_dim": DIM,
        "rag_service_tokens": TOKENS,
        "rag_access_policy": POLICY,
        "metrics_port": None,
    }
    base.update(overrides)
    return Settings(**base)


@pytest.fixture()
def qdrant() -> AsyncQdrantClient:
    return AsyncQdrantClient(location=":memory:")


@pytest.fixture()
def settings() -> Settings:
    return make_settings()


@pytest.fixture()
def fakes(settings: Settings, qdrant: AsyncQdrantClient) -> dict[str, Any]:
    return {
        "docling": FakeDoclingClient(),
        "embeddings": FakeEmbeddingClient(),
        "qdrant": QdrantStore(settings, client=qdrant),
        "litellm": FakeLiteLLMClient(),
    }


def build_client(settings: Settings, fakes: dict[str, Any], **extra: Any) -> TestClient:
    app = create_app(
        settings,
        docling_client=fakes["docling"],
        embedding_client=fakes["embeddings"],
        qdrant_store=fakes["qdrant"],
        litellm_client=fakes["litellm"],
        **extra,
    )
    return TestClient(app)


@pytest.fixture()
def client(settings: Settings, fakes: dict[str, Any]) -> Iterator[TestClient]:
    with build_client(settings, fakes) as test_client:
        yield test_client


def ingest(
    client: TestClient, collection: str, doc_id: str, text: str, **extra: Any
) -> dict[str, Any]:
    resp = client.post(
        "/ingest",
        headers=auth(),
        json={"collection": collection, "doc_id": doc_id, "text": text, **extra},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def sse_events(body: str) -> list[Any]:
    out: list[Any] = []
    for line in body.splitlines():
        if line.startswith("data: "):
            raw = line[len("data: ") :]
            out.append(raw if raw == "[DONE]" else json.loads(raw))
    return out
