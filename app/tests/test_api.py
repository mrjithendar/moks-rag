"""Unit and integration tests for rag-api endpoints, auth, chunking, and sparse encoding."""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi.testclient import TestClient
from tests.conftest import TEST_TOKEN

from rag_api.chunking import chunk_text
from rag_api.sparse import encode_sparse_bm25

AUTH_HEADERS = {
    "Authorization": f"Bearer {TEST_TOKEN}",
    "X-Service-Name": "moks-cms",
}


def test_chunk_text_respects_size_and_overlap() -> None:
    paragraphs = [f"Paragraph {i}. " + ("k3s istio rag " * 15) for i in range(6)]
    text = "\n\n".join(paragraphs)
    chunks = chunk_text(text, chunk_size=300, chunk_overlap=50)
    assert len(chunks) >= 3
    assert all(c.strip() for c in chunks)


def test_encode_sparse_bm25_is_deterministic_and_sorted() -> None:
    vec_a = encode_sparse_bm25("Qdrant hybrid search with BGE-M3 dense and BM25 sparse vectors")
    vec_b = encode_sparse_bm25("Qdrant hybrid search with BGE-M3 dense and BM25 sparse vectors")
    assert vec_a == vec_b
    assert vec_a.indices == sorted(vec_a.indices)
    assert len(vec_a.indices) == len(vec_a.values) > 0


def test_health_and_ready_do_not_require_auth(client: TestClient) -> None:
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    ready = client.get("/ready")
    assert ready.status_code == 200
    assert set(ready.json()["collections"]) == {"sre_knowledge", "cms_content", "seo_intel"}


def test_auth_rejects_missing_or_wrong_token(client: TestClient) -> None:
    assert client.get("/v1/models").status_code == 401
    assert (
        client.get("/v1/models", headers={"Authorization": "Bearer wrong-token"}).status_code == 401
    )


def test_ingest_json_and_service_logging(
    client: TestClient, fakes: dict[str, Any], caplog: Any
) -> None:
    caplog.set_level(logging.INFO, logger="rag_api.access")
    resp = client.post(
        "/ingest",
        headers=AUTH_HEADERS,
        json={
            "collection": "cms_content",
            "doc_id": "article-42",
            "title": "Hetzner k3s Guide",
            "text": "Hetzner CAX ARM64 nodes run k3s with Istio STRICT mTLS and ArgoCD GitOps.",
            "metadata": {"slug": "hetzner-k3s-guide"},
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["doc_id"] == "article-42"
    assert body["collection"] == "cms_content"
    assert body["chunks_ingested"] == 1
    assert body["parsed_via_docling"] is False
    assert body["service"] == "moks-cms"
    assert ("cms_content", "article-42") in fakes["qdrant"].docs
    assert any('"service": "moks-cms"' in rec.message for rec in caplog.records)


def test_ingest_multipart_pdf_uses_docling(client: TestClient, fakes: dict[str, Any]) -> None:
    resp = client.post(
        "/ingest",
        headers={"Authorization": f"Bearer {TEST_TOKEN}", "X-Service-Name": "moks-sre"},
        data={
            "collection": "sre_knowledge",
            "doc_id": "incident-99",
            "metadata": json.dumps({"severity": "sev1"}),
        },
        files={"file": ("postmortem.pdf", b"%PDF-1.4 fake bytes", "application/pdf")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["parsed_via_docling"] is True
    assert body["chunks_ingested"] >= 1
    assert len(fakes["docling"].calls) == 1
    assert fakes["docling"].calls[0]["filename"] == "postmortem.pdf"


def test_delete_document_for_cms_webhook_sync(client: TestClient, fakes: dict[str, Any]) -> None:
    resp = client.delete(
        "/documents/article-42",
        params={"collection": "cms_content"},
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["deleted"] is True
    assert body["collections"] == ["cms_content"]
    assert ("cms_content", "article-42") in fakes["qdrant"].deleted


def test_query_runs_hybrid_search_and_litellm(client: TestClient, fakes: dict[str, Any]) -> None:
    resp = client.post(
        "/query",
        headers={"Authorization": f"Bearer {TEST_TOKEN}", "X-Service-Name": "moks-sre"},
        json={
            "collection": "sre_knowledge",
            "query": "Why am I seeing Istio 503 errors in rag?",
            "top_k": 4,
            "model": "claude-haiku-5-5",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "claude-haiku-5-5"
    assert "runbook-istio-503" in body["answer"]
    assert len(body["sources"]) == 1
    assert fakes["qdrant"].searches[-1]["collection"] == "sre_knowledge"
    assert fakes["qdrant"].searches[-1]["dense_dim"] == 1024
    assert fakes["litellm"].calls[-1]["model"] == "claude-haiku-5-5"


def test_openai_models_and_chat_completions(client: TestClient, fakes: dict[str, Any]) -> None:
    models_resp = client.get("/v1/models", headers=AUTH_HEADERS)
    assert models_resp.status_code == 200
    ids = {m["id"] for m in models_resp.json()["data"]}
    assert ids == {"moks-sre", "moks-cms", "moks-seo-agent"}

    chat_resp = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {TEST_TOKEN}", "X-Service-Name": "open-webui"},
        json={
            "model": "moks-seo-agent",
            "messages": [{"role": "user", "content": "Summarize our latest SEO audit findings"}],
            "stream": False,
        },
    )
    assert chat_resp.status_code == 200
    chat_body = chat_resp.json()
    assert chat_body["model"] == "moks-seo-agent"
    assert fakes["qdrant"].searches[-1]["collection"] == "seo_intel"
    assert chat_body["choices"][0]["message"]["content"]

    stream_resp = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {TEST_TOKEN}", "X-Service-Name": "open-webui"},
        json={
            "model": "moks-sre",
            "messages": [{"role": "user", "content": "Check mTLS runbook"}],
            "stream": True,
        },
    )
    assert stream_resp.status_code == 200
    assert "text/event-stream" in stream_resp.headers["content-type"]
    assert '"model":"moks-sre"' in stream_resp.text
    assert "data: [DONE]" in stream_resp.text
