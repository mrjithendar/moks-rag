"""Endpoint tests: ingest, query, delete, readiness, the OpenAI-compatible chat, errors."""

from __future__ import annotations

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi.testclient import TestClient

from rag_api.errors import UpstreamError
from rag_api.sparse import SparseVectorData
from tests.conftest import (
    FakeEmbeddingClient,
    auth,
    build_client,
    ingest,
    make_settings,
    sse_events,
    svc,
)

RUNBOOK = (
    "# Istio 503 Runbook\n\n## Symptoms\n\nClients see 503 UF upstream connect errors.\n\n"
    "## Fix\n\nCheck PeerAuthentication STRICT mTLS and sidecar injection on the rag namespace."
)


def test_health_and_ready_do_not_require_auth(client: TestClient) -> None:
    assert client.get("/health").json()["status"] == "ok"
    ready = client.get("/ready")
    assert ready.status_code == 200
    assert set(ready.json()["collections"]) == {"sre_knowledge", "cms_content", "seo_intel"}


def test_ready_hides_dependency_details(client: TestClient, fakes: dict[str, Any]) -> None:
    async def broken() -> bool:
        raise RuntimeError("http://qdrant.rag.svc.cluster.local:6333 refused")

    fakes["qdrant"].healthcheck = broken
    resp = client.get("/ready")
    assert resp.status_code == 503
    assert resp.json() == {"detail": "Qdrant unavailable"}


def test_root_redirects_to_docs(client: TestClient) -> None:
    root = client.get("/", follow_redirects=False)
    assert root.status_code == 307
    assert root.headers["location"] == "/rag/docs"


def test_auth_rejects_missing_or_wrong_token(client: TestClient) -> None:
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers=auth("wrong-token")).status_code == 401


def test_ingest_json_and_service_logging(
    client: TestClient, fakes: dict[str, Any], caplog: Any
) -> None:
    caplog.set_level(logging.INFO, logger="rag_api.access")
    body = ingest(
        client,
        "cms_content",
        "article-42",
        "Hetzner CAX ARM64 nodes run k3s with Istio STRICT mTLS and ArgoCD GitOps.",
        title="Hetzner k3s Guide",
        metadata={"slug": "hetzner-k3s-guide"},
    )
    assert body["doc_id"] == "article-42"
    assert body["chunks_ingested"] == 1
    assert body["parsed_via_docling"] is False
    assert body["service"] == "moks-cms"
    assert body["unchanged"] is False
    assert body["index_version"] == "v2"
    records = [json.loads(r.message) for r in caplog.records if r.name == "rag_api.access"]
    http = [r for r in records if r["event"] == "http_request" and r["path"] == "/ingest"]
    assert http and http[0]["service"] == "moks-cms" and http[0]["status_code"] == 200
    assert http[0]["auth"] == "shared-token"


def test_ingest_requires_doc_id(client: TestClient) -> None:
    resp = client.post(
        "/ingest", headers=auth(), json={"collection": "cms_content", "text": "no id given"}
    )
    assert resp.status_code == 422


def test_reingesting_same_document_is_unchanged(client: TestClient, fakes: dict[str, Any]) -> None:
    first = ingest(client, "sre_knowledge", "rb-1", RUNBOOK, title="Istio 503")
    calls = len(fakes["embeddings"].document_calls)
    second = ingest(client, "sre_knowledge", "rb-1", RUNBOOK, title="Istio 503")
    assert first["unchanged"] is False and second["unchanged"] is True
    assert len(fakes["embeddings"].document_calls) == calls  # nothing embedded again
    third = ingest(
        client, "sre_knowledge", "rb-1", RUNBOOK + "\n\nAlso restart the pod.", title="Istio 503"
    )
    assert third["unchanged"] is False


class SlowEmbeddings(FakeEmbeddingClient):
    """Embeds like the fake, but takes a while, and counts how many embeddings ran at once."""

    def __init__(self, delay: float) -> None:
        super().__init__()
        self.delay = delay
        self.running = 0
        self.most_at_once = 0

    async def embed_documents(
        self, texts: list[str]
    ) -> tuple[list[list[float]], list[SparseVectorData]]:
        self.running += 1
        self.most_at_once = max(self.most_at_once, self.running)
        try:
            await asyncio.sleep(self.delay)
            return await super().embed_documents(texts)
        finally:
            self.running -= 1


def test_resends_of_one_document_take_turns_and_embed_once(
    settings: Any, fakes: dict[str, Any], caplog: Any
) -> None:
    # moks-cms gave up after 10 s and sent the same post again, twice: three ingests of one document
    # at once used to embed it three times side by side, and bge-m3 timed them all out (504).
    caplog.set_level(logging.INFO, logger="rag_api.access")
    slow = SlowEmbeddings(delay=0.3)
    fakes["embeddings"] = slow
    with build_client(settings, fakes) as client, ThreadPoolExecutor(3) as pool:
        bodies = list(
            pool.map(lambda _: ingest(client, "cms_content", "blog:same", RUNBOOK), range(3))
        )
    assert len(slow.document_calls) == 1 and slow.most_at_once == 1
    assert sorted(b["unchanged"] for b in bodies) == [False, True, True]
    records = [json.loads(r.message) for r in caplog.records if r.name == "rag_api.access"]
    waits = sorted(r["waited_for_same_document_s"] for r in records if r["event"] == "rag_ingest")
    assert waits[0] < 0.1 and waits[1] >= 0.2 and waits[2] >= 0.2


def test_different_documents_still_ingest_side_by_side(
    settings: Any, fakes: dict[str, Any]
) -> None:
    slow = SlowEmbeddings(delay=0.3)
    fakes["embeddings"] = slow
    with build_client(settings, fakes) as client, ThreadPoolExecutor(3) as pool:
        posts = [(f"blog:post-{i}", f"{RUNBOOK}\n\nPost {i}.") for i in range(3)]
        list(pool.map(lambda post: ingest(client, "cms_content", *post), posts))
    assert len(slow.document_calls) == 3 and slow.most_at_once == 3


def test_ingest_multipart_pdf_uses_docling(client: TestClient, fakes: dict[str, Any]) -> None:
    resp = client.post(
        "/ingest",
        headers=auth(service="moks-sre"),
        data={
            "collection": "sre_knowledge",
            "doc_id": "incident-99",
            "metadata": json.dumps({"severity": "sev1"}),
        },
        files={"file": ("postmortem.pdf", b"%PDF-1.4 fake bytes", "application/pdf")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["parsed_via_docling"] is True
    assert body["chunks_ingested"] >= 1
    assert fakes["docling"].calls[0]["filename"] == "postmortem.pdf"


def test_multipart_rejects_bad_chunk_size(client: TestClient) -> None:
    resp = client.post(
        "/ingest",
        headers=auth(),
        data={
            "collection": "cms_content",
            "doc_id": "x",
            "text": "hello world",
            "chunk_size": "abc",
        },
    )
    assert resp.status_code == 422


def test_ingest_too_long_chunk_is_refused(settings: Any, fakes: dict[str, Any]) -> None:
    fakes["embeddings"].max_chars = 50
    with build_client(settings, fakes) as c:
        resp = c.post(
            "/ingest",
            headers=auth(),
            json={"collection": "cms_content", "doc_id": "long", "text": "word " * 100},
        )
    assert resp.status_code == 422
    assert "context" in resp.json()["detail"]


def test_upload_limit(fakes: dict[str, Any]) -> None:
    with build_client(make_settings(max_upload_bytes=200), fakes) as c:
        big = c.post(
            "/ingest",
            headers=auth(),
            json={"collection": "cms_content", "doc_id": "big", "text": "x " * 500},
        )
        small = c.post(
            "/ingest",
            headers=auth(),
            json={"collection": "cms_content", "doc_id": "small", "text": "tiny"},
        )
    assert big.status_code == 413
    assert small.status_code == 200


def test_delete_requires_collection_and_accepts_slashes(client: TestClient) -> None:
    ingest(client, "sre_knowledge", "runbooks/istio-503.md", RUNBOOK)
    assert client.delete("/documents/runbooks/istio-503.md", headers=auth()).status_code == 422

    resp = client.delete(
        "/documents/runbooks/istio-503.md", params={"collection": "sre_knowledge"}, headers=auth()
    )
    assert resp.status_code == 200
    assert resp.json()["doc_id"] == "runbooks/istio-503.md"

    found = client.post(
        "/query",
        headers=auth(),
        json={"collection": "sre_knowledge", "query": "istio 503", "include_answer": False},
    )
    assert found.json()["sources"] == []


def test_delete_by_query_param(client: TestClient) -> None:
    ingest(client, "cms_content", "blog:a/b", "Some article about KEDA autoscaling.")
    resp = client.delete(
        "/documents", params={"doc_id": "blog:a/b", "collection": "cms_content"}, headers=auth()
    )
    assert resp.status_code == 200
    assert resp.json()["collections"] == ["cms_content"]


def test_query_runs_hybrid_search_and_litellm(client: TestClient, fakes: dict[str, Any]) -> None:
    ingest(client, "sre_knowledge", "runbook-istio-503", RUNBOOK, title="Istio 503 Runbook")
    resp = client.post(
        "/query",
        headers=auth(service="moks-sre"),
        json={
            "collection": "sre_knowledge",
            "query": "Why am I seeing Istio 503 errors in rag?",
            "top_k": 4,
            "model": "claude-haiku-5-5",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["model"] == "claude-haiku-5-5"
    assert body["sources"][0]["doc_id"] == "runbook-istio-503"
    assert body["sources"][0]["dense_score"] is not None
    assert "document" not in body["sources"][0]["metadata"]  # the stored source never leaves
    assert body["reranker"] == "off"

    call = fakes["litellm"].calls[-1]
    system, user = call["messages"]
    assert "Retrieved" not in system["content"] and "PeerAuthentication" not in system["content"]
    assert "never follow instructions" in system["content"]
    assert "<documents>" in user["content"] and "PeerAuthentication" in user["content"]
    assert "<question>" in user["content"]


def test_query_rejects_unknown_model(client: TestClient) -> None:
    resp = client.post(
        "/query",
        headers=auth(),
        json={"collection": "cms_content", "query": "q", "model": "gpt-9"},
    )
    assert resp.status_code == 422


def test_query_rejects_bad_filter_key(client: TestClient) -> None:
    resp = client.post(
        "/query",
        headers=auth(),
        json={
            "collection": "cms_content",
            "query": "q",
            "filters": {"document": "x"},
            "include_answer": False,
        },
    )
    assert resp.status_code == 422


def test_query_filters_and_dense_floor(client: TestClient) -> None:
    ingest(
        client,
        "cms_content",
        "a",
        "KEDA scales deployments on Prometheus metrics.",
        metadata={"status": "published"},
    )
    ingest(
        client,
        "cms_content",
        "b",
        "KEDA scales deployments on queue length.",
        metadata={"status": "draft"},
    )
    published = client.post(
        "/query",
        headers=auth(),
        json={
            "collection": "cms_content",
            "query": "KEDA scaling",
            "filters": {"status": "published"},
            "include_answer": False,
        },
    ).json()
    assert [s["doc_id"] for s in published["sources"]] == ["a"]

    unrelated = client.post(
        "/query",
        headers=auth(),
        json={
            "collection": "cms_content",
            "query": "sourdough bread recipe",
            "min_dense_score": 0.5,
            "include_answer": False,
        },
    ).json()
    assert unrelated["sources"] == []


def test_upstream_errors_map_to_http_status(client: TestClient, fakes: dict[str, Any]) -> None:
    async def rate_limited(**_: Any) -> dict[str, Any]:
        raise UpstreamError(
            "litellm", "answered HTTP 429", kind="rate_limited", status=429, retry_after="7"
        )

    fakes["litellm"].chat_completions = rate_limited
    resp = client.post("/query", headers=auth(), json={"collection": "cms_content", "query": "q"})
    assert resp.status_code == 429
    assert resp.headers["retry-after"] == "7"
    assert resp.json()["dependency"] == "litellm"

    async def down(text: str) -> Any:
        raise UpstreamError("bge-m3", "unreachable (ConnectError)", kind="unavailable")

    fakes["embeddings"].embed_query = down
    resp = client.post("/query", headers=auth(), json={"collection": "cms_content", "query": "q"})
    assert resp.status_code == 503
    assert resp.json()["dependency"] == "bge-m3"


def test_openai_models_and_chat_completions(client: TestClient, fakes: dict[str, Any]) -> None:
    ids = {m["id"] for m in client.get("/v1/models", headers=auth()).json()["data"]}
    assert ids == {"moks-sre", "moks-cms", "moks-seo-agent"}

    chat_resp = client.post(
        "/v1/chat/completions",
        headers=auth(service="open-webui"),
        json={
            "model": "moks-seo-agent",
            "messages": [{"role": "user", "content": "Summarize our latest SEO audit findings"}],
            "stream": False,
        },
    )
    assert chat_resp.status_code == 200, chat_resp.text
    chat_body = chat_resp.json()
    assert chat_body["model"] == "moks-seo-agent"
    assert chat_body["choices"][0]["message"]["content"]
    assert chat_body["usage"]["total_tokens"] == 54

    stream_resp = client.post(
        "/v1/chat/completions",
        headers=auth(service="open-webui"),
        json={
            "model": "moks-sre",
            "messages": [{"role": "user", "content": "Check mTLS runbook"}],
            "stream": True,
        },
    )
    assert stream_resp.status_code == 200
    assert "text/event-stream" in stream_resp.headers["content-type"]
    events = sse_events(stream_resp.text)
    assert events[0]["model"] == "moks-sre"
    assert events[-1] == "[DONE]"
    assert fakes["litellm"].streams[-1].closed


def test_stream_open_error_is_a_real_http_error(client: TestClient, fakes: dict[str, Any]) -> None:
    fakes["litellm"].open_error = UpstreamError(
        "litellm", "answered HTTP 401", kind="rejected", status=401
    )
    resp = client.post(
        "/v1/chat/completions",
        headers=auth(),
        json={"model": "moks-sre", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert resp.status_code == 502
    assert resp.json()["dependency"] == "litellm"


def test_stream_failure_mid_way_sends_error_event(
    client: TestClient, fakes: dict[str, Any]
) -> None:
    fakes["litellm"].stream_fail_after = 1
    resp = client.post(
        "/v1/chat/completions",
        headers=auth(),
        json={"model": "moks-sre", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    events = sse_events(resp.text)
    assert events[0]["choices"][0]["delta"]["content"] == "Streamed RAG answer"
    assert events[1]["error"]["type"] == "upstream_error"
    assert events[-1] == "[DONE]"


def test_chat_puts_documents_in_user_turn_and_condenses_followups(
    client: TestClient, fakes: dict[str, Any]
) -> None:
    ingest(client, "sre_knowledge", "rb", RUNBOOK, title="Istio 503 Runbook")
    llm = fakes["litellm"]
    llm.answers = ["istio 503 sidecar injection fix", "Final grounded answer"]
    resp = client.post(
        "/v1/chat/completions",
        headers=auth(),
        json={
            "model": "moks-sre",
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "What causes Istio 503 errors?"},
                {"role": "assistant", "content": "Usually mTLS or sidecar problems."},
                {"role": "user", "content": "How do I fix the second one?"},
            ],
        },
    )
    assert resp.status_code == 200, resp.text
    condense_call, answer_call = llm.calls[-2], llm.calls[-1]
    assert condense_call["model"] == "claude-haiku-5-5"
    assert condense_call["max_tokens"] is None
    assert fakes["embeddings"].query_calls[-1] == "istio 503 sidecar injection fix"

    roles = [m["role"] for m in answer_call["messages"]]
    assert roles == ["system", "system", "user", "assistant", "user"]
    assert "PeerAuthentication" not in answer_call["messages"][0]["content"]
    last = answer_call["messages"][-1]["content"]
    assert "<documents>" in last and "How do I fix the second one?" in last


def test_open_webui_task_requests_skip_retrieval(client: TestClient, fakes: dict[str, Any]) -> None:
    queries = len(fakes["embeddings"].query_calls)
    resp = client.post(
        "/v1/chat/completions",
        headers=auth(service="open-webui"),
        json={
            "model": "moks-cms",
            "messages": [
                {"role": "user", "content": "### Task:\nGenerate a concise, 3-5 word title..."}
            ],
        },
    )
    assert resp.status_code == 200
    assert len(fakes["embeddings"].query_calls) == queries
    assert fakes["litellm"].calls[-1]["model"] == "claude-haiku-5-5"


def test_history_is_capped_when_configured(fakes: dict[str, Any]) -> None:
    messages = []
    for i in range(6):
        messages += [
            {"role": "user", "content": f"q{i}"},
            {"role": "assistant", "content": f"a{i}"},
        ]
    messages.append({"role": "user", "content": "last"})
    with build_client(make_settings(max_history_messages=2, condense_followups=False), fakes) as c:
        c.post(
            "/v1/chat/completions", headers=auth(), json={"model": "moks-cms", "messages": messages}
        )
    forwarded = fakes["litellm"].calls[-1]["messages"]
    assert [m["content"] for m in forwarded[1:-1]] == ["q5", "a5"]


def test_scoped_tokens_through_endpoints(client: TestClient) -> None:
    ok = client.post(
        "/ingest",
        headers=svc("moks-cms"),
        json={"collection": "cms_content", "doc_id": "p1", "text": "post"},
    )
    assert ok.status_code == 200 and ok.json()["service"] == "moks-cms"
    denied = client.post(
        "/ingest",
        headers=svc("moks-cms"),
        json={"collection": "sre_knowledge", "doc_id": "p1", "text": "post"},
    )
    assert denied.status_code == 403
    other = client.delete(
        "/documents/p1", params={"collection": "sre_knowledge"}, headers=svc("moks-cms")
    )
    assert other.status_code == 403
    read_only = client.post(
        "/ingest",
        headers=svc("open-webui"),
        json={"collection": "cms_content", "doc_id": "p2", "text": "post"},
    )
    assert read_only.status_code == 403
    assert {m["id"] for m in client.get("/v1/models", headers=svc("moks-cms")).json()["data"]} == {
        "moks-cms"
    }
    assert client.get("/admin/collections", headers=svc("moks-cms")).status_code == 403
    assert client.get("/admin/collections", headers=svc("rag-admin")).status_code == 200


# --- review follow-ups ---


def test_delete_all_means_the_collections_the_caller_may_write(client: TestClient) -> None:
    resp = client.delete("/documents/p1", params={"collection": "all"}, headers=svc("moks-cms"))
    assert resp.status_code == 200 and resp.json()["collections"] == ["cms_content"]
    assert (
        client.delete(
            "/documents/p1", params={"collection": "all"}, headers=svc("open-webui")
        ).status_code
        == 403
    )


def test_large_payload_fields_cannot_be_filtered(client: TestClient) -> None:
    for key in ("text", "document", "document.text"):
        resp = client.post(
            "/query",
            headers=auth(),
            json={
                "collection": "cms_content",
                "query": "q",
                "filters": {key: "x"},
                "include_answer": False,
            },
        )
        assert resp.status_code == 422, key


def test_a_broken_embedding_answer_is_a_502_not_a_client_error(
    client: TestClient, fakes: dict[str, Any]
) -> None:
    async def wrong_dim(texts: list[str]) -> Any:
        raise UpstreamError("bge-m3", "returned a 768-dim embedding, expected 32")

    fakes["embeddings"].embed_documents = wrong_dim
    resp = client.post(
        "/ingest",
        headers=auth(),
        json={"collection": "cms_content", "doc_id": "x", "text": "hello"},
    )
    assert resp.status_code == 502 and resp.json()["dependency"] == "bge-m3"
    bad = client.post(
        "/ingest",
        headers=auth(),
        json={
            "collection": "cms_content",
            "doc_id": "x",
            "text": "hello",
            "chunk_size": 100,
            "chunk_overlap": 100,
        },
    )
    assert bad.status_code == 422
