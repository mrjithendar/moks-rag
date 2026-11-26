"""QdrantStore and Indexer on in-memory Qdrant: versions, replacement races, migration, rebuild."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from qdrant_client import AsyncQdrantClient, models

from rag_api import store as store_module
from rag_api.indexer import Indexer, IngestDoc
from rag_api.store import QdrantStore
from tests.conftest import FakeEmbeddingClient, bag_vector, make_settings


def doc(doc_id: str, text: str, collection: str = "sre_knowledge", **kw: Any) -> IngestDoc:
    return IngestDoc(
        collection=collection,
        doc_id=doc_id,
        text=text,
        title=kw.get("title"),
        source=None,
        metadata=kw.get("metadata", {}),
        chunk_size=kw.get("chunk_size", 300),
        chunk_overlap=kw.get("chunk_overlap", 50),
        ingested_by="test",
    )


async def doc_points(client: AsyncQdrantClient, physical: str, doc_id: str) -> list[Any]:
    points, _ = await client.scroll(
        physical,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
        ),
        limit=100,
        with_payload=True,
    )
    return points


@pytest.fixture()
async def store(qdrant: AsyncQdrantClient) -> QdrantStore:
    s = QdrantStore(make_settings(), client=qdrant)
    await s.ensure_collections()
    return s


async def test_collections_are_aliases_to_versioned_physical(
    store: QdrantStore, qdrant: AsyncQdrantClient
) -> None:
    aliases = {a.alias_name: a.collection_name for a in (await qdrant.get_aliases()).aliases}
    assert aliases == {
        "sre_knowledge": "sre_knowledge__v2",
        "cms_content": "cms_content__v2",
        "seo_intel": "seo_intel__v2",
    }


async def test_empty_legacy_collection_is_replaced_by_alias(qdrant: AsyncQdrantClient) -> None:
    await qdrant.create_collection(
        "cms_content",
        vectors_config={"dense": models.VectorParams(size=32, distance=models.Distance.COSINE)},
    )
    s = QdrantStore(make_settings(), client=qdrant)
    await s.ensure_collections()
    state = await s.state("cms_content")
    assert state.mode == "alias" and state.serving == "cms_content__v2"


async def test_non_empty_legacy_collection_is_kept_until_promoted(
    qdrant: AsyncQdrantClient,
) -> None:
    await qdrant.create_collection(
        "cms_content",
        vectors_config={"dense": models.VectorParams(size=32, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    await qdrant.upsert(
        "cms_content",
        points=[
            models.PointStruct(
                id=1,
                vector={"dense": [1.0] * 32},
                payload={"doc_id": "old", "chunk_index": 0, "text": "0.1.0 chunk"},
            )
        ],
    )
    s = QdrantStore(make_settings(), client=qdrant)
    await s.ensure_collections()
    state = await s.state("cms_content")
    assert state.mode == "legacy" and state.serving == "cms_content"
    assert state.write_targets == ["cms_content", "cms_content__v2"]
    assert await s.count_without_source("cms_content") == 1

    with pytest.raises(ValueError):
        await s.promote("cms_content")
    promoted = await s.promote("cms_content", drop_legacy=True)
    assert promoted.serving == "cms_content__v2"
    assert "cms_content" not in {c.name for c in (await qdrant.get_collections()).collections}


async def test_reingest_into_legacy_collection_replaces_its_old_chunks(
    qdrant: AsyncQdrantClient,
) -> None:
    await qdrant.create_collection(
        "sre_knowledge",
        vectors_config={"dense": models.VectorParams(size=32, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    await qdrant.upsert(
        "sre_knowledge",
        points=[
            models.PointStruct(
                id=i,
                vector={"dense": [1.0] * 32},
                payload={"doc_id": "rb", "chunk_index": i, "text": f"0.1.0 chunk {i}"},
            )
            for i in range(3)
        ],
    )
    s = QdrantStore(make_settings(), client=qdrant)
    await s.ensure_collections()
    await Indexer(make_settings(), s, FakeEmbeddingClient()).ingest(
        doc("rb", "the runbook, re-sent")
    )
    legacy = await doc_points(qdrant, "sre_knowledge", "rb")
    assert [p.payload["text"] for p in legacy] == ["the runbook, re-sent"]
    assert len(await doc_points(qdrant, "sre_knowledge__v2", "rb")) == 1
    assert await s.count_without_source("sre_knowledge") == 0


async def test_replace_never_leaves_old_chunks(
    store: QdrantStore, qdrant: AsyncQdrantClient
) -> None:
    indexer = Indexer(make_settings(), store, FakeEmbeddingClient())
    long_text = "\n\n".join(f"Paragraph {i}: " + "istio sidecar mtls " * 12 for i in range(8))
    first = await indexer.ingest(doc("d1", long_text))
    assert first.chunks > 2
    second = await indexer.ingest(doc("d1", "A much shorter replacement document."))
    assert second.chunks == 1
    points = await doc_points(qdrant, "sre_knowledge__v2", "d1")
    assert len(points) == 1 and points[0].payload["chunk_count"] == 1
    assert points[0].payload["document"]["text"] == "A much shorter replacement document."


async def test_concurrent_writes_converge_on_newest(
    store: QdrantStore, qdrant: AsyncQdrantClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    indexer = Indexer(make_settings(), store, FakeEmbeddingClient())
    # B (newer timestamp) finishes first, then A (older timestamp, a slow writer) upserts.
    monkeypatch.setattr(store_module, "_now_us", lambda: 2_000)
    await indexer.ingest(doc("race", "version B text"))
    monkeypatch.setattr(store_module, "_now_us", lambda: 1_000)
    await indexer.ingest(doc("race", "version A text, written late, with more words in it"))
    points = await doc_points(qdrant, "sre_knowledge__v2", "race")
    assert {p.payload["text"] for p in points} == {"version B text"}


async def test_failed_upsert_keeps_previous_version(
    store: QdrantStore, qdrant: AsyncQdrantClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    indexer = Indexer(make_settings(), store, FakeEmbeddingClient())
    await indexer.ingest(doc("keep", "the previous version"))

    async def boom(*_: Any, **__: Any) -> None:
        raise RuntimeError("qdrant timed out")

    monkeypatch.setattr(qdrant, "upsert", boom)
    with pytest.raises(RuntimeError):
        await indexer.ingest(doc("keep", "a new version that never lands"))
    monkeypatch.undo()
    points = await doc_points(qdrant, "sre_knowledge__v2", "keep")
    assert [p.payload["text"] for p in points] == ["the previous version"]


async def test_hybrid_search_reports_dense_scores(store: QdrantStore) -> None:
    indexer = Indexer(make_settings(), store, FakeEmbeddingClient())
    await indexer.ingest(
        doc("k", "KEDA scales deployments from Prometheus metrics", collection="cms_content")
    )
    await indexer.ingest(
        doc("v", "Valkey replaces Redis with Sentinel failover", collection="cms_content")
    )
    query = "KEDA Prometheus scaling"
    from rag_api.sparse import encode_sparse_bm25

    hits = await store.hybrid_search(
        "cms_content", bag_vector(query), encode_sparse_bm25(query), top_k=5
    )
    assert hits[0].doc_id == "k"
    assert all(h.dense_score is not None for h in hits)
    assert hits[0].dense_score > hits[-1].dense_score
    floor = (hits[0].dense_score + hits[-1].dense_score) / 2
    kept = await store.hybrid_search(
        "cms_content", bag_vector(query), encode_sparse_bm25(query), top_k=5, min_dense_score=floor
    )
    assert [h.doc_id for h in kept] == ["k"]


async def test_version_bump_dual_writes_then_rebuild_and_promote(qdrant: AsyncQdrantClient) -> None:
    v2 = QdrantStore(make_settings(), client=qdrant)
    await v2.ensure_collections()
    await Indexer(make_settings(), v2, FakeEmbeddingClient()).ingest(
        doc("old-doc", "stored before the version bump", title="Old")
    )

    settings_v3 = make_settings(index_version="v3")
    v3 = QdrantStore(settings_v3, client=qdrant)
    await v3.ensure_collections()
    state = await v3.state("sre_knowledge")
    assert (state.serving, state.target) == ("sre_knowledge__v2", "sre_knowledge__v3")

    indexer = Indexer(settings_v3, v3, FakeEmbeddingClient())
    await indexer.ingest(doc("new-doc", "written after the bump"))
    assert await v3.count("sre_knowledge__v2") == 2  # still served and written
    assert await v3.count("sre_knowledge__v3") == 1

    status = indexer.start_rebuild("sre_knowledge", promote=True)
    await indexer._rebuild_tasks["sre_knowledge"]
    assert status["state"] == "done", status
    assert status["documents"] == 1 and status["unchanged"] == 1 and status["promoted"] is True
    assert (await v3.state("sre_knowledge")).serving == "sre_knowledge__v3"
    copied = await doc_points(qdrant, "sre_knowledge__v3", "old-doc")
    assert copied and copied[0].payload["index_version"] == "v3"

    await v3.drop_physical("sre_knowledge__v2")
    with pytest.raises(ValueError):
        await v3.drop_physical("sre_knowledge__v3")


async def test_ingest_concurrency_limit_queues_ingests(store: QdrantStore) -> None:
    indexer = Indexer(make_settings(ingest_max_concurrency=1), store, FakeEmbeddingClient())
    running = 0
    peak = 0

    async def one(i: int) -> None:
        nonlocal running, peak
        async with indexer.slot():
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.01)
            running -= 1

    await asyncio.gather(*(one(i) for i in range(4)))
    assert peak == 1
