"""The write path: chunk → embed → replace, shared by /ingest and index-version rebuilds.

- Unchanged documents are skipped: chunk 0 stores a content hash (text, title, source, metadata,
  chunk settings, index version, embedding model); the same hash means nothing is embedded or
  written ("unchanged": true).
- INGEST_MAX_CONCURRENCY bounds how many ingests embed at once, so a bulk reindex cannot queue every
  query embedding behind it (queries do not take a slot). Empty = no limit.
- The source document is stored with chunk 0, so POST /admin/collections/{name}/rebuild can rebuild
  a new index version without the callers re-sending anything.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from rag_api import metrics
from rag_api.chunking import chunk_markdown, embedding_text
from rag_api.config import Settings
from rag_api.embeddings import EmbeddingClient
from rag_api.logs import emit
from rag_api.store import RESERVED_KEYS, PreparedChunk, QdrantStore

logger = logging.getLogger("rag_api.indexer")


@dataclass
class IngestDoc:
    collection: str
    doc_id: str
    text: str
    title: str | None
    source: str | None
    metadata: dict[str, Any]
    chunk_size: int
    chunk_overlap: int
    ingested_by: str

    def as_stored(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "title": self.title,
            "source": self.source,
            "metadata": self.metadata,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "ingested_by": self.ingested_by,
        }


@dataclass
class IngestOutcome:
    chunks: int
    unchanged: bool


class NoIndexableText(ValueError):
    pass


class Indexer:
    def __init__(self, settings: Settings, store: QdrantStore, embeddings: EmbeddingClient) -> None:
        self._settings = settings
        self._store = store
        self._embeddings = embeddings
        limit = settings.ingest_max_concurrency
        self._slots = asyncio.Semaphore(limit) if limit else None
        self.rebuilds: dict[str, dict[str, Any]] = {}
        self._rebuild_tasks: dict[str, asyncio.Task[None]] = {}

    @contextlib.asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """One ingest's turn (parse + embed + write)."""
        if self._slots is None:
            metrics.INGESTS_IN_PROGRESS.inc()
            try:
                yield
            finally:
                metrics.INGESTS_IN_PROGRESS.dec()
            return
        metrics.INGESTS_WAITING.inc()
        try:
            await self._slots.acquire()
        finally:
            metrics.INGESTS_WAITING.dec()
        metrics.INGESTS_IN_PROGRESS.inc()
        try:
            yield
        finally:
            metrics.INGESTS_IN_PROGRESS.dec()
            self._slots.release()

    def content_hash(self, doc: IngestDoc) -> str:
        material = {
            "text": doc.text,
            "title": doc.title,
            "source": doc.source,
            "metadata": doc.metadata,
            "chunk_size": doc.chunk_size,
            "chunk_overlap": doc.chunk_overlap,
            "index_version": self._settings.index_version,
            "embedding_model": self._embeddings.model,
        }
        encoded = json.dumps(material, sort_keys=True, default=str, ensure_ascii=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    async def ingest(self, doc: IngestDoc, *, targets: list[str] | None = None) -> IngestOutcome:
        """Index one document into every physical collection it is written to (or `targets`)."""
        chunks = chunk_markdown(
            doc.text, chunk_size=doc.chunk_size, chunk_overlap=doc.chunk_overlap
        )
        if not chunks:
            raise NoIndexableText("Document produced no indexable text chunks")

        digest = self.content_hash(doc)
        physicals = targets or await self._store.write_targets(doc.collection)
        stale: list[str] = []
        for physical in physicals:
            current = await self._store.current_version(physical, doc.doc_id)
            if current is None or current.content_hash != digest:
                stale.append(physical)
        if not stale:
            return IngestOutcome(chunks=len(chunks), unchanged=True)

        texts = [embedding_text(c, doc.title) for c in chunks]
        dense, sparse = await self._embeddings.embed_documents(texts)
        prepared = [
            PreparedChunk(text=c.text, heading_path=c.heading_path, dense=d, sparse=s)
            for c, d, s in zip(chunks, dense, sparse, strict=True)
        ]
        payload_base = {
            **{k: v for k, v in doc.metadata.items() if k not in RESERVED_KEYS},
            "content_hash": digest,
            "index_version": self._settings.index_version,
            "embedding_model": self._embeddings.model,
        }
        count = 0
        for physical in stale:
            count = await self._store.replace_document(
                physical, doc.doc_id, prepared, payload_base, doc.as_stored()
            )
        return IngestOutcome(chunks=count, unchanged=False)

    # --- rebuild ---
    # ----------------------------------------------------------------------------------------------

    def rebuild_status(self, logical: str) -> dict[str, Any] | None:
        return self.rebuilds.get(logical)

    def start_rebuild(self, logical: str, *, promote: bool) -> dict[str, Any]:
        task = self._rebuild_tasks.get(logical)
        if task is not None and not task.done():
            return self.rebuilds[logical]
        status: dict[str, Any] = {
            "state": "running",
            "promote": promote,
            "documents": 0,
            "unchanged": 0,
            "failed": 0,
            "started_at": time.time(),
        }
        self.rebuilds[logical] = status
        self._rebuild_tasks[logical] = asyncio.create_task(self._rebuild(logical, promote, status))
        return status

    async def _rebuild(self, logical: str, promote: bool, status: dict[str, Any]) -> None:
        try:
            state = await self._store.state(logical)
            status.update(source=state.serving, target=state.target)
            if state.serving != state.target:
                async for stored in self._store.iter_documents(state.serving):
                    latest = await self._store.current_version(state.serving, stored.doc_id)
                    if latest is not None and latest.ingest_ts != stored.ingest_ts:
                        continue  # an older copy left by a race; its newest version is copied
                    d = stored.document
                    doc = IngestDoc(
                        collection=logical,
                        doc_id=stored.doc_id,
                        text=str(d.get("text") or ""),
                        title=d.get("title"),
                        source=d.get("source"),
                        metadata=dict(d.get("metadata") or {}),
                        chunk_size=int(d.get("chunk_size") or self._settings.chunk_size),
                        chunk_overlap=int(d.get("chunk_overlap") or self._settings.chunk_overlap),
                        ingested_by=str(d.get("ingested_by") or "rebuild"),
                    )
                    try:
                        async with self.slot():
                            outcome = await self.ingest(doc, targets=[state.target])
                        status["unchanged" if outcome.unchanged else "documents"] += 1
                    except Exception as exc:  # noqa: BLE001 - one bad document must not stop the rebuild
                        status["failed"] += 1
                        emit(
                            logger,
                            "rebuild_document_failed",
                            level=logging.WARNING,
                            collection=logical,
                            doc_id=stored.doc_id,
                            error=type(exc).__name__,
                        )
                status["documents_without_source"] = await self._store.count_without_source(
                    state.serving
                )
            if promote:
                if status["failed"] or status.get("documents_without_source"):
                    status["promoted"] = False
                    status["promote_skipped"] = (
                        "some documents could not be copied; check them, then call promote"
                    )
                else:
                    await self._store.promote(logical)
                    status["promoted"] = True
            status["state"] = "done"
        except Exception as exc:  # noqa: BLE001
            status["state"] = "failed"
            status["error"] = f"{type(exc).__name__}: {exc}"[:300]
            emit(logger, "rebuild_failed", level=logging.WARNING, collection=logical)
        finally:
            status["finished_at"] = time.time()
