"""Qdrant storage: versioned collections behind aliases, gap-free document replacement, hybrid
search.

Collections. Each logical collection (sre_knowledge, cms_content, seo_intel) is an alias for a
physical collection `<name>__<index version>`. On start-up rag-api makes sure the physical
collection of its INDEX_VERSION exists:

- alias already on it: served from it ("alias").
- alias on an older version: still served from the older one; every write also goes to the new one,
  and POST /admin/collections/{name}/rebuild fills the new one from the documents stored in the old
  one, then .../promote moves the alias ("alias", target differs).
- a real collection with the logical name (0.1.0): empty, it is replaced by the alias at once; with
  points, it is kept and served ("legacy") until .../promote?drop_legacy=true, because 0.1.0 points
  do not carry their source document and only the callers can re-send them.

Documents. A write never deletes first. The new chunks are upserted under new point ids (ingest_ts
in the id), then every older version of the doc_id is deleted, then, if a newer version appeared
meanwhile, this one deletes itself. Searches therefore never see the document missing, a failed
write leaves the previous version in place, and concurrent writes of one doc_id converge on the
newest instead of mixing their chunks.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from rag_api import metrics
from rag_api.config import COLLECTIONS, Settings
from rag_api.errors import UpstreamError
from rag_api.models import RetrievedChunk
from rag_api.sparse import SparseVectorData

if TYPE_CHECKING:
    from qdrant_client import AsyncQdrantClient

from rag_api.logs import emit

logger = logging.getLogger("rag_api.store")

DEPENDENCY = "qdrant"
SEP = "__"
# Payload keys rag-api manages; caller metadata cannot override them and search results never expose
# "document".
RESERVED_KEYS = frozenset(
    {
        "doc_id",
        "chunk_index",
        "chunk_count",
        "text",
        "heading_path",
        "ingest_ts",
        "content_hash",
        "index_version",
        "embedding_model",
        "document",
    }
)
HIDDEN_KEYS = ("document",)
_SCROLL_PAGE = 64


@dataclass
class CollectionState:
    logical: str
    serving: str  # physical collection reads go to
    target: str  # physical collection of this INDEX_VERSION
    mode: Literal["alias", "legacy"]

    @property
    def write_targets(self) -> list[str]:
        return [self.serving] if self.serving == self.target else [self.serving, self.target]


@dataclass
class PreparedChunk:
    text: str
    heading_path: str
    dense: list[float]
    sparse: SparseVectorData


@dataclass
class StoredVersion:
    ingest_ts: int
    content_hash: str | None
    chunk_count: int


@dataclass
class StoredDocument:
    doc_id: str
    ingest_ts: int
    document: dict[str, Any]


def _now_us() -> int:
    # Microseconds: exact in Qdrant's float range filters (< 2**53), ordered across replicas by wall
    # clock.
    return time.time_ns() // 1000


def new_version_ts() -> int:
    """The timestamp of a new document version (calls _now_us, so tests can patch that)."""
    return _now_us()


class QdrantStore:
    """Manages isolated Qdrant collections with named dense + sparse vectors and RRF fusion."""

    def __init__(self, settings: Settings, client: AsyncQdrantClient | None = None) -> None:
        from qdrant_client import AsyncQdrantClient

        self._dim = settings.embedding_dim
        self._prefetch_limit = settings.prefetch_limit
        self._version = settings.index_version
        self._states: dict[str, CollectionState] = {}
        self._indexed: dict[str, set[str]] = {}
        self._lock = asyncio.Lock()
        self._client = client or AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            timeout=int(settings.http_timeout_seconds),
        )

    # --- lifecycle ---
    # --------------------------------------------------------------------------------------------

    async def close(self) -> None:
        await self._client.close()

    @property
    def ready(self) -> bool:
        return len(self._states) == len(COLLECTIONS)

    async def healthcheck(self) -> bool:
        await self._client.get_collections()
        if not self.ready:
            await self.ensure_collections()
        return True

    def physical_name(self, logical: str, version: str | None = None) -> str:
        return f"{logical}{SEP}{version or self._version}"

    async def _create_physical(self, name: str) -> None:
        from qdrant_client import models

        await self._client.create_collection(
            collection_name=name,
            vectors_config={
                "dense": models.VectorParams(size=self._dim, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )
        emit(logger, "qdrant_collection_created", collection=name)

    async def _ensure_base_indexes(self, physical: str) -> None:
        from qdrant_client import models

        wanted = {
            "doc_id": models.PayloadSchemaType.KEYWORD,
            "chunk_index": models.PayloadSchemaType.INTEGER,
            "ingest_ts": models.PayloadSchemaType.INTEGER,
        }
        done = self._indexed.setdefault(physical, set())
        for key, schema in wanted.items():
            if key not in done:
                await self._client.create_payload_index(
                    collection_name=physical, field_name=key, field_schema=schema
                )
                done.add(key)

    async def _set_alias(self, logical: str, physical: str) -> None:
        from qdrant_client import models

        await self._client.update_collection_aliases(
            change_aliases_operations=[
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(collection_name=physical, alias_name=logical)
                )
            ]
        )

    async def ensure_collections(self) -> None:
        """Create this version's physical collections and resolve what each logical name is served
        from.
        """
        async with self._lock:
            existing = {c.name for c in (await self._client.get_collections()).collections}
            aliases = {
                a.alias_name: a.collection_name for a in (await self._client.get_aliases()).aliases
            }
            for logical in COLLECTIONS:
                target = self.physical_name(logical)
                if target not in existing:
                    await self._create_physical(target)
                    existing.add(target)
                await self._ensure_base_indexes(target)

                if logical in aliases:
                    serving = aliases[logical]
                    state = CollectionState(logical, serving, target, "alias")
                    if serving != target:
                        emit(
                            logger,
                            "index_version_pending",
                            level=logging.WARNING,
                            collection=logical,
                            serving=serving,
                            target=target,
                        )
                elif logical in existing:
                    count = (await self._client.count(logical, exact=True)).count
                    if count == 0:
                        await self._client.delete_collection(logical)
                        await self._set_alias(logical, target)
                        state = CollectionState(logical, target, target, "alias")
                        emit(
                            logger,
                            "legacy_collection_replaced",
                            collection=logical,
                            alias_to=target,
                        )
                    else:
                        state = CollectionState(logical, logical, target, "legacy")
                        await self._ensure_base_indexes(logical)
                        emit(
                            logger,
                            "legacy_collection_in_use",
                            level=logging.WARNING,
                            collection=logical,
                            points=count,
                        )
                else:
                    await self._set_alias(logical, target)
                    state = CollectionState(logical, target, target, "alias")
                if state.serving != target:
                    await self._ensure_base_indexes(state.serving)
                self._states[logical] = state
                metrics.COLLECTION_INFO.labels(
                    collection=logical, physical=state.serving, mode=state.mode
                ).set(1)

    async def _state(self, logical: str) -> CollectionState:
        if logical not in self._states:
            try:
                await self.ensure_collections()
            except UpstreamError:
                raise
            except Exception as exc:  # noqa: BLE001 - qdrant-client raises several transport types
                raise UpstreamError(
                    DEPENDENCY, f"unreachable ({type(exc).__name__})", kind="unavailable"
                ) from exc
        return self._states[logical]

    async def state(self, logical: str) -> CollectionState:
        return await self._state(logical)

    async def write_targets(self, logical: str) -> list[str]:
        return (await self._state(logical)).write_targets

    # --- filters ---
    # ----------------------------------------------------------------------------------------------

    async def _ensure_filter_indexes(self, physical: str, filters: dict[str, Any]) -> None:
        """Index a payload key the first time a query filters on it (keyword, integer, float or
        bool).
        """
        from qdrant_client import models

        done = self._indexed.setdefault(physical, set())
        for key, value in filters.items():
            if key in done:
                continue
            if isinstance(value, bool):
                schema = models.PayloadSchemaType.BOOL
            elif isinstance(value, int):
                schema = models.PayloadSchemaType.INTEGER
            elif isinstance(value, float):
                schema = models.PayloadSchemaType.FLOAT
            else:
                schema = models.PayloadSchemaType.KEYWORD
            try:
                await self._client.create_payload_index(
                    collection_name=physical, field_name=key, field_schema=schema, wait=False
                )
                emit(logger, "payload_index_created", collection=physical, field=key)
            except Exception as exc:  # noqa: BLE001 - an index of another type exists; filtering still works
                emit(
                    logger,
                    "payload_index_failed",
                    level=logging.WARNING,
                    collection=physical,
                    field=key,
                    error=type(exc).__name__,
                )
            done.add(key)

    @staticmethod
    def _filter(filters: dict[str, Any] | None) -> Any:
        from qdrant_client import models

        if not filters:
            return None
        return models.Filter(
            must=[
                models.FieldCondition(key=key, match=models.MatchValue(value=value))
                for key, value in filters.items()
            ]
        )

    @staticmethod
    def _doc_filter(doc_id: str, **ts: int) -> Any:
        from qdrant_client import models

        must: list[Any] = [
            models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))
        ]
        if ts:
            must.append(models.FieldCondition(key="ingest_ts", range=models.Range(**ts)))
        return models.Filter(must=must)

    @staticmethod
    def _older_than(doc_id: str, ts: int) -> Any:
        """Chunks of doc_id written before ts, 0.1.0 chunks (no ingest_ts) included."""
        from qdrant_client import models

        return models.Filter(
            must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))],
            must_not=[models.FieldCondition(key="ingest_ts", range=models.Range(gte=ts))],
        )

    # --- documents ---
    # --------------------------------------------------------------------------------------------

    async def current_version(self, physical: str, doc_id: str) -> StoredVersion | None:
        """The newest stored version of a document (from its chunk 0), if any."""
        from qdrant_client import models

        points, _ = await self._client.scroll(
            collection_name=physical,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)),
                    models.FieldCondition(key="chunk_index", match=models.MatchValue(value=0)),
                ]
            ),
            limit=_SCROLL_PAGE,
            with_payload=["ingest_ts", "content_hash", "chunk_count"],
            with_vectors=False,
        )
        best: StoredVersion | None = None
        for p in points:
            payload = p.payload or {}
            ts = int(payload.get("ingest_ts") or 0)
            if best is None or ts > best.ingest_ts:
                best = StoredVersion(
                    ingest_ts=ts,
                    content_hash=payload.get("content_hash"),
                    chunk_count=int(payload.get("chunk_count") or 0),
                )
        return best

    async def replace_document(
        self,
        physical: str,
        doc_id: str,
        chunks: list[PreparedChunk],
        payload_base: dict[str, Any],
        document: dict[str, Any],
        *,
        ts: int | None = None,
    ) -> int:
        """Write a new version of a document, then remove the older ones (see module docstring).

        `ts` is the version's timestamp: one per ingest (the same in every collection it is written
        to), or the source's own when a rebuild copies a document, so that any live write is newer
        and wins.
        """
        from qdrant_client import models

        ts = ts if ts is not None else _now_us()
        points: list[models.PointStruct] = []
        for idx, chunk in enumerate(chunks):
            payload = {
                **payload_base,
                "doc_id": doc_id,
                "chunk_index": idx,
                "chunk_count": len(chunks),
                "heading_path": chunk.heading_path,
                "ingest_ts": ts,
                "text": chunk.text,
            }
            if idx == 0:
                payload["document"] = document
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{physical}:{doc_id}:{ts}:{idx}")),
                    vector={
                        "dense": chunk.dense,
                        "sparse": models.SparseVector(
                            indices=chunk.sparse.indices, values=chunk.sparse.values
                        ),
                    },
                    payload=payload,
                )
            )

        await self._client.upsert(collection_name=physical, points=points, wait=True)
        await self.prune_older(physical, doc_id, ts)
        newer = await self._client.count(
            collection_name=physical, count_filter=self._doc_filter(doc_id, gt=ts), exact=True
        )
        if newer.count:
            # A newer write of this doc_id finished its upsert first: it wins, this version goes.
            await self._client.delete(
                collection_name=physical,
                points_selector=models.FilterSelector(
                    filter=self._doc_filter(doc_id, gte=ts, lte=ts)
                ),
                wait=True,
            )
            emit(logger, "ingest_superseded", collection=physical, doc_id=doc_id)
        return len(points)

    async def prune_older(self, physical: str, doc_id: str, ts: int) -> None:
        """Delete every version of doc_id older than ts (idempotent). Also run when an ingest finds
        the newest version unchanged, so a write whose clean-up failed is completed by its retry.
        """
        from qdrant_client import models

        await self._client.delete(
            collection_name=physical,
            points_selector=models.FilterSelector(filter=self._older_than(doc_id, ts)),
            wait=True,
        )

    async def delete_document(self, logical: str, doc_id: str) -> None:
        """Delete every version of a document wherever it is written."""
        from qdrant_client import models

        for physical in await self.write_targets(logical):
            await self._client.delete(
                collection_name=physical,
                points_selector=models.FilterSelector(filter=self._doc_filter(doc_id)),
                wait=True,
            )

    async def iter_documents(self, physical: str) -> AsyncIterator[StoredDocument]:
        """Every stored source document (chunk 0 with its `document` payload), newest version per
        doc_id.
        """
        from qdrant_client import models

        offset: Any = None
        while True:
            points, offset = await self._client.scroll(
                collection_name=physical,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(key="chunk_index", match=models.MatchValue(value=0))
                    ]
                ),
                limit=_SCROLL_PAGE,
                offset=offset,
                with_payload=["doc_id", "ingest_ts", "document"],
                with_vectors=False,
            )
            for p in points:
                payload = p.payload or {}
                document = payload.get("document")
                if isinstance(document, dict) and payload.get("doc_id"):
                    yield StoredDocument(
                        doc_id=str(payload["doc_id"]),
                        ingest_ts=int(payload.get("ingest_ts") or 0),
                        document=document,
                    )
            if offset is None:
                return

    async def count_without_source(self, physical: str) -> int:
        """Chunk-0 points that carry no source document (written by 0.1.0): rebuild cannot copy
        them.
        """
        from qdrant_client import models

        first_chunks = models.FieldCondition(key="chunk_index", match=models.MatchValue(value=0))
        result = await self._client.count(
            collection_name=physical, count_filter=models.Filter(must=[first_chunks]), exact=True
        )
        # 0.1.0 never wrote content_hash; every version since writes it together with `document`.
        without = await self._client.count(
            collection_name=physical,
            count_filter=models.Filter(
                must=[
                    first_chunks,
                    models.IsEmptyCondition(is_empty=models.PayloadField(key="content_hash")),
                ]
            ),
            exact=True,
        )
        return min(result.count, without.count)

    async def count(self, physical: str) -> int:
        return (await self._client.count(collection_name=physical, exact=True)).count

    # --- versions ---
    # ---------------------------------------------------------------------------------------------

    async def promote(self, logical: str, *, drop_legacy: bool = False) -> CollectionState:
        """Serve the logical collection from this version's physical collection."""
        from qdrant_client import models

        state = await self._state(logical)
        if state.serving == state.target:
            return state
        async with self._lock:
            if state.mode == "legacy":
                if not drop_legacy:
                    raise ValueError(
                        f"'{logical}' is a 0.1.0 collection with points; promoting deletes it "
                        "(pass drop_legacy=true once callers have re-sent their documents)"
                    )
                await self._client.delete_collection(logical)
                await self._set_alias(logical, state.target)
            else:
                await self._client.update_collection_aliases(
                    change_aliases_operations=[
                        models.DeleteAliasOperation(
                            delete_alias=models.DeleteAlias(alias_name=logical)
                        ),
                        models.CreateAliasOperation(
                            create_alias=models.CreateAlias(
                                collection_name=state.target, alias_name=logical
                            )
                        ),
                    ]
                )
            metrics.COLLECTION_INFO.labels(
                collection=logical, physical=state.serving, mode=state.mode
            ).set(0)
            new_state = CollectionState(logical, state.target, state.target, "alias")
            self._states[logical] = new_state
            metrics.COLLECTION_INFO.labels(
                collection=logical, physical=new_state.serving, mode="alias"
            ).set(1)
            emit(logger, "collection_promoted", collection=logical, serving=new_state.serving)
            return new_state

    async def physical_collections(self) -> list[str]:
        return sorted(c.name for c in (await self._client.get_collections()).collections)

    async def drop_physical(self, physical: str) -> None:
        """Delete an old physical collection that nothing serves or writes."""
        if not self.ready:
            await self.ensure_collections()  # know what is served before deleting anything
        in_use = {name for s in self._states.values() for name in (s.serving, s.target)}
        if physical in in_use:
            raise ValueError(f"'{physical}' is being served or written; it cannot be deleted")
        if SEP not in physical or physical.split(SEP, 1)[0] not in COLLECTIONS:
            raise ValueError(f"'{physical}' is not a moks-rag index version")
        await self._client.delete_collection(physical)
        self._indexed.pop(physical, None)

    # --- search ---
    async def hybrid_search(
        self,
        collection: str,
        dense_vector: list[float],
        sparse_vector: SparseVectorData,
        top_k: int = 6,
        filters: dict[str, str | int | float | bool] | None = None,
        min_dense_score: float | None = None,
    ) -> list[RetrievedChunk]:
        """Dense + sparse prefetch fused by RRF. Each chunk also carries its dense cosine
        (`dense_score`); with min_dense_score, chunks less similar than that are dropped (the fused
        score is rank-based and cannot be compared with a fixed number).
        """
        from qdrant_client import models

        physical = (await self._state(collection)).serving
        if filters:
            await self._ensure_filter_indexes(physical, filters)
        query_filter = self._filter(filters)

        prefetch_k = max(self._prefetch_limit, top_k * 3)
        prefetches: list[models.Prefetch] = [
            models.Prefetch(
                query=dense_vector,
                using="dense",
                limit=prefetch_k,
                filter=query_filter,
                score_threshold=min_dense_score,
            )
        ]
        if sparse_vector.indices:
            prefetches.append(
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_vector.indices, values=sparse_vector.values
                    ),
                    using="sparse",
                    limit=prefetch_k,
                    filter=query_filter,
                )
            )

        with metrics.stage("search"):
            result = await self._client.query_points(
                collection_name=physical,
                prefetch=prefetches,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k,
                with_payload=models.PayloadSelectorExclude(exclude=list(HIDDEN_KEYS)),
                query_filter=query_filter,
            )
            ids = [p.id for p in result.points]
            dense_scores: dict[Any, float] = {}
            if ids:
                dense = await self._client.query_points(
                    collection_name=physical,
                    query=dense_vector,
                    using="dense",
                    query_filter=models.Filter(must=[models.HasIdCondition(has_id=ids)]),
                    limit=len(ids),
                    with_payload=False,
                )
                dense_scores = {p.id: float(p.score) for p in dense.points}

        chunks: list[RetrievedChunk] = []
        for point in result.points:
            dense_score = dense_scores.get(point.id)
            if min_dense_score is not None and (
                dense_score is None or dense_score < min_dense_score
            ):
                continue
            payload = dict(point.payload or {})
            text = str(payload.pop("text", ""))
            chunks.append(
                RetrievedChunk(
                    id=str(point.id),
                    doc_id=str(payload.get("doc_id", "")),
                    score=round(float(point.score), 6),
                    dense_score=round(dense_score, 6) if dense_score is not None else None,
                    text=text,
                    metadata=payload,
                )
            )
        return chunks
