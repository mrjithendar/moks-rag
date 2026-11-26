"""FastAPI application entrypoint for moks-rag."""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

import httpx
from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse

from rag_api import __version__
from rag_api.auth import (
    RequestContext,
    ServiceLoggingMiddleware,
    configure_logging,
    log_event,
    require_auth,
)
from rag_api.chunking import chunk_text
from rag_api.clients import DoclingClient, EmbeddingClient, LiteLLMClient, QdrantStore
from rag_api.config import COLLECTIONS, VIRTUAL_MODELS, CollectionName, Settings, get_settings
from rag_api.models import (
    ChatCompletionChoice,
    ChatCompletionMessageOut,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    DeleteDocumentResponse,
    IngestJSONRequest,
    IngestResponse,
    ModelCard,
    ModelListResponse,
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
)

logger = logging.getLogger("rag_api.main")


def _format_rag_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "No relevant documents were found in the knowledge base."

    blocks: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        title = chunk.metadata.get("title") or chunk.metadata.get("source") or chunk.doc_id
        blocks.append(
            f"[{idx}] doc_id={chunk.doc_id} | title={title} | score={chunk.score:.4f}\n{chunk.text}"
        )
    return "\n\n---\n\n".join(blocks)


def _extract_answer_and_usage(
    llm_response: dict[str, Any],
) -> tuple[str, str | None, ChatCompletionUsage]:
    choices = llm_response.get("choices") or []
    content = ""
    finish_reason: str | None = "stop"
    if choices:
        first = choices[0]
        message = first.get("message") or {}
        content = str(message.get("content") or "")
        finish_reason = first.get("finish_reason", "stop")

    usage_raw = llm_response.get("usage") or {}
    usage = ChatCompletionUsage(
        prompt_tokens=int(usage_raw.get("prompt_tokens") or 0),
        completion_tokens=int(usage_raw.get("completion_tokens") or 0),
        total_tokens=int(usage_raw.get("total_tokens") or 0),
    )
    return content, finish_reason, usage


def create_app(
    settings: Settings | None = None,
    *,
    docling_client: DoclingClient | None = None,
    embedding_client: EmbeddingClient | None = None,
    qdrant_store: QdrantStore | None = None,
    litellm_client: LiteLLMClient | None = None,
) -> FastAPI:
    """Build and wire the FastAPI application (supports dependency injection for tests)."""
    cfg = settings or get_settings()
    configure_logging(cfg.log_level)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(cfg.http_timeout_seconds, connect=10.0)
        )
        application.state.http_client = http_client
        application.state.docling = docling_client or DoclingClient(cfg, http_client)
        application.state.embeddings = embedding_client or EmbeddingClient(cfg, http_client)
        application.state.qdrant = qdrant_store or QdrantStore(cfg)
        application.state.litellm = litellm_client or LiteLLMClient(cfg, http_client)

        if cfg.ensure_collections_on_startup:
            try:
                await application.state.qdrant.ensure_collections()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Qdrant collection bootstrap deferred: %s", exc)

        yield

        await http_client.aclose()
        if qdrant_store is None:
            await application.state.qdrant.close()

    app = FastAPI(
        title="moks-rag API",
        version=__version__,
        root_path=cfg.root_path,
        lifespan=lifespan,
    )
    app.state.settings = cfg
    app.add_middleware(ServiceLoggingMiddleware)

    # --- Health & readiness probes (unauthenticated for kubelet) ---

    @app.get("/health", tags=["ops"])
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": cfg.app_name, "version": __version__}

    @app.get("/ready", tags=["ops"])
    async def ready(request: Request) -> dict[str, Any]:
        try:
            await request.app.state.qdrant.healthcheck()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Qdrant unavailable: {exc}",
            ) from exc
        return {"status": "ready", "collections": list(COLLECTIONS)}

    # --- Ingestion endpoint (JSON text or multipart file upload via docling-serve) ---

    @app.post("/ingest", response_model=IngestResponse, tags=["rag"])
    async def ingest(
        request: Request,
        ctx: RequestContext = Depends(require_auth),
    ) -> IngestResponse:
        content_type = (request.headers.get("content-type") or "").lower()
        parsed_via_docling = False

        if "multipart/form-data" in content_type:
            form = await request.form()
            collection_raw = str(form.get("collection") or "").strip()
            if collection_raw not in COLLECTIONS:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"collection must be one of {list(COLLECTIONS)}",
                )
            collection: CollectionName = collection_raw  # type: ignore[assignment]
            doc_id = str(form.get("doc_id") or "").strip() or str(uuid.uuid4())
            title = str(form.get("title") or "").strip() or None
            source = str(form.get("source") or "").strip() or None

            metadata: dict[str, Any] = {}
            metadata_raw = form.get("metadata")
            if isinstance(metadata_raw, str) and metadata_raw.strip():
                try:
                    parsed_meta = json.loads(metadata_raw)
                    if not isinstance(parsed_meta, dict):
                        raise ValueError("metadata must be a JSON object")
                    metadata = parsed_meta
                except (json.JSONDecodeError, ValueError) as exc:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail=f"Invalid metadata JSON: {exc}",
                    ) from exc

            chunk_size = int(form.get("chunk_size") or cfg.chunk_size)
            chunk_overlap = int(form.get("chunk_overlap") or cfg.chunk_overlap)

            upload = form.get("file")
            text_field = str(form.get("text") or "").strip()

            if upload is not None and hasattr(upload, "read"):
                raw_bytes = await upload.read()
                if not raw_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Uploaded file is empty",
                    )
                filename = getattr(upload, "filename", None) or "upload"
                upload_ct = getattr(upload, "content_type", None)
                if not title:
                    title = filename
                metadata.setdefault("filename", filename)

                if DoclingClient.needs_docling(filename, upload_ct):
                    text = await request.app.state.docling.parse_file(
                        filename=filename,
                        content=raw_bytes,
                        content_type=upload_ct,
                    )
                    parsed_via_docling = True
                else:
                    text = raw_bytes.decode("utf-8", errors="replace")
            elif text_field:
                text = text_field
            else:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Multipart ingest requires either 'file' or 'text'",
                )
        else:
            try:
                body = await request.json()
                payload = IngestJSONRequest.model_validate(body)
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Invalid JSON ingest payload: {exc}",
                ) from exc

            collection = payload.collection
            text = payload.text
            doc_id = payload.doc_id or str(uuid.uuid4())
            title = payload.title
            source = payload.source
            metadata = dict(payload.metadata)
            chunk_size = payload.chunk_size or cfg.chunk_size
            chunk_overlap = (
                payload.chunk_overlap if payload.chunk_overlap is not None else cfg.chunk_overlap
            )

        if title:
            metadata.setdefault("title", title)
        if source:
            metadata.setdefault("source", source)
        metadata["ingested_by"] = ctx.service_name

        try:
            chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc

        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document produced no indexable text chunks",
            )

        dense_vectors, sparse_vectors = await request.app.state.embeddings.embed_hybrid(chunks)
        count = await request.app.state.qdrant.upsert_chunks(
            collection=collection,
            doc_id=doc_id,
            chunks=chunks,
            dense_vectors=dense_vectors,
            sparse_vectors=sparse_vectors,
            metadata=metadata,
        )

        log_event(
            "rag_ingest",
            request_id=ctx.request_id,
            service=ctx.service_name,
            collection=collection,
            doc_id=doc_id,
            chunks_ingested=count,
            parsed_via_docling=parsed_via_docling,
        )
        return IngestResponse(
            doc_id=doc_id,
            collection=collection,
            chunks_ingested=count,
            parsed_via_docling=parsed_via_docling,
            service=ctx.service_name,
        )

    # --- Document deletion endpoint (for moks-cms webhook sync) ---

    @app.delete(
        "/documents/{doc_id}",
        response_model=DeleteDocumentResponse,
        tags=["rag"],
    )
    async def delete_document(
        doc_id: str,
        request: Request,
        collection: CollectionName | Literal["all"] = Query(
            default="all",
            description="Target collection (or 'all' to remove from every collection)",
        ),
        ctx: RequestContext = Depends(require_auth),
    ) -> DeleteDocumentResponse:
        targets: list[CollectionName] = list(COLLECTIONS) if collection == "all" else [collection]
        for target in targets:
            await request.app.state.qdrant.delete_document(collection=target, doc_id=doc_id)

        log_event(
            "rag_delete_document",
            request_id=ctx.request_id,
            service=ctx.service_name,
            doc_id=doc_id,
            collections=targets,
        )
        return DeleteDocumentResponse(
            doc_id=doc_id,
            collections=list(targets),
            deleted=True,
            service=ctx.service_name,
        )

    # --- Hybrid RAG query endpoint ---

    @app.post("/query", response_model=QueryResponse, tags=["rag"])
    async def query_rag(
        payload: QueryRequest,
        request: Request,
        ctx: RequestContext = Depends(require_auth),
    ) -> QueryResponse:
        dense_list, sparse_list = await request.app.state.embeddings.embed_hybrid([payload.query])
        sources = await request.app.state.qdrant.hybrid_search(
            collection=payload.collection,
            dense_vector=dense_list[0],
            sparse_vector=sparse_list[0],
            top_k=payload.top_k,
            filters=payload.filters,
        )

        llm_model = payload.model or cfg.default_llm_model
        answer: str | None = None

        if payload.include_answer:
            base_prompt = payload.system_prompt or (
                "You are a grounded RAG assistant. Answer strictly using the retrieved context "
                "below and cite doc_ids inline. If the context is insufficient, say so clearly."
            )
            context_block = _format_rag_context(sources)
            messages = [
                {
                    "role": "system",
                    "content": f"{base_prompt}\n\nRetrieved Context:\n{context_block}",
                },
                {"role": "user", "content": payload.query},
            ]
            llm_resp = await request.app.state.litellm.chat_completions(
                model=llm_model,
                messages=messages,
                temperature=payload.temperature,
                max_tokens=payload.max_tokens,
            )
            answer, _, _ = _extract_answer_and_usage(llm_resp)

        log_event(
            "rag_query",
            request_id=ctx.request_id,
            service=ctx.service_name,
            collection=payload.collection,
            model=llm_model,
            hits=len(sources),
            include_answer=payload.include_answer,
        )
        return QueryResponse(
            query=payload.query,
            collection=payload.collection,
            model=llm_model,
            answer=answer,
            sources=sources,
            service=ctx.service_name,
        )

    # --- OpenAI-compatible endpoints for Open WebUI ---

    @app.get("/v1/models", response_model=ModelListResponse, tags=["openai"])
    async def list_models(ctx: RequestContext = Depends(require_auth)) -> ModelListResponse:
        log_event("openai_list_models", request_id=ctx.request_id, service=ctx.service_name)
        cards = [
            ModelCard(
                id=model_id,
                description=meta["description"],
                collection=meta["collection"],
            )
            for model_id, meta in VIRTUAL_MODELS.items()
        ]
        return ModelListResponse(data=cards)

    @app.post("/v1/chat/completions", response_model=None, tags=["openai"])
    async def chat_completions(
        payload: ChatCompletionRequest,
        request: Request,
        ctx: RequestContext = Depends(require_auth),
    ) -> ChatCompletionResponse | StreamingResponse:
        virtual_cfg = VIRTUAL_MODELS.get(payload.model)
        if virtual_cfg is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"Unknown model '{payload.model}'. "
                    f"Available models: {list(VIRTUAL_MODELS.keys())}"
                ),
            )

        collection: CollectionName = virtual_cfg["collection"]  # type: ignore[assignment]
        llm_model = virtual_cfg.get("llm_model") or cfg.default_llm_model

        # Use the latest user message as the retrieval query.
        user_queries = [m.text_content().strip() for m in payload.messages if m.role == "user"]
        latest_query = next((q for q in reversed(user_queries) if q), "")
        if not latest_query:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="At least one non-empty user message is required",
            )

        top_k = payload.top_k or cfg.default_top_k
        dense_list, sparse_list = await request.app.state.embeddings.embed_hybrid([latest_query])
        sources = await request.app.state.qdrant.hybrid_search(
            collection=collection,
            dense_vector=dense_list[0],
            sparse_vector=sparse_list[0],
            top_k=top_k,
            filters=payload.filters,
        )

        context_block = _format_rag_context(sources)
        grounded_system = (
            f"{virtual_cfg['system_prompt']}\n\n"
            f"Collection: {collection}\n"
            f"Retrieved Context:\n{context_block}"
        )

        forwarded_messages: list[dict[str, Any]] = [{"role": "system", "content": grounded_system}]
        for msg in payload.messages:
            if msg.role == "system":
                forwarded_messages.append({"role": "system", "content": msg.text_content()})
            else:
                forwarded_messages.append({"role": msg.role, "content": msg.text_content()})

        log_event(
            "openai_chat_completions",
            request_id=ctx.request_id,
            service=ctx.service_name,
            virtual_model=payload.model,
            collection=collection,
            llm_model=llm_model,
            hits=len(sources),
            stream=payload.stream,
        )

        if payload.stream:
            stream_iter = request.app.state.litellm.stream_chat_completions(
                model=llm_model,
                virtual_model=payload.model,
                messages=forwarded_messages,
                temperature=payload.temperature,
                max_tokens=payload.max_tokens,
            )
            return StreamingResponse(
                stream_iter,
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        llm_resp = await request.app.state.litellm.chat_completions(
            model=llm_model,
            messages=forwarded_messages,
            temperature=payload.temperature,
            max_tokens=payload.max_tokens,
        )
        answer, finish_reason, usage = _extract_answer_and_usage(llm_resp)

        return ChatCompletionResponse(
            id=str(llm_resp.get("id") or f"chatcmpl-{uuid.uuid4().hex}"),
            created=int(llm_resp.get("created") or int(time.time())),
            model=payload.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessageOut(content=answer),
                    finish_reason=finish_reason,
                )
            ],
            usage=usage,
        )

    return app


app = create_app()
