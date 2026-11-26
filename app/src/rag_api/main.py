"""FastAPI application entrypoint for moks-rag."""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from pydantic import ValidationError

from rag_api import __version__, metrics
from rag_api.auth import (
    AccessLogMiddleware,
    AuthDep,
    Principal,
    TokenRegistry,
    configure_logging,
    log_event,
)
from rag_api.config import COLLECTIONS, VIRTUAL_MODELS, CollectionName, Settings, get_settings
from rag_api.docling import DoclingClient
from rag_api.embeddings import EmbeddingClient
from rag_api.errors import (
    DocumentConversionFailed,
    EmbeddingInputTooLong,
    UpstreamError,
    from_httpx,
)
from rag_api.indexer import Indexer, IngestDoc, NoIndexableText
from rag_api.llm import DONE, LiteLLMClient, LLMStream
from rag_api.models import (
    ChatCompletionChoice,
    ChatCompletionMessageOut,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
    CollectionsStatusResponse,
    CollectionStatus,
    DeleteDocumentResponse,
    IngestJSONRequest,
    IngestResponse,
    ModelCard,
    ModelListResponse,
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
)
from rag_api.prompts import (
    DEFAULT_QUERY_PROMPT,
    condense_request,
    grounded_user_message,
    is_open_webui_task,
    system_prompt,
)
from rag_api.rerank import RerankGate
from rag_api.store import HIDDEN_KEYS, QdrantStore

logger = logging.getLogger("rag_api.main")

_FILTER_KEY = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")


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


def _check_filters(filters: dict[str, Any] | None) -> None:
    for key in filters or {}:
        if not _FILTER_KEY.match(key) or key in HIDDEN_KEYS:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"filter key {key!r} is not a filterable payload field",
            )


def _validation_detail(exc: ValidationError) -> Any:
    return jsonable_encoder(exc.errors(include_url=False, include_context=False))


def _upstream_response(request: Request, exc: UpstreamError) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    log_event(
        "upstream_error",
        request_id=request_id,
        dependency=exc.dependency,
        kind=exc.kind,
        upstream_status=exc.status,
        detail=exc.message[:300],
    )
    headers = {"Retry-After": exc.retry_after} if exc.retry_after else None
    return JSONResponse(
        status_code=exc.http_status,
        content={
            "detail": f"{exc.dependency} {exc.message}",
            "dependency": exc.dependency,
            "kind": exc.kind,
            "request_id": request_id,
        },
        headers=headers,
    )


def create_app(
    settings: Settings | None = None,
    *,
    docling_client: Any = None,
    embedding_client: Any = None,
    qdrant_store: Any = None,
    litellm_client: Any = None,
    reranker: Any = None,
) -> FastAPI:
    """Build and wire the FastAPI application (supports dependency injection for tests)."""
    cfg = settings or get_settings()
    configure_logging(cfg.log_level)
    tokens = TokenRegistry(cfg)
    for problem in tokens.problems:
        logger.warning(json.dumps({"event": "access_config_problem", "detail": problem}))

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
        application.state.reranker = reranker or RerankGate(cfg)
        application.state.indexer = Indexer(
            cfg, application.state.qdrant, application.state.embeddings
        )
        if cfg.metrics_port:
            metrics.start_metrics_server(cfg.metrics_port)

        if cfg.ensure_collections_on_startup:
            try:
                await application.state.qdrant.ensure_collections()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    json.dumps({"event": "qdrant_bootstrap_deferred", "error": type(exc).__name__})
                )

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
    app.state.tokens = tokens
    app.add_middleware(AccessLogMiddleware)

    # --- Error mapping ---

    @app.exception_handler(UpstreamError)
    async def upstream_error_handler(request: Request, exc: UpstreamError) -> JSONResponse:
        return _upstream_response(request, exc)

    @app.exception_handler(httpx.HTTPError)
    async def httpx_error_handler(request: Request, exc: httpx.HTTPError) -> JSONResponse:
        return _upstream_response(request, from_httpx("dependency", exc))

    try:
        from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

        @app.exception_handler(UnexpectedResponse)
        async def qdrant_status_handler(request: Request, exc: UnexpectedResponse) -> JSONResponse:
            code = exc.status_code or 500
            kind = "unavailable" if code in (502, 503, 504) else "error"
            return _upstream_response(
                request, UpstreamError("qdrant", f"answered HTTP {code}", kind=kind, status=code)
            )

        @app.exception_handler(ResponseHandlingException)
        async def qdrant_transport_handler(
            request: Request, exc: ResponseHandlingException
        ) -> JSONResponse:
            source = getattr(exc, "source", None)
            kind = "timeout" if isinstance(source, httpx.TimeoutException) else "unavailable"
            return _upstream_response(request, UpstreamError("qdrant", "unreachable", kind=kind))
    except ImportError:  # pragma: no cover
        pass

    @app.exception_handler(EmbeddingInputTooLong)
    async def too_long_handler(request: Request, exc: EmbeddingInputTooLong) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(DocumentConversionFailed)
    async def conversion_handler(request: Request, exc: DocumentConversionFailed) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(NoIndexableText)
    async def no_text_handler(request: Request, exc: NoIndexableText) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    # --- Helpers ---

    async def retrieve(
        request: Request,
        principal: Principal,
        *,
        collection: str,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None,
        min_dense_score: float | None,
        endpoint: str,
    ) -> tuple[list[RetrievedChunk], str]:
        _check_filters(filters)
        state = request.app.state
        dense, sparse = await state.embeddings.embed_query(query)
        gate: RerankGate = state.reranker
        candidates = await state.qdrant.hybrid_search(
            collection=collection,
            dense_vector=dense,
            sparse_vector=sparse,
            top_k=gate.candidates(top_k),
            filters=filters,
            min_dense_score=min_dense_score,
        )
        chunks, outcome = await gate.rerank(query, candidates, top_k)
        metrics.RETRIEVALS.labels(
            service=principal.metrics_service, collection=collection, endpoint=endpoint
        ).inc()
        if not chunks:
            metrics.ZERO_HITS.labels(service=principal.metrics_service, collection=collection).inc()
        return chunks, outcome

    async def complete(
        request: Request,
        principal: Principal,
        *,
        model: str,
        purpose: str,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        labels = {"service": principal.metrics_service, "model": model, "purpose": purpose}
        try:
            with metrics.stage("llm"):
                resp = await request.app.state.litellm.chat_completions(
                    model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
                )
        except Exception:
            metrics.LLM_REQUESTS.labels(**labels, outcome="error").inc()
            raise
        metrics.LLM_REQUESTS.labels(**labels, outcome="ok").inc()
        metrics.record_usage(principal.metrics_service, model, purpose, resp.get("usage"))
        return resp

    def stream_response(
        principal: Principal,
        stream: LLMStream,
        *,
        model: str,
        virtual_model: str,
        purpose: str,
        log_fields: dict[str, Any],
    ) -> StreamingResponse:
        started = time.perf_counter()

        async def body() -> AsyncIterator[str]:
            outcome = "ok"
            try:
                async for event in stream.events():
                    if event == DONE:
                        yield "data: [DONE]\n\n"
                        continue
                    assert isinstance(event, dict)
                    event["model"] = virtual_model
                    yield f"data: {json.dumps(event)}\n\n"
            except UpstreamError as exc:
                outcome = "error"
                error = {
                    "message": f"{exc.dependency} {exc.message}",
                    "type": "upstream_error",
                    "code": exc.kind,
                }
                yield f"data: {json.dumps({'error': error})}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                await stream.aclose()
                metrics.LLM_REQUESTS.labels(
                    service=principal.metrics_service, model=model, purpose=purpose, outcome=outcome
                ).inc()
                metrics.record_usage(principal.metrics_service, model, purpose, stream.usage)
                metrics.STAGE_DURATION.labels(stage="llm").observe(time.perf_counter() - started)
                log_event(
                    "openai_stream_finished",
                    **log_fields,
                    outcome=outcome,
                    finish_reason=stream.finish_reason,
                    usage=stream.usage,
                    duration_ms=round((time.perf_counter() - started) * 1000, 2),
                )

        return StreamingResponse(
            body(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    def require_collection(name: str) -> CollectionName:
        if name not in COLLECTIONS:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"collection must be one of {list(COLLECTIONS)}",
            )
        return name  # type: ignore[return-value]

    # --- Landing: https://jithendar.com/rag has nothing to show, so send browsers to the docs ---

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url=f"{cfg.root_path}/docs")

    # --- Health & readiness probes (unauthenticated for kubelet) ---

    @app.get("/health", tags=["ops"])
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": cfg.app_name, "version": __version__}

    @app.get("/ready", tags=["ops"])
    async def ready(request: Request) -> dict[str, Any]:
        try:
            await request.app.state.qdrant.healthcheck()
        except Exception as exc:  # noqa: BLE001
            log_event(
                "readiness_failed",
                request_id=getattr(request.state, "request_id", None),
                error=type(exc).__name__,
                detail=str(exc)[:300],
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Qdrant unavailable"
            ) from exc
        return {"status": "ready", "collections": list(COLLECTIONS)}

    # --- Ingestion endpoint (JSON text or multipart file upload via docling-serve) ---

    @app.post("/ingest", response_model=IngestResponse, tags=["rag"])
    async def ingest(request: Request, principal: Principal = AuthDep) -> IngestResponse:
        if cfg.max_upload_bytes is not None:
            length = request.headers.get("content-length")
            if length is None:
                raise HTTPException(
                    status_code=status.HTTP_411_LENGTH_REQUIRED,
                    detail="Content-Length is required (MAX_UPLOAD_BYTES is set)",
                )
            if not length.isdigit() or int(length) > cfg.max_upload_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail=f"Request body is larger than {cfg.max_upload_bytes} bytes",
                )

        content_type = (request.headers.get("content-type") or "").lower()
        parsed_via_docling = False
        upload: Any = None
        multipart = "multipart/form-data" in content_type

        if multipart:
            form = await request.form()
            metadata: Any = {}
            metadata_raw = form.get("metadata")
            if isinstance(metadata_raw, str) and metadata_raw.strip():
                try:
                    metadata = json.loads(metadata_raw)
                except json.JSONDecodeError as exc:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                        detail=f"Invalid metadata JSON: {exc}",
                    ) from exc
            upload = form.get("file")
            has_file = upload is not None and hasattr(upload, "read")
            text_field = str(form.get("text") or "").strip()
            if not has_file and not text_field:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail="Multipart ingest requires either 'file' or 'text'",
                )
            fields = {
                "collection": str(form.get("collection") or "").strip(),
                "doc_id": str(form.get("doc_id") or "").strip(),
                "title": str(form.get("title") or "").strip() or None,
                "source": str(form.get("source") or "").strip() or None,
                "metadata": metadata,
                "chunk_size": str(form.get("chunk_size") or "").strip() or None,
                "chunk_overlap": str(form.get("chunk_overlap") or "").strip() or None,
                "text": text_field or "(file)",
            }
            if not has_file:
                upload = None
        else:
            try:
                fields = await request.json()
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail=f"Invalid JSON ingest payload: {exc}",
                ) from exc

        try:
            payload = IngestJSONRequest.model_validate(fields)
        except ValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=_validation_detail(exc)
            ) from exc

        collection = payload.collection
        principal.require_write(collection)
        metadata_out = dict(payload.metadata)
        title, source = payload.title, payload.source
        indexer: Indexer = request.app.state.indexer
        labels = {"service": principal.metrics_service, "collection": collection}

        try:
            async with indexer.slot():
                text = payload.text
                if upload is not None:
                    raw_bytes = await upload.read()
                    if not raw_bytes:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty"
                        )
                    filename = getattr(upload, "filename", None) or "upload"
                    upload_ct = getattr(upload, "content_type", None)
                    title = title or filename
                    metadata_out.setdefault("filename", filename)
                    if DoclingClient.needs_docling(filename, upload_ct):
                        text = await request.app.state.docling.parse_file(
                            filename=filename, content=raw_bytes, content_type=upload_ct
                        )
                        parsed_via_docling = True
                    else:
                        text = raw_bytes.decode("utf-8", errors="replace")

                if title:
                    metadata_out.setdefault("title", title)
                if source:
                    metadata_out.setdefault("source", source)
                metadata_out["ingested_by"] = principal.service_name

                doc = IngestDoc(
                    collection=collection,
                    doc_id=payload.doc_id,
                    text=text,
                    title=title,
                    source=source,
                    metadata=metadata_out,
                    chunk_size=payload.chunk_size or cfg.chunk_size,
                    chunk_overlap=(
                        payload.chunk_overlap
                        if payload.chunk_overlap is not None
                        else cfg.chunk_overlap
                    ),
                    ingested_by=principal.service_name,
                )
                try:
                    outcome = await indexer.ingest(doc)
                except ValueError as exc:
                    if isinstance(exc, NoIndexableText | EmbeddingInputTooLong):
                        raise
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
                    ) from exc
        except HTTPException:
            metrics.INGESTS.labels(**labels, outcome="rejected").inc()
            raise
        except (NoIndexableText, EmbeddingInputTooLong, DocumentConversionFailed):
            metrics.INGESTS.labels(**labels, outcome="rejected").inc()
            raise
        except Exception:
            metrics.INGESTS.labels(**labels, outcome="error").inc()
            raise

        metrics.INGESTS.labels(
            **labels, outcome="unchanged" if outcome.unchanged else "indexed"
        ).inc()
        if not outcome.unchanged:
            metrics.INGEST_CHUNKS.labels(**labels).inc(outcome.chunks)
        log_event(
            "rag_ingest",
            request_id=principal.request_id,
            service=principal.service_name,
            collection=collection,
            doc_id=payload.doc_id,
            chunks_ingested=outcome.chunks,
            unchanged=outcome.unchanged,
            parsed_via_docling=parsed_via_docling,
        )
        return IngestResponse(
            doc_id=payload.doc_id,
            collection=collection,
            chunks_ingested=outcome.chunks,
            parsed_via_docling=parsed_via_docling,
            service=principal.service_name,
            unchanged=outcome.unchanged,
            index_version=cfg.index_version,
        )

    # --- Document deletion (doc_id in the path, slashes allowed, or as ?doc_id=) ---

    async def delete_doc(
        request: Request, principal: Principal, doc_id: str, collection: str
    ) -> DeleteDocumentResponse:
        if not doc_id:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="doc_id is required"
            )
        targets: list[str] = list(COLLECTIONS) if collection == "all" else [collection]
        for target in targets:
            principal.require_write(target)
        for target in targets:
            await request.app.state.qdrant.delete_document(target, doc_id)
        log_event(
            "rag_delete_document",
            request_id=principal.request_id,
            service=principal.service_name,
            doc_id=doc_id,
            collections=targets,
        )
        return DeleteDocumentResponse(
            doc_id=doc_id, collections=targets, deleted=True, service=principal.service_name
        )

    collection_param = Query(
        description="Target collection, or 'all' to remove it from every collection you may write",
    )

    @app.delete("/documents", response_model=DeleteDocumentResponse, tags=["rag"])
    async def delete_document_by_query(
        request: Request,
        doc_id: str = Query(min_length=1),
        collection: CollectionName | Literal["all"] = collection_param,
        principal: Principal = AuthDep,
    ) -> DeleteDocumentResponse:
        return await delete_doc(request, principal, doc_id, collection)

    @app.delete("/documents/{doc_id:path}", response_model=DeleteDocumentResponse, tags=["rag"])
    async def delete_document(
        doc_id: str,
        request: Request,
        collection: CollectionName | Literal["all"] = collection_param,
        principal: Principal = AuthDep,
    ) -> DeleteDocumentResponse:
        return await delete_doc(request, principal, doc_id, collection)

    # --- Hybrid RAG query endpoint ---

    @app.post("/query", response_model=QueryResponse, tags=["rag"])
    async def query_rag(
        payload: QueryRequest, request: Request, principal: Principal = AuthDep
    ) -> QueryResponse:
        principal.require_read(payload.collection)
        if payload.model is not None and payload.model not in cfg.allowed_llm_models():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"model must be one of {sorted(cfg.allowed_llm_models())}",
            )
        sources, reranker_outcome = await retrieve(
            request,
            principal,
            collection=payload.collection,
            query=payload.query,
            top_k=payload.top_k,
            filters=payload.filters,
            min_dense_score=payload.min_dense_score,
            endpoint="query",
        )

        llm_model = payload.model or cfg.default_llm_model
        answer: str | None = None
        if payload.include_answer:
            messages = [
                {
                    "role": "system",
                    "content": system_prompt(payload.system_prompt or DEFAULT_QUERY_PROMPT),
                },
                {"role": "user", "content": grounded_user_message(sources, payload.query)},
            ]
            llm_resp = await complete(
                request,
                principal,
                model=llm_model,
                purpose="answer",
                messages=messages,
                temperature=payload.temperature,
                max_tokens=payload.max_tokens,
            )
            answer, _, _ = _extract_answer_and_usage(llm_resp)

        log_event(
            "rag_query",
            request_id=principal.request_id,
            service=principal.service_name,
            collection=payload.collection,
            model=llm_model,
            hits=len(sources),
            include_answer=payload.include_answer,
            reranker=reranker_outcome,
        )
        return QueryResponse(
            query=payload.query,
            collection=payload.collection,
            model=llm_model,
            answer=answer,
            sources=sources,
            service=principal.service_name,
            reranker=reranker_outcome,
        )

    # --- OpenAI-compatible endpoints for Open WebUI ---

    @app.get("/v1/models", response_model=ModelListResponse, tags=["openai"])
    async def list_models(principal: Principal = AuthDep) -> ModelListResponse:
        log_event(
            "openai_list_models", request_id=principal.request_id, service=principal.service_name
        )
        cards = [
            ModelCard(id=model_id, description=meta["description"], collection=meta["collection"])
            for model_id, meta in VIRTUAL_MODELS.items()
            if principal.can_read(meta["collection"])
        ]
        return ModelListResponse(data=cards)

    def forwardable(msg: ChatMessage) -> dict[str, Any] | None:
        text = msg.text_content()
        if not text.strip():
            return None
        if msg.role == "tool":
            return {"role": "user", "content": f"[tool result]\n{text}"}
        if msg.role == "developer":
            return {"role": "system", "content": text}
        return {"role": msg.role, "content": text}

    async def condense(
        request: Request, principal: Principal, history: list[ChatMessage], latest: str
    ) -> str:
        turns = [(m.role, m.text_content()) for m in history if m.role in ("user", "assistant")]
        try:
            with metrics.stage("condense"):
                resp = await complete(
                    request,
                    principal,
                    model=cfg.fast_llm_model,
                    purpose="condense",
                    messages=condense_request(turns, latest),
                    temperature=0.0,
                    max_tokens=None,
                )
        except Exception as exc:  # noqa: BLE001 - fall back to the raw message
            log_event("condense_failed", request_id=principal.request_id, error=type(exc).__name__)
            return latest
        rewritten, _, _ = _extract_answer_and_usage(resp)
        return rewritten.strip() or latest

    @app.post("/v1/chat/completions", response_model=None, tags=["openai"])
    async def chat_completions(
        payload: ChatCompletionRequest, request: Request, principal: Principal = AuthDep
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
        collection: str = virtual_cfg["collection"]
        principal.require_read(collection)

        last_user = next(
            (
                i
                for i in range(len(payload.messages) - 1, -1, -1)
                if payload.messages[i].role == "user" and payload.messages[i].text_content().strip()
            ),
            None,
        )
        if last_user is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="At least one non-empty user message is required",
            )
        latest = payload.messages[last_user].text_content().strip()
        log_fields: dict[str, Any] = {
            "request_id": principal.request_id,
            "service": principal.service_name,
            "virtual_model": payload.model,
            "collection": collection,
            "stream": payload.stream,
        }

        if is_open_webui_task(latest):
            # Title/tags/follow-up generation: no retrieval, the fast model, messages as sent.
            llm_model, purpose = cfg.fast_llm_model, "task"
            forwarded = [m for m in (forwardable(msg) for msg in payload.messages) if m]
            log_event("openai_task_request", **log_fields, llm_model=llm_model)
        else:
            llm_model, purpose = virtual_cfg.get("llm_model") or cfg.default_llm_model, "chat"
            history = [
                m for m in payload.messages[:last_user] if m.role not in ("system", "developer")
            ]
            if cfg.max_history_messages is not None:
                history = history[-cfg.max_history_messages :]
            search_query = latest
            if cfg.condense_followups and any(m.role == "assistant" for m in history):
                search_query = await condense(request, principal, history, latest)

            min_dense = (
                payload.min_dense_score
                if payload.min_dense_score is not None
                else cfg.default_min_dense_score
            )
            sources, reranker_outcome = await retrieve(
                request,
                principal,
                collection=collection,
                query=search_query,
                top_k=payload.top_k or cfg.default_top_k,
                filters=payload.filters,
                min_dense_score=min_dense,
                endpoint="chat",
            )
            system_text = system_prompt(f"{virtual_cfg['system_prompt']}\nCollection: {collection}")
            forwarded = [{"role": "system", "content": system_text}]
            forwarded += [
                m
                for m in (forwardable(msg) for msg in payload.messages[:last_user])
                if m and m["role"] == "system"
            ]
            forwarded += [m for m in (forwardable(msg) for msg in history) if m]
            forwarded.append({"role": "user", "content": grounded_user_message(sources, latest)})
            log_event(
                "openai_chat_completions",
                **log_fields,
                llm_model=llm_model,
                hits=len(sources),
                condensed=search_query != latest,
                reranker=reranker_outcome,
            )

        if payload.stream:
            try:
                stream = await request.app.state.litellm.open_stream(
                    model=llm_model,
                    messages=forwarded,
                    temperature=payload.temperature,
                    max_tokens=payload.max_tokens,
                )
            except Exception:
                metrics.LLM_REQUESTS.labels(
                    service=principal.metrics_service,
                    model=llm_model,
                    purpose=purpose,
                    outcome="error",
                ).inc()
                raise
            return stream_response(
                principal,
                stream,
                model=llm_model,
                virtual_model=payload.model,
                purpose=purpose,
                log_fields=log_fields,
            )

        llm_resp = await complete(
            request,
            principal,
            model=llm_model,
            purpose=purpose,
            messages=forwarded,
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

    # --- Admin: index versions (store.py, README "Index versions") ---

    @app.get("/admin/collections", response_model=CollectionsStatusResponse, tags=["admin"])
    async def collections_status(
        request: Request, principal: Principal = AuthDep
    ) -> CollectionsStatusResponse:
        principal.require_admin()
        store: QdrantStore = request.app.state.qdrant
        indexer: Indexer = request.app.state.indexer
        out: list[CollectionStatus] = []
        for logical in COLLECTIONS:
            state = await store.state(logical)
            out.append(
                CollectionStatus(
                    collection=logical,
                    mode=state.mode,
                    serving=state.serving,
                    target=state.target,
                    points_serving=await store.count(state.serving),
                    points_target=await store.count(state.target),
                    documents_without_source=await store.count_without_source(state.serving),
                    rebuild=indexer.rebuild_status(logical),
                )
            )
        return CollectionsStatusResponse(
            index_version=cfg.index_version,
            collections=out,
            physical_collections=await store.physical_collections(),
        )

    @app.post("/admin/collections/{collection}/rebuild", status_code=202, tags=["admin"])
    async def rebuild_collection(
        collection: str,
        request: Request,
        promote: bool = Query(default=False, description="Promote when every document was copied"),
        principal: Principal = AuthDep,
    ) -> dict[str, Any]:
        principal.require_admin()
        logical = require_collection(collection)
        status_ = request.app.state.indexer.start_rebuild(logical, promote=promote)
        log_event("rebuild_started", request_id=principal.request_id, collection=logical)
        return {"collection": logical, "rebuild": status_}

    @app.post("/admin/collections/{collection}/promote", tags=["admin"])
    async def promote_collection(
        collection: str,
        request: Request,
        drop_legacy: bool = Query(default=False),
        principal: Principal = AuthDep,
    ) -> dict[str, Any]:
        principal.require_admin()
        logical = require_collection(collection)
        try:
            state = await request.app.state.qdrant.promote(logical, drop_legacy=drop_legacy)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        log_event("collection_promoted", request_id=principal.request_id, collection=logical)
        return {"collection": logical, "serving": state.serving, "mode": state.mode}

    @app.delete("/admin/physical/{physical}", tags=["admin"])
    async def drop_physical(
        physical: str, request: Request, principal: Principal = AuthDep
    ) -> dict[str, Any]:
        principal.require_admin()
        try:
            await request.app.state.qdrant.drop_physical(physical)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        log_event("physical_collection_dropped", request_id=principal.request_id, physical=physical)
        return {"deleted": physical}

    return app


app = create_app()
