"""Bearer authentication and per-service (X-Service-Name) structured request logging."""

from __future__ import annotations

import hmac
import json
import logging
import time
import uuid
from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from rag_api.config import Settings

logger = logging.getLogger("rag_api.access")


@dataclass(frozen=True)
class RequestContext:
    """Authenticated caller identity extracted from request headers."""

    service_name: str
    request_id: str


def configure_logging(level: str = "INFO") -> None:
    """Configure root logger for structured output suitable for Loki/Alloy scraping."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
    )


def log_event(event: str, **fields: object) -> None:
    """Emit a single-line JSON log record."""
    payload = {"event": event, **fields}
    logger.info(json.dumps(payload, default=str))


class ServiceLoggingMiddleware(BaseHTTPMiddleware):
    """Logs every HTTP request with X-Service-Name, latency, and status code."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        service_name = (request.headers.get("X-Service-Name") or "unknown").strip() or "unknown"
        request.state.request_id = request_id
        request.state.service_name = service_name

        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            duration_ms = round((time.perf_counter() - started) * 1000.0, 2)
            log_event(
                "http_request",
                request_id=request_id,
                service=service_name,
                method=request.method,
                path=request.url.path,
                status_code=status_code,
                duration_ms=duration_ms,
            )


def get_app_settings(request: Request) -> Settings:
    """Fetch Settings instance attached to FastAPI app state."""
    return request.app.state.settings


def require_auth(
    request: Request,
    authorization: str | None = Header(default=None),
    x_service_name: str | None = Header(default=None, alias="X-Service-Name"),
    settings: Settings = Depends(get_app_settings),
) -> RequestContext:
    """Validate Authorization: Bearer <RAG_BEARER_TOKEN> and extract X-Service-Name."""
    expected = settings.rag_bearer_token
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server misconfiguration: RAG_BEARER_TOKEN is not set",
        )

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    presented = authorization.removeprefix("Bearer ").strip()
    if not hmac.compare_digest(presented, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    service_name = (x_service_name or "unknown").strip() or "unknown"
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    request.state.service_name = service_name
    return RequestContext(service_name=service_name, request_id=request_id)
