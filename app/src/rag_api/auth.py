"""Bearer authentication with per-service tokens and access rules, plus structured access logging.

Two kinds of token:

- A per-service token (RAG_SERVICE_TOKENS, JSON {"<service>": "<token>"}): the service is who the
  token belongs to, and RAG_ACCESS_POLICY says which collections it may read and write. A service
  without a policy entry may do nothing.
- The shared token of 0.1.0 (RAG_BEARER_TOKEN): full access, as before, while
  RAG_SHARED_TOKEN_ENABLED is true. It exists so callers keep working during the move to their own
  tokens; turn it off afterwards.

X-Service-Name is only a label. With a per-service token the logged service is the token's owner (a
different claim is logged as such); with the shared token it stays the claimed name, as in 0.1.0.
"""

from __future__ import annotations

import hmac
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field

from fastapi import Depends, HTTPException, Request, status
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from rag_api import metrics
from rag_api.config import COLLECTIONS, AccessRule, Settings

logger = logging.getLogger("rag_api.access")

SHARED_TOKEN_SERVICE = "shared-token"
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:\-]{1,128}$")
_SERVICE_NAME = re.compile(r"^[A-Za-z0-9._\-]{1,64}$")


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


def _clean_service_name(raw: str | None) -> str:
    value = (raw or "").strip()
    return value if _SERVICE_NAME.match(value) else "unknown"


@dataclass(frozen=True)
class Principal:
    """The authenticated caller and what it may do."""

    service: str
    claimed_service: str
    request_id: str
    read: frozenset[str] = field(default_factory=frozenset)
    write: frozenset[str] = field(default_factory=frozenset)
    admin: bool = False
    shared_token: bool = False

    @property
    def service_name(self) -> str:
        """The name logs use: the claimed one for the shared token (as in 0.1.0), else the token's
        owner.
        """
        return self.claimed_service if self.shared_token else self.service

    @property
    def metrics_service(self) -> str:
        """A bounded label for metrics: never a claimed name."""
        return SHARED_TOKEN_SERVICE if self.shared_token else self.service

    def can_read(self, collection: str) -> bool:
        return "*" in self.read or collection in self.read

    def can_write(self, collection: str) -> bool:
        return "*" in self.write or collection in self.write

    def require_read(self, collection: str) -> None:
        if not self.can_read(collection):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Service '{self.service}' may not read collection '{collection}'",
            )

    def require_write(self, collection: str) -> None:
        if not self.can_write(collection):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Service '{self.service}' may not write collection '{collection}'",
            )

    def require_admin(self) -> None:
        if not self.admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Service '{self.service}' may not use admin endpoints",
            )


class TokenRegistry:
    """Maps presented tokens to principals; built once from Settings."""

    def __init__(self, settings: Settings) -> None:
        self._shared = settings.rag_bearer_token if settings.rag_shared_token_enabled else ""
        self._tokens: list[tuple[bytes, str]] = []
        self._policy: dict[str, AccessRule] = dict(settings.rag_access_policy)
        self.problems: list[str] = []

        holders: dict[str, list[str]] = {}
        for service, token in settings.rag_service_tokens.items():
            holders.setdefault(token, []).append(service)
        for service, token in settings.rag_service_tokens.items():
            if not _SERVICE_NAME.match(service):
                self.problems.append(f"service name {service!r} is not [A-Za-z0-9._-]{{1,64}}")
                continue
            if not token or len(token) < 16:
                self.problems.append(f"token of {service!r} is shorter than 16 characters; ignored")
                continue
            if len(holders[token]) > 1 or token == settings.rag_bearer_token:
                # Ambiguous: nobody gets it rather than whichever service happens to come first.
                self.problems.append(f"token of {service!r} is not unique; ignored")
                continue
            self._tokens.append((token.encode(), service))
            if service not in self._policy:
                self.problems.append(
                    f"{service!r} has a token but no access policy: it may do nothing"
                )
        known = set(COLLECTIONS) | {"*"}
        for service, rule in self._policy.items():
            for name in [*rule.read, *rule.write]:
                if name not in known:
                    self.problems.append(f"policy of {service!r} names unknown collection {name!r}")

    @property
    def configured(self) -> bool:
        return bool(self._shared or self._tokens)

    def resolve(self, presented: str, claimed: str, request_id: str) -> Principal | None:
        presented_bytes = presented.encode()
        owner: str | None = None
        # Compare against every token so the time taken does not say which service (if any) matched.
        for token, service in self._tokens:
            if hmac.compare_digest(presented_bytes, token):
                owner = service
        if owner is not None:
            rule = self._policy.get(owner, AccessRule())
            return Principal(
                service=owner,
                claimed_service=claimed,
                request_id=request_id,
                read=frozenset(rule.read) | frozenset(rule.write),
                write=frozenset(rule.write),
                admin=rule.admin,
            )
        if self._shared and hmac.compare_digest(presented_bytes, self._shared.encode()):
            everything = frozenset({"*"})
            return Principal(
                service=SHARED_TOKEN_SERVICE,
                claimed_service=claimed,
                request_id=request_id,
                read=everything,
                write=everything,
                admin=True,
                shared_token=True,
            )
        return None


def request_id_of(request: Request) -> str:
    rid = getattr(request.state, "request_id", None)
    return rid if isinstance(rid, str) else str(uuid.uuid4())


def get_app_settings(request: Request) -> Settings:
    """Fetch Settings instance attached to FastAPI app state."""
    return request.app.state.settings


def require_auth(request: Request) -> Principal:
    """Validate `Authorization: Bearer <token>` and return who is calling."""
    registry: TokenRegistry = request.app.state.tokens
    if not registry.configured:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Server misconfiguration: no token is configured "
                "(RAG_BEARER_TOKEN / RAG_SERVICE_TOKENS)"
            ),
        )

    authorization = request.headers.get("authorization") or ""
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    claimed = _clean_service_name(request.headers.get("x-service-name"))
    principal = registry.resolve(
        authorization.removeprefix("Bearer ").strip(), claimed, request_id_of(request)
    )
    if principal is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not principal.shared_token and claimed not in ("unknown", principal.service):
        log_event(
            "service_name_mismatch",
            request_id=principal.request_id,
            service=principal.service,
            claimed_service=claimed,
        )
    request.state.service_name = principal.service_name
    request.state.auth = "shared-token" if principal.shared_token else "service-token"
    return principal


AuthDep = Depends(require_auth)


class AccessLogMiddleware:
    """Logs and measures every request until its last byte is sent, so streamed answers count in
    full.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        raw_id = headers.get("x-request-id", "")
        request_id = raw_id if _REQUEST_ID.match(raw_id) else str(uuid.uuid4())
        state = scope.setdefault("state", {})
        state["request_id"] = request_id
        state["service_name"] = _clean_service_name(headers.get("x-service-name"))

        started = time.perf_counter()
        status_code = 500
        logged = False

        def finish() -> None:
            nonlocal logged
            if logged:
                return
            logged = True
            duration = time.perf_counter() - started
            route = scope.get("route")
            route_path = getattr(route, "path", None) or "unmatched"
            metrics.HTTP_REQUESTS.labels(
                route=route_path, method=scope["method"], status=str(status_code)
            ).inc()
            metrics.HTTP_DURATION.labels(route=route_path, method=scope["method"]).observe(duration)
            log_event(
                "http_request",
                request_id=request_id,
                service=state.get("service_name", "unknown"),
                auth=state.get("auth", "none"),
                method=scope["method"],
                path=scope.get("path", ""),
                status_code=status_code,
                duration_ms=round(duration * 1000.0, 2),
            )

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                message.setdefault("headers", [])
                message["headers"] = [
                    *message["headers"],
                    (b"x-request-id", request_id.encode("latin-1")),
                ]
            await send(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                finish()

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            finish()
