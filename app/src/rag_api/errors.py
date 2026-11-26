"""Failures of the services rag-api depends on, mapped to HTTP answers a caller can act on.

A dependency that is down answers 503, one that is too slow 504, a rate limit stays 429 (callers
retry it), and anything else the dependency refused or broke becomes 502. The body names the
dependency, never the request.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal

import httpx

from rag_api import metrics

UpstreamKind = Literal["unavailable", "timeout", "rate_limited", "rejected", "error"]

_HTTP_STATUS: dict[str, int] = {
    "unavailable": 503,
    "timeout": 504,
    "rate_limited": 429,
    "rejected": 502,
    "error": 502,
}


class UpstreamError(Exception):
    """A dependency (bge-m3, qdrant, docling-serve, litellm) did not do what rag-api asked."""

    def __init__(
        self,
        dependency: str,
        message: str,
        *,
        kind: UpstreamKind = "error",
        status: int | None = None,
        retry_after: str | None = None,
    ) -> None:
        super().__init__(f"{dependency}: {message}")
        self.dependency = dependency
        self.message = message
        self.kind = kind
        self.status = status
        self.retry_after = retry_after
        metrics.UPSTREAM_ERRORS.labels(dependency=dependency, kind=kind).inc()

    @property
    def http_status(self) -> int:
        return _HTTP_STATUS[self.kind]


class EmbeddingInputTooLong(ValueError):
    """A chunk is longer than the embedding model's context; embedding it would silently drop its
    end.
    """


class DocumentConversionFailed(ValueError):
    """docling-serve could not turn an uploaded file into text."""


def _snippet(response: httpx.Response) -> str:
    try:
        return response.text[:300]
    except Exception:  # noqa: BLE001 - a streamed body that was never read
        return ""


def from_httpx(dependency: str, exc: httpx.HTTPError) -> UpstreamError:
    """Translate an httpx failure into an UpstreamError."""
    if isinstance(exc, httpx.TimeoutException):
        return UpstreamError(dependency, "did not answer in time", kind="timeout")
    if isinstance(exc, httpx.HTTPStatusError):
        return from_response(dependency, exc.response)
    if isinstance(exc, httpx.TransportError):
        return UpstreamError(dependency, f"unreachable ({type(exc).__name__})", kind="unavailable")
    return UpstreamError(dependency, type(exc).__name__)


def from_response(
    dependency: str, response: httpx.Response, body: str | None = None
) -> UpstreamError:
    """An UpstreamError for a non-2xx answer."""
    status = response.status_code
    detail = (body if body is not None else _snippet(response)).strip()
    message = f"answered HTTP {status}" + (f": {detail}" if detail else "")
    if status == 429:
        return UpstreamError(
            dependency,
            message,
            kind="rate_limited",
            status=status,
            retry_after=response.headers.get("retry-after"),
        )
    if status in (502, 503, 504):
        return UpstreamError(dependency, message, kind="unavailable", status=status)
    if 400 <= status < 500:
        return UpstreamError(dependency, message, kind="rejected", status=status)
    return UpstreamError(dependency, message, kind="error", status=status)


@contextmanager
def upstream(dependency: str) -> Iterator[None]:
    """Run calls to one dependency; any httpx failure leaves as an UpstreamError."""
    try:
        yield
    except UpstreamError:
        raise
    except httpx.HTTPError as exc:
        raise from_httpx(dependency, exc) from exc
