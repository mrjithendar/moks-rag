"""Prometheus metrics, served on their own port (METRICS_PORT) so the public edge never routes them.

Label values are bounded: `service` is the authenticated service (or "shared-token"), never the
X-Service-Name a caller claims; `route` is the matched route template.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from prometheus_client import Counter, Gauge, Histogram, start_http_server

from rag_api.logs import emit

logger = logging.getLogger("rag_api.metrics")

_LATENCY_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300)

HTTP_REQUESTS = Counter(
    "rag_api_http_requests_total",
    "HTTP requests answered, by route template and status code",
    ["route", "method", "status"],
)
HTTP_DURATION = Histogram(
    "rag_api_http_request_duration_seconds",
    "Time from request to the last byte of the answer (streams included)",
    ["route", "method"],
    buckets=_LATENCY_BUCKETS,
)
STAGE_DURATION = Histogram(
    "rag_api_stage_duration_seconds",
    "Time spent in one pipeline stage",
    ["stage"],
    buckets=_LATENCY_BUCKETS,
)
RETRIEVALS = Counter(
    "rag_api_retrievals_total",
    "Hybrid searches run",
    ["service", "collection", "endpoint"],
)
ZERO_HITS = Counter(
    "rag_api_retrieval_zero_hits_total",
    "Hybrid searches that returned no chunk",
    ["service", "collection"],
)
LLM_REQUESTS = Counter(
    "rag_api_llm_requests_total",
    "Calls to LiteLLM",
    ["service", "model", "purpose", "outcome"],
)
LLM_TOKENS = Counter(
    "rag_api_llm_tokens_total",
    "Tokens LiteLLM reported, by kind (prompt or completion)",
    ["service", "model", "purpose", "kind"],
)
INGESTS = Counter(
    "rag_api_ingests_total",
    "Documents received by /ingest, by outcome (indexed, unchanged, rejected, error)",
    ["service", "collection", "outcome"],
)
INGEST_CHUNKS = Counter(
    "rag_api_ingest_chunks_total",
    "Chunks written to Qdrant",
    ["service", "collection"],
)
INGESTS_IN_PROGRESS = Gauge("rag_api_ingests_in_progress", "Ingests currently running")
INGESTS_WAITING = Gauge(
    "rag_api_ingests_waiting", "Ingests waiting for a slot (INGEST_MAX_CONCURRENCY)"
)
INGESTS_WAITING_FOR_DOCUMENT = Gauge(
    "rag_api_ingests_waiting_for_document",
    "Ingests waiting for an earlier ingest of the same document to finish",
)
UPSTREAM_ERRORS = Counter(
    "rag_api_upstream_errors_total",
    "Failed calls to a dependency, by kind (unavailable, timeout, rate_limited, rejected, error)",
    ["dependency", "kind"],
)
RERANKS = Counter(
    "rag_api_reranks_total",
    "Reranker outcomes (applied, off, unavailable, busy, timeout, error)",
    ["outcome"],
)
COLLECTION_INFO = Gauge(
    "rag_api_collection_info",
    "1 for the physical collection each logical collection is served from",
    ["collection", "physical", "mode"],
)


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Time one pipeline stage."""
    started = time.perf_counter()
    try:
        yield
    finally:
        STAGE_DURATION.labels(stage=name).observe(time.perf_counter() - started)


def record_usage(service: str, model: str, purpose: str, usage: dict[str, Any] | None) -> None:
    """Count the tokens of one LLM answer (LiteLLM's OpenAI-style `usage`)."""
    if not usage:
        return
    for kind, key in (("prompt", "prompt_tokens"), ("completion", "completion_tokens")):
        value = usage.get(key)
        if isinstance(value, int | float) and value > 0:
            LLM_TOKENS.labels(service=service, model=model, purpose=purpose, kind=kind).inc(value)


_server: Any = None


def start_metrics_server(port: int) -> None:
    """Start the /metrics listener once per process."""
    global _server
    if _server is not None:
        return
    result = start_http_server(port, addr="0.0.0.0")
    _server = result[0] if isinstance(result, tuple) else result
    emit(logger, "metrics_listening", port=port)


def stop_metrics_server() -> None:
    global _server
    if _server is not None and hasattr(_server, "shutdown"):
        _server.shutdown()
    _server = None
