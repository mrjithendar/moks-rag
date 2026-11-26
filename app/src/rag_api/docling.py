"""Converts binary/rich documents (PDF, DOCX, HTML, ...) to Markdown via docling-serve's async API.

rag-api submits the file (/v1/convert/file/async), long-polls the task
(/v1/status/poll/{id}?wait=...) and then fetches the result (/v1/result/{id}). No single HTTP
request waits for a whole conversion, so a large PDF is not cut off by rag-api's HTTP timeout.
DOCLING_TIMEOUT_SECONDS bounds the whole conversion (empty = no limit).
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from rag_api import metrics
from rag_api.config import Settings
from rag_api.errors import DocumentConversionFailed, UpstreamError, upstream

DEPENDENCY = "docling-serve"
# Seconds one status request waits for the task to finish (server-side long poll); not a limit on
# the task.
_POLL_WAIT_SECONDS = 5.0

_PLAIN_TEXT_EXTENSIONS = frozenset(
    {".txt", ".md", ".mdx", ".rst", ".csv", ".json", ".yaml", ".yml"}
)
_PLAIN_TEXT_MIMES = frozenset(
    {"text/plain", "text/markdown", "text/x-markdown", "application/json"}
)
_DONE = frozenset({"success", "partial_success"})
_FAILED = frozenset({"failure", "skipped"})


def _extract_markdown(payload: dict[str, Any]) -> str | None:
    document = payload.get("document")
    if isinstance(document, dict):
        for key in ("md_content", "text_content"):
            value = document.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    documents = payload.get("documents")
    if isinstance(documents, list) and documents and isinstance(documents[0], dict):
        for key in ("md_content", "text_content"):
            value = documents[0].get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


class DoclingClient:
    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self._base_url = settings.docling_url.rstrip("/")
        self._timeout = settings.docling_timeout_seconds
        self._client = client

    @staticmethod
    def needs_docling(filename: str | None, content_type: str | None) -> bool:
        ext = os.path.splitext((filename or "").lower())[1]
        mime = (content_type or "").split(";")[0].strip().lower()
        return not (ext in _PLAIN_TEXT_EXTENSIONS or mime in _PLAIN_TEXT_MIMES)

    async def parse_file(
        self,
        filename: str,
        content: bytes,
        content_type: str | None = None,
    ) -> str:
        """Convert one file to Markdown text."""
        started = time.monotonic()
        files = {
            "files": (filename or "document", content, content_type or "application/octet-stream")
        }
        with metrics.stage("docling"), upstream(DEPENDENCY):
            resp = await self._client.post(
                f"{self._base_url}/v1/convert/file/async",
                files=files,
                data={"to_formats": "md"},
            )
            resp.raise_for_status()
            task_id = resp.json().get("task_id")
            if not isinstance(task_id, str) or not task_id:
                raise UpstreamError(DEPENDENCY, "accepted the file but returned no task_id")

            while True:
                if self._timeout is not None and time.monotonic() - started > self._timeout:
                    raise UpstreamError(
                        DEPENDENCY,
                        f"conversion did not finish within {self._timeout:g} s",
                        kind="timeout",
                    )
                poll = await self._client.get(
                    f"{self._base_url}/v1/status/poll/{task_id}",
                    params={"wait": _POLL_WAIT_SECONDS},
                )
                poll.raise_for_status()
                status = poll.json()
                task_status = str(status.get("task_status") or "")
                if task_status in _DONE:
                    break
                if task_status in _FAILED:
                    reason = status.get("error_message") or task_status
                    raise DocumentConversionFailed(
                        f"docling-serve could not convert the file: {reason}"
                    )

            result = await self._client.get(f"{self._base_url}/v1/result/{task_id}")
            result.raise_for_status()
            payload = result.json()

        markdown = _extract_markdown(payload) if isinstance(payload, dict) else None
        if not markdown:
            raise DocumentConversionFailed("docling-serve returned no text for the file")
        return markdown
