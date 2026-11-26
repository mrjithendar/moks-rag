"""OpenAI-compatible client for the internal LiteLLM proxy.

Streams are opened (and their status checked) before rag-api answers its own caller, so an upstream
401/429/5xx becomes a proper HTTP error instead of a 200 followed by an empty stream. Streams ask
for usage (stream_options.include_usage) so token counts are known for streamed answers too.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from rag_api.config import Settings
from rag_api.errors import from_response, upstream

DEPENDENCY = "litellm"
DONE = "[DONE]"


class LLMStream:
    """An open streaming answer: iterate events() for parsed chunks (dicts) and the final DONE
    marker.
    """

    def __init__(self, response: httpx.Response) -> None:
        self._response = response
        self.usage: dict[str, Any] | None = None
        self.finish_reason: str | None = None

    async def events(self) -> AsyncIterator[dict[str, Any] | str]:
        try:
            with upstream(DEPENDENCY):
                async for line in self._response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    raw = line[len("data:") :].strip()
                    if raw == DONE:
                        yield DONE
                        continue
                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(event, dict):
                        continue
                    if isinstance(event.get("usage"), dict):
                        self.usage = event["usage"]
                    for choice in event.get("choices") or []:
                        if isinstance(choice, dict) and choice.get("finish_reason"):
                            self.finish_reason = str(choice["finish_reason"])
                    yield event
        finally:
            await self.aclose()

    async def aclose(self) -> None:
        await self._response.aclose()


class LiteLLMClient:
    """OpenAI-compatible client targeting internal LiteLLM proxy."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self._base_url = settings.litellm_url.rstrip("/")
        self._api_key = settings.litellm_api_key
        self._client = client

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    @staticmethod
    def _payload(
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int | None,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        return payload

    async def chat_completions(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int | None = 1024,
    ) -> dict[str, Any]:
        with upstream(DEPENDENCY):
            resp = await self._client.post(
                f"{self._base_url}/chat/completions",
                headers=self._headers(),
                json=self._payload(model, messages, temperature, max_tokens, stream=False),
            )
            resp.raise_for_status()
        return resp.json()

    async def open_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int | None = 1024,
    ) -> LLMStream:
        """Open a streaming completion; raises UpstreamError before any byte goes to the caller."""
        request = self._client.build_request(
            "POST",
            f"{self._base_url}/chat/completions",
            headers=self._headers(),
            json=self._payload(model, messages, temperature, max_tokens, stream=True),
        )
        with upstream(DEPENDENCY):
            resp = await self._client.send(request, stream=True)
        if resp.status_code >= 400:
            try:
                body = (await resp.aread()).decode("utf-8", errors="replace")[:300]
            finally:
                await resp.aclose()
            raise from_response(DEPENDENCY, resp, body)
        return LLMStream(resp)
