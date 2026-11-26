"""Pydantic request and response models for rag-api."""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from rag_api.config import CollectionName


class IngestJSONRequest(BaseModel):
    """JSON payload for text ingestion."""

    collection: CollectionName
    text: str = Field(min_length=1, description="Raw or markdown text to chunk and index")
    doc_id: str | None = Field(
        default=None,
        description="Stable document identifier used for idempotent updates and DELETE sync",
    )
    title: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunk_size: int | None = Field(default=None, ge=100, le=16000)
    chunk_overlap: int | None = Field(default=None, ge=0, le=4000)


class IngestResponse(BaseModel):
    """Response returned after chunking, embedding, and upserting a document."""

    doc_id: str
    collection: CollectionName
    chunks_ingested: int
    parsed_via_docling: bool
    service: str


class DeleteDocumentResponse(BaseModel):
    """Response returned after deleting chunks by doc_id."""

    doc_id: str
    collections: list[str]
    deleted: bool
    service: str


class QueryRequest(BaseModel):
    """Hybrid search + grounded LLM generation request."""

    collection: CollectionName
    query: str = Field(min_length=1)
    top_k: int = Field(default=6, ge=1, le=50)
    model: str | None = Field(
        default=None,
        description="LiteLLM model override (e.g. claude-sonnet-5-5 or claude-haiku-5-5)",
    )
    filters: dict[str, str | int | float | bool] | None = Field(
        default=None,
        description="Exact-match payload metadata filters",
    )
    system_prompt: str | None = None
    include_answer: bool = Field(
        default=True,
        description="Set false to return hybrid-ranked chunks without calling LiteLLM",
    )
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=8192)


class RetrievedChunk(BaseModel):
    """Single retrieved chunk returned from Qdrant hybrid RRF fusion."""

    id: str
    doc_id: str
    score: float
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    """Grounded RAG response with ranked source citations."""

    query: str
    collection: CollectionName
    model: str
    answer: str | None
    sources: list[RetrievedChunk]
    service: str


# --- OpenAI-compatible schemas for Open WebUI ---


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "moks-rag"
    description: str | None = None
    collection: str | None = None


class ModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]]

    def text_content(self) -> str:
        """Flatten string or OpenAI multi-part content into plain text."""
        if isinstance(self.content, str):
            return self.content
        parts: list[str] = []
        for item in self.content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(p for p in parts if p)


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=1024, ge=1, le=8192)
    stream: bool = False
    top_k: int | None = Field(default=None, ge=1, le=50)
    filters: dict[str, str | int | float | bool] | None = None


class ChatCompletionMessageOut(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionMessageOut
    finish_reason: str | None = "stop"


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage = Field(default_factory=ChatCompletionUsage)
