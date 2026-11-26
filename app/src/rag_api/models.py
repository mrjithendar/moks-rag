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
    doc_id: str = Field(
        min_length=1,
        description=(
            "Stable document identifier: sending the same doc_id again replaces the document, "
            "and DELETE /documents uses it"
        ),
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
    unchanged: bool = Field(
        default=False,
        description="True when the stored version was identical: nothing was embedded or written",
    )
    index_version: str = ""


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
    min_dense_score: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description=(
            "Drop chunks whose dense cosine similarity to the query is below this (empty = no "
            "floor). "
            "`score` is a rank-fusion score and cannot be compared with a fixed number; "
            "`dense_score` can."
        ),
    )
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
    score: float = Field(
        description="Reciprocal Rank Fusion score: orders one answer's chunks only"
    )
    dense_score: float | None = Field(
        default=None, description="Cosine similarity of the chunk's dense vector to the query"
    )
    rerank_score: float | None = Field(
        default=None, description="Cross-encoder relevance logit, when the reranker ran"
    )
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
    reranker: str = Field(
        default="off", description="applied, off, unavailable, busy, timeout or error"
    )


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
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None = None

    def text_content(self) -> str:
        """Flatten string or OpenAI multi-part content into plain text."""
        if self.content is None:
            return ""
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
    min_dense_score: float | None = Field(default=None, ge=-1.0, le=1.0)


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


# --- Admin ---


class CollectionStatus(BaseModel):
    collection: str
    mode: Literal["alias", "legacy"]
    serving: str = Field(description="Physical collection searches read")
    target: str = Field(description="Physical collection of this rag-api's INDEX_VERSION")
    points_serving: int
    points_target: int
    documents_without_source: int = Field(
        description="0.1.0 documents in the served collection that a rebuild cannot copy"
    )
    rebuild: dict[str, Any] | None = None


class CollectionsStatusResponse(BaseModel):
    index_version: str
    collections: list[CollectionStatus]
    physical_collections: list[str]
