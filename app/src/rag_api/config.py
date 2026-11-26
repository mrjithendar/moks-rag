"""Application configuration, access policy shape and collection/virtual-model definitions.

Every limit here is empty (None) by default, meaning "no limit": the owner sets the numbers in the
deployment, the code never picks them.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

CollectionName = Literal["sre_knowledge", "cms_content", "seo_intel", "github_repos"]

COLLECTIONS: tuple[CollectionName, ...] = (
    "sre_knowledge",
    "cms_content",
    "seo_intel",
    "github_repos",
)

# Version of everything that decides what a stored point looks like: the chunker, the sparse
# encoder, the text that is embedded and the payload layout. Each version lives in its own physical
# Qdrant collection `<collection>__<version>`; the alias `<collection>` points at the one being
# served (store.py, README "Index versions"). Bump it when a change makes old points incompatible
# with new queries, then rebuild. v1: the unversioned collections of 0.1.0 (character chunks,
# ASCII-only unstemmed sparse terms). v2: Markdown-section chunks with "title › heading" embedded,
# Unicode + stemmed sparse terms, the source document stored with chunk 0 so rag-api can rebuild an
# index version by itself.
INDEX_VERSION = "v2"

VIRTUAL_MODELS: dict[str, dict[str, str]] = {
    "moks-sre": {
        "collection": "sre_knowledge",
        "description": "MOKS SRE Knowledge Base (runbooks, incidents, k3s/Istio operations)",
        "llm_model": "claude-sonnet-5-5",
        "system_prompt": (
            "You are the MOKS SRE Assistant. Answer using the retrieved SRE knowledge documents. "
            "If the documents do not contain enough information, state that clearly instead of "
            "guessing."
        ),
    },
    "moks-cms": {
        "collection": "cms_content",
        "description": "MOKS CMS Content Knowledge Base (articles, guides, site content)",
        "llm_model": "claude-sonnet-5-5",
        "system_prompt": (
            "You are the MOKS CMS Assistant. Answer grounded in the retrieved CMS content and cite "
            "source titles/doc_ids where relevant. If the documents do not cover the question, say "
            "so "
            "explicitly."
        ),
    },
    "moks-seo-agent": {
        "collection": "seo_intel",
        "description": "MOKS SEO Intelligence Knowledge Base (audits, SERP, keyword intelligence)",
        "llm_model": "claude-sonnet-5-5",
        "system_prompt": (
            "You are the MOKS SEO Intelligence Assistant. Provide concrete, evidence-based "
            "recommendations grounded in the retrieved SEO intelligence documents, citing source "
            "doc_ids/titles."
        ),
    },
    "mrjithendar-repos": {
        "collection": "github_repos",
        "description": (
            "The mrjithendar GitHub repositories' documentation (READMEs, docs, plans; "
            "rag_api.repo_index)"
        ),
        "llm_model": "claude-sonnet-5-5",
        "system_prompt": (
            "You are the mrjithendar repositories assistant. Answer from the retrieved "
            "repository documents only. Name the repository and file each fact comes from, and "
            "link it with the document's url when it has one. The documents are the default "
            "branch as of the last nightly index, so say when something may have changed since. "
            "If the documents do not "
            "cover the question, say so instead of guessing."
        ),
    },
}


class AccessRule(BaseModel):
    """What one service may do. Collections are names from COLLECTIONS, or "*" for every
    collection.
    """

    read: list[str] = Field(default_factory=list)
    write: list[str] = Field(default_factory=list)
    admin: bool = False


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables (an empty variable counts as unset)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,
    )

    app_name: str = "moks-rag-api"
    root_path: str = Field(default="/rag", description="FastAPI root_path behind Istio prefix")
    log_level: str = Field(default="INFO")

    # --- Authentication and access (auth.py) ---
    rag_bearer_token: str = Field(
        default="",
        description="The shared token of 0.1.0: full access to everything while it is enabled",
    )
    rag_shared_token_enabled: bool = Field(
        default=True,
        description="Accept RAG_BEARER_TOKEN. Turn off once every caller has its own token.",
    )
    rag_service_tokens: dict[str, str] = Field(
        default_factory=dict,
        description='Per-service tokens, JSON {"<service>": "<token>"} (Secret rag-secrets)',
    )
    rag_access_policy: dict[str, AccessRule] = Field(
        default_factory=dict,
        description="What each service may read/write, JSON (ConfigMap rag-api-access)",
    )

    # --- Internal cluster service URLs (*.rag.svc.cluster.local) ---
    qdrant_url: str = Field(default="http://qdrant.rag.svc.cluster.local:6333")
    qdrant_api_key: str = Field(default="")
    index_version: str = Field(default=INDEX_VERSION, pattern=r"^[a-z0-9][a-z0-9_-]{0,31}$")

    bge_m3_url: str = Field(default="http://bge-m3.rag.svc.cluster.local:8001")
    bge_m3_model: str = Field(default="bge-m3:567m")
    # "ollama" calls Ollama's native /api/embed with truncate=false for documents, so a chunk longer
    # than the model's context is refused instead of being embedded from its first part only.
    # "openai" (/v1/embeddings) cannot tell when Ollama truncates.
    bge_m3_api_style: Literal["openai", "ollama"] = Field(default="ollama")
    embedding_dim: int = Field(default=1024, ge=1)
    embedding_batch_size: int = Field(default=32, ge=1, le=256)

    docling_url: str = Field(default="http://docling-serve.rag.svc.cluster.local:5001")
    docling_timeout_seconds: float | None = Field(
        default=None,
        gt=0,
        description="How long one conversion may take in docling-serve (empty = no limit)",
    )

    litellm_url: str = Field(default="http://litellm.rag.svc.cluster.local:4000/v1")
    litellm_api_key: str = Field(default="")
    default_llm_model: str = Field(default="claude-sonnet-5-5")
    fast_llm_model: str = Field(default="claude-haiku-5-5")

    # --- Chunking and retrieval ---
    chunk_size: int = Field(default=1500, ge=100, le=16000)
    chunk_overlap: int = Field(default=200, ge=0, le=4000)
    default_top_k: int = Field(default=6, ge=1, le=50)
    prefetch_limit: int = Field(default=24, ge=1, le=200)
    default_min_dense_score: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Cosine floor for chat retrieval when the request has none (empty = no floor)",
    )
    http_timeout_seconds: float = Field(default=120.0, ge=5.0, le=600.0)
    ensure_collections_on_startup: bool = Field(default=True)

    # --- Chat (OpenAI-compatible endpoint for Open WebUI) ---
    condense_followups: bool = Field(
        default=True,
        description="Rewrite a follow-up into a standalone search query with FAST_LLM_MODEL",
    )
    max_history_messages: int | None = Field(
        default=None,
        ge=1,
        description="Earlier chat messages forwarded to the model (empty = all of them)",
    )

    # --- Ingestion limits ---
    ingest_max_concurrency: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Ingests (parse + embed + write) running at once; queries never wait (empty = no limit)"
        ),
    )
    max_upload_bytes: int | None = Field(
        default=None,
        ge=1,
        description="Largest /ingest request body accepted (empty = no limit)",
    )

    # --- Reranker (rerank.py), off by default ---
    reranker_enabled: bool = Field(default=False)
    reranker_dir: str = Field(default="", description="Directory with model.onnx + tokenizer.json")
    reranker_candidates: int | None = Field(
        default=None,
        ge=1,
        le=200,
        description="Fused results the reranker reads (empty = top_k, so it only reorders them)",
    )
    reranker_timeout_ms: int | None = Field(default=None, ge=1)
    reranker_max_concurrent: int | None = Field(default=None, ge=1)
    reranker_threads: int | None = Field(
        default=None, ge=1, description="Empty = ONNX Runtime default"
    )

    # --- Observability ---
    metrics_port: int | None = Field(
        default=None,
        ge=1,
        le=65535,
        description="Port of the /metrics listener (empty = off); the edge never routes it",
    )

    def allowed_llm_models(self) -> set[str]:
        """Models a /query request may name: the ones LiteLLM is configured with."""
        return {self.default_llm_model, self.fast_llm_model}


@lru_cache
def get_settings() -> Settings:
    """Return cached runtime settings."""
    return Settings()
