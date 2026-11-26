"""Application configuration and collection/virtual-model definitions."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

CollectionName = Literal["sre_knowledge", "cms_content", "seo_intel"]

COLLECTIONS: tuple[CollectionName, ...] = (
    "sre_knowledge",
    "cms_content",
    "seo_intel",
)


class VirtualModelConfig(dict[str, str]):
    """Typed dictionary shape for virtual OpenAI models exposed to Open WebUI."""


VIRTUAL_MODELS: dict[str, dict[str, str]] = {
    "moks-sre": {
        "collection": "sre_knowledge",
        "description": "MOKS SRE Knowledge Base (runbooks, incidents, k3s/Istio operations)",
        "llm_model": "claude-sonnet-5-5",
        "system_prompt": (
            "You are the MOKS SRE Assistant. Answer using the retrieved SRE knowledge context "
            "below. Cite source doc_id or title inline (e.g. [source: <doc_id>]). If the context "
            "does not contain enough information, state that clearly instead of guessing."
        ),
    },
    "moks-cms": {
        "collection": "cms_content",
        "description": "MOKS CMS Content Knowledge Base (articles, guides, site content)",
        "llm_model": "claude-sonnet-5-5",
        "system_prompt": (
            "You are the MOKS CMS Assistant. Answer grounded in the retrieved CMS content below "
            "and cite source titles/doc_ids where relevant. If the retrieved snippets do not cover "
            "the question, say so explicitly."
        ),
    },
    "moks-seo-agent": {
        "collection": "seo_intel",
        "description": "MOKS SEO Intelligence Knowledge Base (audits, SERP, keyword intelligence)",
        "llm_model": "claude-sonnet-5-5",
        "system_prompt": (
            "You are the MOKS SEO Intelligence Assistant. Provide concrete, evidence-based "
            "recommendations grounded in the retrieved SEO intelligence snippets below, citing "
            "source doc_ids/titles."
        ),
    },
}


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "moks-rag-api"
    root_path: str = Field(default="/rag", description="FastAPI root_path behind Istio prefix")
    log_level: str = Field(default="INFO")

    # Authentication
    rag_bearer_token: str = Field(default="", description="Shared Bearer token for rag-api")

    # Internal cluster service URLs (*.rag.svc.cluster.local)
    qdrant_url: str = Field(default="http://qdrant.rag.svc.cluster.local:6333")
    qdrant_api_key: str = Field(default="")

    bge_m3_url: str = Field(default="http://bge-m3.rag.svc.cluster.local:8001")
    bge_m3_model: str = Field(default="bge-m3:567m")
    bge_m3_api_style: Literal["openai", "ollama"] = Field(default="openai")
    embedding_dim: int = Field(default=1024, ge=1)
    embedding_batch_size: int = Field(default=32, ge=1, le=256)
    sparse_mode: Literal["bm25", "bge_m3"] = Field(default="bm25")

    docling_url: str = Field(default="http://docling-serve.rag.svc.cluster.local:5001")

    litellm_url: str = Field(default="http://litellm.rag.svc.cluster.local:4000/v1")
    litellm_api_key: str = Field(default="")
    default_llm_model: str = Field(default="claude-sonnet-5-5")
    fast_llm_model: str = Field(default="claude-haiku-5-5")

    # Chunking and retrieval defaults
    chunk_size: int = Field(default=1500, ge=100, le=16000)
    chunk_overlap: int = Field(default=200, ge=0, le=4000)
    default_top_k: int = Field(default=6, ge=1, le=50)
    prefetch_limit: int = Field(default=24, ge=1, le=200)
    http_timeout_seconds: float = Field(default=120.0, ge=5.0, le=600.0)
    ensure_collections_on_startup: bool = Field(default=True)


@lru_cache
def get_settings() -> Settings:
    """Return cached runtime settings."""
    return Settings()
