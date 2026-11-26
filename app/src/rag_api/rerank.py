"""Optional cross-encoder reranker over the fused candidates. OFF by default (RERANKER_ENABLED).

The model and the gate come from moks-seo-agent (app/reranker.py, PR #31):
cross-encoder/ms-marco-MiniLM-L6-v2 at revision 233902d, its int8 ONNX export, baked into the image
at RERANKER_DIR with a pinned sha256 (Dockerfile). CPU only, ONNX Runtime. seo-agent measured +2.4
to +4.7 Recall@3 on its blog eval set; p95 on GitHub's Linux ARM runners was 2.9 s at 1 thread for
~10-20 candidates, so measure on mk4 (rag_api_stage_duration_seconds{stage= "rerank"}) before
switching it on.

Search never fails because of the reranker: off, model missing, at its concurrency limit, past its
deadline or failing, the search keeps the fused order and says why (QueryResponse.reranker).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rag_api import metrics
from rag_api.config import Settings
from rag_api.logs import emit
from rag_api.models import RetrievedChunk

logger = logging.getLogger("rag_api.rerank")

PASSAGE_CHARS = 2000


def passage_text(chunk: RetrievedChunk) -> str:
    title = chunk.metadata.get("title") or chunk.doc_id
    heading = chunk.metadata.get("heading_path") or ""
    return f"{title} › {heading}\n{chunk.text[:PASSAGE_CHARS]}"


class CrossEncoderReranker:
    MAX_TOKENS = 512

    def __init__(self, model_dir: str, threads: int | None = None) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        d = Path(model_dir)
        self.name = f"onnx:{d.name}"
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tok.enable_truncation(max_length=self.MAX_TOKENS)
        pad = self.tok.token_to_id("[PAD]") or self.tok.token_to_id("<pad>") or 0
        self.tok.enable_padding(pad_id=pad)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads or 0  # 0 = ONNX Runtime's default
        opts.inter_op_num_threads = 1
        opts.enable_cpu_mem_arena = False
        self.session = ort.InferenceSession(
            str(d / "model.onnx"), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.inputs = {i.name for i in self.session.get_inputs()}

    def scores(self, query: str, passages: list[str]) -> list[float]:
        import numpy as np

        enc = self.tok.encode_batch([(query, p) for p in passages])
        feed = {
            "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
        }
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.array([e.type_ids for e in enc], dtype=np.int64)
        out = self.session.run(None, feed)[0]
        return [float(x) for x in np.asarray(out).reshape(len(passages), -1)[:, 0]]


class RerankGate:
    """Applies the reranker to one search, or says why it did not. The model loads on first use; a
    failed load is remembered until restart. A rerank's slot is freed when its computation ends, not
    when the search stops waiting (a native ONNX call cannot be interrupted).
    """

    def __init__(
        self,
        settings: Settings,
        factory: Callable[[str, int | None], Any] = CrossEncoderReranker,
    ) -> None:
        self._settings = settings
        self._factory = factory
        self._model: Any = None
        self.load_error: str | None = None
        self._loading = asyncio.Lock()
        self.running = 0

    @property
    def enabled(self) -> bool:
        return self._settings.reranker_enabled and bool(self._settings.reranker_dir)

    def candidates(self, top_k: int) -> int:
        """How many fused results a search should fetch for the reranker to read."""
        if not self.enabled:
            return top_k
        return max(top_k, self._settings.reranker_candidates or top_k)

    async def _load(self) -> Any:
        if self._model is not None or self.load_error is not None:
            return self._model
        async with self._loading:
            if self._model is None and self.load_error is None:
                try:
                    self._model = await asyncio.to_thread(
                        self._factory, self._settings.reranker_dir, self._settings.reranker_threads
                    )
                    emit(logger, "reranker_loaded", model=self._model.name)
                except Exception as exc:  # noqa: BLE001 - missing files, a broken model: search goes on
                    self.load_error = type(exc).__name__
                    emit(
                        logger, "reranker_load_failed", level=logging.WARNING, error=self.load_error
                    )
        return self._model

    def _done(self, task: asyncio.Future[Any], started: float) -> None:
        self.running -= 1
        metrics.STAGE_DURATION.labels(stage="rerank").observe(time.perf_counter() - started)
        if not task.cancelled() and task.exception() is not None:
            emit(
                logger,
                "reranker_failed",
                level=logging.WARNING,
                error=type(task.exception()).__name__,
            )

    async def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int
    ) -> tuple[list[RetrievedChunk], str]:
        outcome, result = await self._rerank(query, chunks, top_k)
        metrics.RERANKS.labels(outcome=outcome).inc()
        return result, outcome

    async def _rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int
    ) -> tuple[str, list[RetrievedChunk]]:
        if not self.enabled:
            return "off", chunks[:top_k]
        if not chunks:
            return "applied", []
        model = await self._load()
        if model is None:
            return "unavailable", chunks[:top_k]
        limit = self._settings.reranker_max_concurrent
        if limit is not None and self.running >= limit:
            return "busy", chunks[:top_k]
        self.running += 1
        started = time.perf_counter()
        task = asyncio.ensure_future(
            asyncio.to_thread(model.scores, query, [passage_text(c) for c in chunks])
        )
        task.add_done_callback(lambda t: self._done(t, started))
        timeout_ms = self._settings.reranker_timeout_ms
        try:
            scores = await asyncio.wait_for(
                asyncio.shield(task), timeout_ms / 1000 if timeout_ms else None
            )
        except TimeoutError:
            return "timeout", chunks[:top_k]
        except Exception:  # noqa: BLE001
            return "error", chunks[:top_k]
        if len(scores) != len(chunks):
            return "error", chunks[:top_k]
        order = sorted(range(len(chunks)), key=lambda i: -scores[i])
        ranked = [
            chunks[i].model_copy(update={"rerank_score": round(scores[i], 6)})
            for i in order[:top_k]
        ]
        return "applied", ranked
