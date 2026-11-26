# syntax=docker/dockerfile:1.7

FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip wheel --wheel-dir /wheels -r requirements.txt

# The optional reranker (src/rag_api/rerank.py, off until RERANKER_ENABLED=true): the same model and pins as
# moks-seo-agent's (Dockerfile there): cross-encoder/ms-marco-MiniLM-L6-v2 (Apache-2.0) at a pinned revision,
# its int8 ONNX export (23 MB), and its tokenizer.json, both checked by sha256 so the image is reproducible and
# pods never download anything at start-up.
ARG RERANKER_REPO=cross-encoder/ms-marco-MiniLM-L6-v2
ARG RERANKER_REV=233902d25c440f23af6f7d6e94d2946bac0bee0a
ARG RERANKER_FILE=onnx/model_qint8_arm64.onnx
ARG RERANKER_SHA256=3573b6b9593cb2f75987a31815d409ca3dd8808629118fd20451bb1a5d90cec7
ARG RERANKER_TOKENIZER_SHA256=d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66
RUN mkdir -p /models/ms-marco-MiniLM-L6-v2 \
 && python -c "import urllib.request as u; base='https://huggingface.co/${RERANKER_REPO}/resolve/${RERANKER_REV}'; u.urlretrieve(f'{base}/${RERANKER_FILE}', '/models/ms-marco-MiniLM-L6-v2/model.onnx'); u.urlretrieve(f'{base}/tokenizer.json', '/models/ms-marco-MiniLM-L6-v2/tokenizer.json')" \
 && echo "${RERANKER_SHA256}  /models/ms-marco-MiniLM-L6-v2/model.onnx" | sha256sum -c - \
 && echo "${RERANKER_TOKENIZER_SHA256}  /models/ms-marco-MiniLM-L6-v2/tokenizer.json" | sha256sum -c -

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    RERANKER_DIR=/models/ms-marco-MiniLM-L6-v2 \
    TOKENIZERS_PARALLELISM=false

RUN groupadd --gid 1000 rag && \
    useradd --uid 1000 --gid 1000 --create-home --shell /usr/sbin/nologin rag

WORKDIR /app

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/*.whl && \
    rm -rf /wheels
COPY --from=builder /models /models

COPY --chown=rag:rag pyproject.toml ./
COPY --chown=rag:rag src ./src

USER rag

EXPOSE 8000 9464

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "rag_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
