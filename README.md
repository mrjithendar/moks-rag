# `moks-rag` — Hybrid RAG Platform for Hetzner k3s

Production, CPU-only Hybrid RAG platform deployed in the `rag` namespace on Hetzner k3s (ARM64 CAX nodes) for `jithendar.com`, managed via GitHub Actions, ArgoCD, and Istio (`STRICT` mTLS).

## Architecture

All components run in the `rag` namespace and communicate over internal Kubernetes DNS (`*.rag.svc.cluster.local`). Only `rag-api` (`https://jithendar.com/rag`), `open-webui` (`https://jithendar.com/chat`) and Qdrant's REST port with its Web UI (`https://jithendar.com/qdrant`, behind Qdrant's API key, see [Browsing Qdrant](#browsing-qdrant)) are exposed through the Istio ingress gateway (`istio-ingress/public`); every other backend service is strictly `ClusterIP`-only.

| Component | Image / Chart | Internal Endpoint | Purpose |
| --- | --- | --- | --- |
| `bge-m3` | `ollama/ollama:0.40.16` (`bge-m3:567m`) | `http://bge-m3.rag.svc.cluster.local:8001` (and `:11434`) | 1024-dim dense embeddings on ARM64 CPU (migrated from `embeddings` namespace) |
| `qdrant` | Helm `qdrant/qdrant` chart `1.20.2` (ArgoCD multi-source) | `http://qdrant.rag.svc.cluster.local:6333` | Vector store (`15Gi` PVC; 3 aliases over versioned dense+sparse collections, see [Index versions](#index-versions)) |
| `docling-serve` | `ghcr.io/docling-project/docling-serve-cpu` 1.39.0, pinned by digest | `http://docling-serve.rag.svc.cluster.local:5001` | Document parsing (PDF/DOCX/HTML -> Markdown via its async API, 2–4Gi RAM) |
| `litellm` | `ghcr.io/berriai/litellm`, pinned by digest | `http://litellm.rag.svc.cluster.local:4000/v1` | OpenAI-compatible proxy routing `claude-sonnet-5-5` & `claude-haiku-5-5` |
| `rag-api` | `ghcr.io/mrjithendar/moks-rag-api:<sha>` | `http://rag-api.rag.svc.cluster.local:8000` (metrics `:9464`) | Custom FastAPI orchestrator (`root_path="/rag"`) |
| `open-webui` | `ghcr.io/open-webui/open-webui`, pinned by digest | `http://open-webui.rag.svc.cluster.local:80` | Chat UI (`2Gi` PVC) pointing at `rag-api` `/v1` |

### Collections & Virtual Models

| Open WebUI Virtual Model | Qdrant Collection | Vectors | Intended Consumer |
| --- | --- | --- | --- |
| `moks-sre` | `sre_knowledge` | `dense` (1024 cosine) + `sparse` (BM25/IDF) | `moks-sre` runbooks & incidents |
| `moks-cms` | `cms_content` | `dense` (1024 cosine) + `sparse` (BM25/IDF) | `moks-cms` articles & webhook sync |
| `moks-seo-agent` | `seo_intel` | `dense` (1024 cosine) + `sparse` (BM25/IDF) | `moks-seo-agent` audits & SERP intel |

---

## Repository Layout

```text
moks-rag/
├── app/                                 # FastAPI rag-api service, Dockerfile, pytest suite
│   ├── src/rag_api/
│   │   ├── main.py        # endpoints, error mapping
│   │   ├── auth.py        # per-service tokens, access rules, access log
│   │   ├── store.py       # Qdrant: aliases, index versions, document replacement, hybrid search
│   │   ├── indexer.py     # chunk -> embed -> write, unchanged skip, rebuilds
│   │   ├── chunking.py    # Markdown-section chunks with heading paths
│   │   ├── sparse.py      # Unicode + stemmed BM25-style sparse terms
│   │   ├── embeddings.py  # bge-m3 (Ollama /api/embed)
│   │   ├── docling.py     # docling-serve async conversion
│   │   ├── llm.py         # LiteLLM, streams checked before answering
│   │   ├── prompts.py     # fixed system prompt, documents as data in the user turn
│   │   ├── rerank.py      # optional int8 cross-encoder (off)
│   │   ├── metrics.py     # Prometheus
│   │   └── eval.py        # python -m rag_api.eval
│   ├── tests/
│   ├── Dockerfile
│   ├── pyproject.toml
│   └── requirements.txt
├── deploy/
│   ├── k8s/                             # Kustomize stack deployed by ArgoCD (moks-rag-stack)
│   │   ├── kustomization.yaml
│   │   ├── namespace.yaml
│   │   ├── bge-m3.yaml
│   │   ├── docling-serve.yaml
│   │   ├── litellm.yaml
│   │   ├── rag-api.yaml
│   │   ├── open-webui.yaml
│   │   ├── istio-routing.yaml
│   │   ├── rag-secrets.sealed-secret.yaml.template
│   │   └── rag-secrets.external-secret.yaml.template
│   └── argocd/                          # ArgoCD AppProject + Applications + Qdrant Helm values
│       ├── ci-rbac.yaml
│       ├── project.yaml
│       ├── rag-stack-app.yaml
│       ├── qdrant-app.yaml
│       └── qdrant-values.yaml
├── scripts/
│   └── issue-service-token.sh           # give one caller its own token (README "Access")
└── .github/workflows/
    ├── build-and-deploy.yaml            # Lint/test (+ image build on PRs) -> multi-arch GHCR push -> Kustomize tag bump
    └── deploy-argocd-apps.yaml          # Validate & apply ArgoCD AppProject/Applications to k3s
```

---

## Step-by-Step Bootstrap on Hetzner k3s

Set your kubeconfig first:

```bash
export KUBECONFIG=/path/to/kubeconfig-hetzner
```

### 1. Create the `rag` Namespace & `rag-secrets`

Never commit plaintext secrets. Choose **one** option below:

**Option A — Out-of-band `kubectl` Secret (fastest bootstrap):**

```bash
kubectl apply -f deploy/k8s/namespace.yaml

export ANTHROPIC_API_KEY="sk-ant-..."
export QDRANT_API_KEY="$(openssl rand -hex 32)"
export LITELLM_MASTER_KEY="sk-litellm-$(openssl rand -hex 24)"
export RAG_BEARER_TOKEN="$(openssl rand -hex 32)"
export WEBUI_SECRET_KEY="$(openssl rand -hex 32)"

kubectl -n rag create secret generic rag-secrets \
  --from-literal=ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}" \
  --from-literal=QDRANT_API_KEY="${QDRANT_API_KEY}" \
  --from-literal=LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY}" \
  --from-literal=RAG_BEARER_TOKEN="${RAG_BEARER_TOKEN}" \
  --from-literal=WEBUI_SECRET_KEY="${WEBUI_SECRET_KEY}" \
  --dry-run=client -o yaml | kubectl apply -f -
```

Two optional keys are added later by `scripts/issue-service-token.sh` (see [Access](#access-per-service-tokens)):
`RAG_SERVICE_TOKENS` (JSON, one token per calling service) and `OPEN_WEBUI_RAG_TOKEN`.

**Option B — GitOps with SealedSecrets / External Secrets Operator:**

- Use `deploy/k8s/rag-secrets.sealed-secret.yaml.template` with `kubeseal`, or `deploy/k8s/rag-secrets.external-secret.yaml.template` with Vault ESO, then add the generated file to `deploy/k8s/kustomization.yaml`.

### 2. Wire `jithendar.com/rag` and `jithendar.com/chat` in the Edge Gateway

In your cluster edge chart (`k8s-deployments/deploy/minikube/istio/edge/values.yaml`), add three delegates under `root.delegates` so the shared `istio-ingress/jithendar-com` root VirtualService delegates `/rag`, `/chat` and `/qdrant`:

```yaml
root:
  delegates:
    - {name: rag, prefix: /rag, to: rag/rag-api}     # <-- add
    - {name: chat, prefix: /chat, to: rag/open-webui} # <-- add
    - {name: qdrant, prefix: /qdrant, to: rag/qdrant} # <-- add
```

> **Open WebUI under `/chat`:** its prebuilt SvelteKit frontend assumes `/`. `deploy/k8s/open-webui.yaml`'s `patch_subpath.py` rewrites it at start-up and then verifies the result: if a new Open WebUI build no longer matches, the container refuses to start (visible in Argo CD) instead of serving a `/chat` that loads nothing. The image is pinned by digest for that reason.

### 3. Apply ArgoCD `AppProject` and `Application`s (Manual or via GitHub Actions)

**Option A — Automated via `.github/workflows/deploy-argocd-apps.yaml`:**

Configure GitHub Actions authentication using either a least-privilege ServiceAccount token (recommended) or a base64 kubeconfig:

```bash
# Option A1: Scoped ServiceAccount in argocd namespace (recommended)
kubectl apply -f deploy/argocd/ci-rbac.yaml
kubectl -n argocd get secret github-rag-argocd-deployer-token \
  -o jsonpath='{.data.token}' | base64 -d | gh secret set K8S_ARGOCD_TOKEN -R mrjithendar/moks-rag
gh variable set K8S_API_SERVER -R mrjithendar/moks-rag \
  --body "$(kubectl config view --raw --minify -o jsonpath='{.clusters[0].cluster.server}')"
gh variable set K8S_CA_DATA -R mrjithendar/moks-rag \
  --body "$(kubectl config view --raw --minify -o jsonpath='{.clusters[0].cluster.certificate-authority-data}')"

# Option A2: Full kubeconfig secret
base64 -i kubeconfig-hetzner | tr -d '\n' | gh secret set KUBECONFIG_B64 -R mrjithendar/moks-rag
```

Pushing changes under `deploy/argocd/**` to `main` (or triggering `Deploy ArgoCD Applications` via `workflow_dispatch`) validates and applies `project.yaml`, `qdrant-app.yaml`, and `rag-stack-app.yaml`, refreshes ArgoCD, and checks `Synced`/`Healthy` status.

**Option B — Manual `kubectl apply`:**

```bash
kubectl apply -f deploy/argocd/project.yaml
kubectl apply -f deploy/argocd/qdrant-app.yaml
kubectl apply -f deploy/argocd/rag-stack-app.yaml
```

Watch both ArgoCD Applications sync and all pods become `Ready`:

```bash
kubectl -n argocd get applications moks-rag-qdrant moks-rag-stack
kubectl -n rag get pods,svc,pvc -w
```

*(First boot note: `bge-m3`'s `fetch-model` initContainer downloads the 1.16 GB `bge-m3:567m` weights once onto its `5Gi` `hcloud-volumes` PVC and verifies the pinned SHA-256 digest.)*

### 4. Migrate `bge-m3` from `embeddings` Namespace & Clean Up

Follow this safe zero-downtime cutover checklist:

1. **Verify `bge-m3` in `rag` is healthy and returning 1024-dim vectors:**
   ```bash
   kubectl -n rag logs deploy/bge-m3 -c fetch-model
   kubectl -n rag rollout status deploy/bge-m3
   kubectl -n rag port-forward svc/bge-m3 8001:8001 &
   PF_PID=$!
   curl -s http://127.0.0.1:8001/v1/embeddings \
     -H "Content-Type: application/json" \
     -d '{"model":"bge-m3:567m","input":["health check"]}' \
     | python3 -c 'import json,sys; v=json.load(sys.stdin)["data"][0]["embedding"]; print("dim:", len(v))'
   kill $PF_PID
   ```
2. **Update existing cluster callers** (`moks-sre` and `jithendar/seo-agent`) from `http://bge-m3.embeddings:11434` to `http://bge-m3.rag.svc.cluster.local:11434` (or `:8001`) and update their egress `NetworkPolicy` from namespace `embeddings` to `rag`.
3. **Confirm zero remaining traffic to `embeddings/bge-m3`:**
   ```bash
   kubectl -n embeddings logs deploy/bge-m3 -c ollama --since=15m
   ```
4. **Remove the legacy `embeddings` Helm release (if present) and namespace:**
   ```bash
   helm -n embeddings uninstall bge-m3 || true
   kubectl -n embeddings delete pvc bge-m3-models --ignore-not-found
   kubectl delete namespace embeddings
   ```
   *(If Terraform in `k8s-deployments/minikube-infrastructure` previously managed `embeddings`, remove or update that release so Terraform does not recreate it.)*

### 5. Verify External Endpoints & Hybrid RAG Pipeline

**Health check (public edge, no auth required):**

```bash
curl -fsS https://jithendar.com/rag/health
```

**List OpenAI-compatible virtual models:**

```bash
curl -fsS https://jithendar.com/rag/v1/models \
  -H "Authorization: Bearer ${RAG_BEARER_TOKEN}" \
  -H "X-Service-Name: smoke-test"
```

**Ingest a document, query with hybrid RRF + Claude, and delete by `doc_id`:**

```bash
# 1) Ingest JSON text
curl -fsS -X POST https://jithendar.com/rag/ingest \
  -H "Authorization: Bearer ${RAG_BEARER_TOKEN}" \
  -H "X-Service-Name: moks-cms" \
  -H "Content-Type: application/json" \
  -d '{
    "collection": "cms_content",
    "doc_id": "smoke-doc-1",
    "title": "Hetzner k3s RAG Smoke Test",
    "text": "moks-rag runs bge-m3, Qdrant hybrid RRF, docling-serve, LiteLLM, and Open WebUI in namespace rag."
  }'

# 2) Hybrid query (dense + sparse RRF -> LiteLLM)
curl -fsS -X POST https://jithendar.com/rag/query \
  -H "Authorization: Bearer ${RAG_BEARER_TOKEN}" \
  -H "X-Service-Name: moks-cms" \
  -H "Content-Type: application/json" \
  -d '{
    "collection": "cms_content",
    "query": "What components run in the rag namespace?",
    "top_k": 4
  }'

# 3) Delete document (moks-cms webhook sync)
curl -fsS -X DELETE "https://jithendar.com/rag/documents/smoke-doc-1?collection=cms_content" \
  -H "Authorization: Bearer ${RAG_BEARER_TOKEN}" \
  -H "X-Service-Name: moks-cms"
```

**Verify Open WebUI:**

```bash
curl -fsSI https://jithendar.com/chat/health
```

Open `https://jithendar.com/chat/` in your browser, sign in (the first registered admin account is created on initial setup; `ENABLE_SIGNUP=false` prevents further public signups), and select **`moks-sre`**, **`moks-cms`**, or **`moks-seo-agent`** from the model selector.

---

## Browsing Qdrant

Qdrant's own Web UI is at **https://jithendar.com/qdrant/dashboard** (`/qdrant` redirects there). It lists the collections (the `__v2` physical collections; `cms_content`, `seo_intel` and `sre_knowledge` are aliases of them) and pages through their points: one point per chunk, with its payload (`doc_id`, `title`, `heading_path`, `text`, ...; chunk 0 also holds the source document). Filter by a payload field (`doc_id:blog:...`) or look a point up by id.

The UI asks for an API key and keeps it in the browser's localStorage. jithendar.com is one origin for the site and every admin tool, so any script on the site can read that key. Give the UI a **read-only key**, which can browse and search but not change or delete anything:

```bash
kubectl -n rag patch secret rag-secrets --type merge \
  -p "{\"stringData\":{\"QDRANT_READ_ONLY_API_KEY\":\"$(openssl rand -hex 32)\"}}"
kubectl -n rag rollout restart statefulset qdrant     # Qdrant reads its keys at start-up
kubectl -n rag get secret rag-secrets -o jsonpath='{.data.QDRANT_READ_ONLY_API_KEY}' | base64 -d | pbcopy
```

Paste it in the UI (key icon, top right). The full `QDRANT_API_KEY` works too, but then the UI can delete collections.

How `/qdrant` works: the `qdrant` VirtualService in `deploy/k8s/istio-routing.yaml` strips the prefix and sets `Cache-Control: no-store` on everything except the UI's hashed files, so Cloudflare never keeps an API answer; the `qdrant-ingress` AuthorizationPolicy lets the gateway reach port 6333 only. This Qdrant build's UI loads its files from `/dashboard/...` and its API client ignores the path of its URL, so start-up copies the UI and patches it for `/qdrant` (`args` in `deploy/argocd/qdrant-values.yaml`). After a chart upgrade, check Qdrant's log for `dashboard-ui: Web UI patched for /qdrant/dashboard`; a `WARNING` there means only the UI under `/qdrant` broke, and the patch needs updating.

---

## API contract (0.2.0)

| Endpoint | Scope | Notes |
| --- | --- | --- |
| `POST /ingest` | write on `collection` | JSON or multipart (`file` via docling-serve). **`doc_id` is required**: sending the same `doc_id` again replaces the document. The answer has `unchanged: true` when the stored version was identical (nothing embedded or written). A chunk too long for bge-m3's context is refused with 422 instead of being embedded from its start only. |
| `DELETE /documents/{doc_id}?collection=…` | write | **`collection` is required** (`all` = every collection the caller may write). `doc_id` may contain `/`. Also `DELETE /documents?doc_id=…&collection=…`. |
| `POST /query` | read | Each source has `score` (rank fusion: orders one answer only), **`dense_score`** (cosine similarity, comparable across queries) and `rerank_score` when the reranker ran. `min_dense_score` drops chunks below a cosine floor. `filters` match payload fields exactly (not `text` or `document*`; the first filter on a field indexes it). `model` must be `DEFAULT_LLM_MODEL` or `FAST_LLM_MODEL`. |
| `GET /v1/models`, `POST /v1/chat/completions` | read on the model's collection | OpenAI-compatible, for Open WebUI. Follow-ups are rewritten into a standalone search query with `FAST_LLM_MODEL` (`CONDENSE_FOLLOWUPS`). Open WebUI's background tasks (title, tags, follow-ups: messages starting `### Task:`) skip retrieval and use `FAST_LLM_MODEL`. |
| `GET /admin/collections`, `POST /admin/collections/{name}/rebuild`, `…/promote`, `DELETE /admin/physical/{name}` | admin | [Index versions](#index-versions). |

Prompts: the system prompt is fixed per model; retrieved chunks go in the user turn inside `<documents><document doc_id=… title=… section=…>` and the system prompt says they are data, not instructions (retrieved text can come from third-party pages, e.g. `seo_intel`).

Errors name the dependency that failed: `503` unreachable, `504` too slow, `429` rate-limited (with `Retry-After`), `502` anything else, body `{"detail", "dependency", "kind", "request_id"}`. A stream that fails after it started ends with an SSE `{"error": …}` event and `[DONE]`.

## Access: per-service tokens

Every caller can get its own token, limited by the access policy in `deploy/k8s/rag-api.yaml` (ConfigMap `rag-api-access`): `moks-cms` writes `cms_content`; `moks-seo-agent` writes `seo_intel` and reads `cms_content`; `moks-sre-agent` writes `sre_knowledge`; `open-webui` reads everything; `rag-admin` may use `/admin/*`. With its own token, a caller's logs and metrics carry the token's service, whatever `X-Service-Name` says.

The 0.1.0 shared `RAG_BEARER_TOKEN` keeps full access while `RAG_SHARED_TOKEN_ENABLED` is `"true"`, so nothing breaks during the move. To move one caller (no code change in the caller: each one already reads its token from its own Secret):

```bash
scripts/issue-service-token.sh moks-cms         # Secret jithendar/jithendar-com-rag (key token)
scripts/issue-service-token.sh moks-seo-agent   # Secret jithendar/seo-agent-rag (RAG_BEARER_TOKEN)
scripts/issue-service-token.sh moks-sre-agent   # Secret moks-sre/moks-rag-client (RAG_BEARER_TOKEN)
scripts/issue-service-token.sh open-webui       # rag-secrets OPEN_WEBUI_RAG_TOKEN
```

Then restart the caller it names and check rag-api's log lines show `"auth": "service-token"` for it. When no request uses `"auth": "shared-token"` any more, set `RAG_SHARED_TOKEN_ENABLED` to `"false"` in `deploy/k8s/rag-api.yaml`. Open WebUI keeps its connection key in its database once it was saved in Admin → Settings → Connections; then paste the new token there too.

## Index versions

Each logical collection is a Qdrant **alias** for a physical collection `<name>__<INDEX_VERSION>` (now `v2`). The index version covers everything that decides what a stored point looks like: chunking, the sparse encoder, the embedded text and the payload. On start-up rag-api creates the physical collections of its version and:

- replaces an **empty** 0.1.0 collection (a real collection named `cms_content` etc.) with the alias, at once;
- keeps a **non-empty** 0.1.0 collection in service (`mode: legacy`, writes also go to `__v2`) until `POST /admin/collections/{name}/promote?drop_legacy=true`, because 0.1.0 points do not carry their source document and only the callers can re-send them.

Every chunk 0 stores its source document, so a later version bump needs no caller: set `INDEX_VERSION`, deploy (writes now go to the old and the new version, reads stay on the old one), then

```bash
curl -X POST -H "Authorization: Bearer $ADMIN" "http://rag-api.rag.svc.cluster.local:8000/admin/collections/cms_content/rebuild?promote=true"
curl -H "Authorization: Bearer $ADMIN" http://rag-api.rag.svc.cluster.local:8000/admin/collections   # progress
curl -X DELETE -H "Authorization: Bearer $ADMIN" http://rag-api.rag.svc.cluster.local:8000/admin/physical/cms_content__v2   # later, the old version
```

Re-ingesting never deletes first: new chunks are written under new point ids, then older versions of the `doc_id` are removed, so a search never sees the document missing and a failed write keeps the previous version.

## Limits and tuning

All of these are empty in `deploy/k8s/rag-api.yaml`, which means **no limit**; set numbers there when you want one.

| Variable | What it bounds |
| --- | --- |
| `INGEST_MAX_CONCURRENCY` | ingests (parse + embed + write) running at once; queries never wait for a slot, so a bulk reindex cannot queue them behind it |
| `MAX_UPLOAD_BYTES` | largest `/ingest` body (413 above it) |
| `DOCLING_TIMEOUT_SECONDS` | one file conversion in docling-serve (rag-api polls its async API, so no single HTTP request waits for the whole conversion) |
| `MAX_HISTORY_MESSAGES` | earlier chat messages forwarded to the model |
| `DEFAULT_MIN_DENSE_SCORE` | cosine floor for chat retrieval (requests can send `min_dense_score`) |
| `RERANKER_ENABLED`, `RERANKER_CANDIDATES`, `RERANKER_TIMEOUT_MS`, `RERANKER_MAX_CONCURRENT`, `RERANKER_THREADS` | the optional cross-encoder (`cross-encoder/ms-marco-MiniLM-L6-v2` int8, baked into the image, same model and pins as moks-seo-agent). Off; measure `rag_api_stage_duration_seconds{stage="rerank"}` on mk4 and the eval below before switching it on. |

## Observability

rag-api serves Prometheus metrics on `:9464` (`METRICS_PORT`), never routed by the edge; Prometheus job `rag-api` finds the pods through the headless Service `rag-api-metrics`. Labels use the authenticated service (`shared-token` for the shared token), never a claimed name.

| Metric | Meaning |
| --- | --- |
| `rag_api_http_requests_total`, `rag_api_http_request_duration_seconds` | by route template; streams are timed to their last byte |
| `rag_api_stage_duration_seconds{stage}` | `embed_query`, `embed_documents`, `search`, `rerank`, `docling`, `llm`, `condense` |
| `rag_api_retrievals_total`, `rag_api_retrieval_zero_hits_total` | searches and empty results, by service and collection |
| `rag_api_llm_tokens_total{service,model,purpose,kind}` | tokens LiteLLM reported (`purpose`: answer, chat, condense, task), streamed answers included |
| `rag_api_ingests_total{outcome}`, `rag_api_ingest_chunks_total`, `rag_api_ingests_in_progress`, `rag_api_ingests_waiting` | the write path |
| `rag_api_upstream_errors_total{dependency,kind}` | failed calls to bge-m3, qdrant, docling-serve, litellm |
| `rag_api_collection_info{collection,physical,mode}` | which physical collection each alias serves |

## Retrieval evaluation

`python -m rag_api.eval` asks a running rag-api (`POST /query`, `include_answer: false`, so no LLM cost) a question set and reports Recall@k, MRR, per type/tag, misses, and what no-answer questions got back (with `dense_score`, to calibrate `min_dense_score`). moks-seo-agent's 95-question blog set works as is for `cms_content`:

```bash
kubectl -n rag port-forward svc/rag-api 8000:8000 &
RAG_EVAL_TOKEN=... python -m rag_api.eval --collection cms_content \
  --questions ../moks-seo-agent/app/data/rag_eval.yaml --doc-id-template 'blog:{slug}' --k 1 3 5 --json report.json
```

Run it before and after a retrieval change (chunking, reranker, floor) and compare.

## Upgrading to 0.2.0

1. Deploy (merge). On start-up the empty 0.1.0 collections become aliases for `__v2` (on 2026-11-26 all three had 0 points). If one has points by then, it stays in service as `legacy`: re-send its documents from the caller (moks-cms Admin → Integrations → Hybrid RAG → Reindex all posts; seo-agent `seo_intel_sync_flow`; `moks shared-kb sync --all`), then `promote?drop_legacy=true`.
2. Callers need no change: they already send `doc_id` and `collection`, and keep the shared token until step 3.
3. Issue per-service tokens ([Access](#access-per-service-tokens)), then switch the shared token off.
4. Merge the Prometheus job `rag-api` (k8s-deployments) so the metrics are scraped.

**Rolling back to 0.1.0** after the aliases exist: 0.1.0 creates real collections named `sre_knowledge` etc. and cannot
while aliases of those names exist, so first remove the aliases (Qdrant `POST /collections/aliases` with
`delete_alias` for each), then deploy the old image; 0.1.0 starts with empty collections and the callers re-send.

---

## Local Development & CI/CD

```bash
cd app
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ruff check src tests
pytest -v
```

The tests run a real `QdrantStore` on qdrant-client's in-memory mode (no server). In-memory Qdrant ignores payload indexes, so index behaviour is only exercised on the cluster.

Pushing changes under `app/**` to `main` triggers `.github/workflows/build-and-deploy.yaml`:
1. Runs `ruff` and `pytest` on `app/`; on pull requests also builds the image (amd64, not pushed).
2. Builds and pushes a multi-arch (`linux/arm64,linux/amd64`) image to `ghcr.io/<owner>/moks-rag-api:<sha>`.
3. Updates `newTag` in `deploy/k8s/kustomization.yaml` and commits back to `main` (`[skip ci]`), triggering ArgoCD auto-sync.
