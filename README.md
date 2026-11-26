# `moks-rag` — Hybrid RAG Platform for Hetzner k3s

Production, CPU-only Hybrid RAG platform deployed in the `rag` namespace on Hetzner k3s (ARM64 CAX nodes) for `jithendar.com`, managed via GitHub Actions, ArgoCD, and Istio (`STRICT` mTLS).

## Architecture

All components run in the `rag` namespace and communicate over internal Kubernetes DNS (`*.rag.svc.cluster.local`). Only `rag-api` (`https://jithendar.com/rag`) and `open-webui` (`https://jithendar.com/chat`) are exposed through the Istio ingress gateway (`istio-ingress/public`); every backend service is strictly `ClusterIP`-only.

| Component | Image / Chart | Internal Endpoint | Purpose |
| --- | --- | --- | --- |
| `bge-m3` | `ollama/ollama:0.40.16` (`bge-m3:567m`) | `http://bge-m3.rag.svc.cluster.local:8001` (and `:11434`) | 1024-dim dense embeddings on ARM64 CPU (migrated from `embeddings` namespace) |
| `qdrant` | Helm `qdrant/qdrant` (ArgoCD multi-source) | `http://qdrant.rag.svc.cluster.local:6333` | Vector store (`15Gi` PVC, 3 isolated dense+sparse RRF collections) |
| `docling-serve` | `ghcr.io/docling-project/docling-serve-cpu:latest` | `http://docling-serve.rag.svc.cluster.local:5001` | Document parsing (PDF/DOCX/HTML -> Markdown, 2–4Gi RAM) |
| `litellm` | `ghcr.io/berriai/litellm:main-latest` | `http://litellm.rag.svc.cluster.local:4000/v1` | OpenAI-compatible proxy routing `claude-sonnet-5-5` & `claude-haiku-5-5` |
| `rag-api` | `ghcr.io/mrjithendar/moks-rag-api:<sha>` | `http://rag-api.rag.svc.cluster.local:8000` | Custom FastAPI orchestrator (`root_path="/rag"`) |
| `open-webui` | `ghcr.io/open-webui/open-webui:main` | `http://open-webui.rag.svc.cluster.local:80` | Chat UI (`2Gi` PVC) pointing at `rag-api` `/v1` |

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
└── .github/workflows/
    ├── build-and-deploy.yaml            # Lint/test -> multi-arch GHCR push -> Kustomize tag bump
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

**Option B — GitOps with SealedSecrets / External Secrets Operator:**

- Use `deploy/k8s/rag-secrets.sealed-secret.yaml.template` with `kubeseal`, or `deploy/k8s/rag-secrets.external-secret.yaml.template` with Vault ESO, then add the generated file to `deploy/k8s/kustomization.yaml`.

### 2. Wire `jithendar.com/rag` and `jithendar.com/chat` in the Edge Gateway

In your cluster edge chart (`k8s-deployments/deploy/minikube/istio/edge/values.yaml`), add two delegates under `root.delegates` so the shared `istio-ingress/jithendar-com` root VirtualService delegates `/rag` and `/chat`:

```yaml
root:
  delegates:
    - {name: rag, prefix: /rag, to: rag/rag-api}     # <-- add
    - {name: chat, prefix: /chat, to: rag/open-webui} # <-- add
```

> **Open WebUI subpath note:** Upstream `ghcr.io/open-webui/open-webui:main` builds its SvelteKit frontend with `base: ""` (root-absolute `/_app/...`, `/static/...`, `/api/...`, `/ws/socket.io/...`). If assets 404 under `/chat/`, either build Open WebUI with SvelteKit `paths.base = "/chat"` or configure a `302` redirect in `istio-edge` from `/chat` to `chat.jithendar.com` (the same pattern used for Harbor's `/harbor` -> `registry.jithendar.com`).

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

## Local Development & CI/CD

```bash
cd app
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ruff check src tests
pytest -v
```

Pushing changes under `app/**` to `main` triggers `.github/workflows/build-and-deploy.yaml`:
1. Runs `ruff` and `pytest` on `app/`.
2. Builds and pushes a multi-arch (`linux/arm64,linux/amd64`) image to `ghcr.io/<owner>/moks-rag-api:<sha>`.
3. Updates `newTag` in `deploy/k8s/kustomization.yaml` and commits back to `main` (`[skip ci]`), triggering ArgoCD auto-sync.
