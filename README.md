# `moks-rag` — Hybrid RAG Platform for Hetzner k3s

Production, CPU-only Hybrid RAG platform deployed in the `rag` namespace on Hetzner k3s (ARM64 CAX nodes) for `jithendar.com`, managed via GitHub Actions, ArgoCD, and Istio (`STRICT` mTLS).

## Architecture

All components run in the `rag` namespace and communicate over internal Kubernetes DNS (`*.rag.svc.cluster.local`). Only `rag-api` (`https://jithendar.com/rag`), `open-webui` (`https://jithendar.com/chat`), Qdrant's REST port with its Web UI (`https://jithendar.com/qdrant`, behind Qdrant's API key, see [Browsing Qdrant](#browsing-qdrant)) and LiteLLM with its Admin UI (`https://jithendar.com/llm`, behind LiteLLM's own keys and sign-in, see [LiteLLM Admin UI](#litellm-admin-ui)) are exposed through the Istio ingress gateway (`istio-ingress/public`); every other backend service is strictly `ClusterIP`-only.

| Component | Image / Chart | Internal Endpoint | Purpose |
| --- | --- | --- | --- |
| `bge-m3` | `ollama/ollama:0.40.16` (`bge-m3:567m`) | `http://bge-m3.rag.svc.cluster.local:8001` (and `:11434`) | 1024-dim dense embeddings on ARM64 CPU (migrated from `embeddings` namespace) |
| `qdrant` | Helm `qdrant/qdrant` chart `1.20.2` (ArgoCD multi-source) | `http://qdrant.rag.svc.cluster.local:6333` | Vector store (`15Gi` PVC; 3 aliases over versioned dense+sparse collections, see [Index versions](#index-versions)) |
| `docling-serve` | `ghcr.io/docling-project/docling-serve-cpu` 1.39.0, pinned by digest | `http://docling-serve.rag.svc.cluster.local:5001` | Document parsing (PDF/DOCX/HTML -> Markdown via its async API, 2–4Gi RAM); Web UI at `jithendar.com/docling` behind the gateway's login |
| `litellm` | `ghcr.io/berriai/litellm` 1.112.0.dev2, pinned by digest | `http://litellm.rag.svc.cluster.local:4000/v1` | OpenAI-compatible proxy: `claude-sonnet-5-5` & `claude-haiku-5-5` (config), `moks-cms-default` and UI-added models (PostgreSQL `litellm` on the shared CloudNativePG cluster); counters and model-change broadcast on the shared Valkey (database 8); Admin UI at `jithendar.com/llm/ui` |
| `rag-api` | `ghcr.io/mrjithendar/moks-rag-api:<sha>` | `http://rag-api.rag.svc.cluster.local:8000` (metrics `:9464`) | Custom FastAPI orchestrator (`root_path="/rag"`) |
| `open-webui` | `ghcr.io/open-webui/open-webui`, pinned by digest | `http://open-webui.rag.svc.cluster.local:80` | Chat UI (`2Gi` PVC) pointing at `rag-api` `/v1` |

### Collections & Virtual Models

| Open WebUI Virtual Model | Qdrant Collection | Vectors | Intended Consumer |
| --- | --- | --- | --- |
| `moks-sre` | `sre_knowledge` | `dense` (1024 cosine) + `sparse` (BM25/IDF) | `moks-sre` runbooks & incidents |
| `moks-cms` | `cms_content` | `dense` (1024 cosine) + `sparse` (BM25/IDF) | `moks-cms` articles & webhook sync |
| `moks-seo-agent` | `seo_intel` | `dense` (1024 cosine) + `sparse` (BM25/IDF) | `moks-seo-agent` audits & SERP intel |
| `mrjithendar-repos` | `github_repos` | `dense` (1024 cosine) + `sparse` (BM25/IDF) | the GitHub repositories' documentation, nightly ([Indexing the GitHub repositories](#indexing-the-github-repositories)) |

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
│   │   ├── repo_index.py  # python -m rag_api.repo_index (CronJob repo-index)
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
│   │   ├── litellm.yaml         # LiteLLM, its config models, the moks-cms-default seed (PostSync Job)
│   │   ├── rag-api.yaml
│   │   ├── open-webui.yaml
│   │   ├── istio-routing.yaml
│   │   ├── repo-index.yaml      # what to index from GitHub + the nightly CronJob
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
export LITELLM_SALT_KEY="sk-$(openssl rand -hex 32)"      # never change it later (LiteLLM Admin UI > Operational warnings)
export LITELLM_UI_USERNAME="<your admin name, not admin>"
export LITELLM_UI_PASSWORD="$(openssl rand -hex 24)"
export RAG_BEARER_TOKEN="$(openssl rand -hex 32)"
export WEBUI_SECRET_KEY="$(openssl rand -hex 32)"

kubectl -n rag create secret generic rag-secrets \
  --from-literal=ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}" \
  --from-literal=QDRANT_API_KEY="${QDRANT_API_KEY}" \
  --from-literal=LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY}" \
  --from-literal=LITELLM_SALT_KEY="${LITELLM_SALT_KEY}" \
  --from-literal=LITELLM_UI_USERNAME="${LITELLM_UI_USERNAME}" \
  --from-literal=LITELLM_UI_PASSWORD="${LITELLM_UI_PASSWORD}" \
  --from-literal=RAG_BEARER_TOKEN="${RAG_BEARER_TOKEN}" \
  --from-literal=WEBUI_SECRET_KEY="${WEBUI_SECRET_KEY}" \
  --dry-run=client -o yaml | kubectl apply -f -
```

Two optional keys are added later by `scripts/issue-service-token.sh` (see [Access](#access-per-service-tokens)):
`RAG_SERVICE_TOKENS` (JSON, one token per calling service) and `OPEN_WEBUI_RAG_TOKEN`. LiteLLM's database URL is not in
`rag-secrets`: k8s-deployments' Terraform writes Secret `rag/postgres-db-credentials` ([First deployment](#first-deployment)).

**Option B — GitOps with SealedSecrets / External Secrets Operator:**

- Use `deploy/k8s/rag-secrets.sealed-secret.yaml.template` with `kubeseal`, or `deploy/k8s/rag-secrets.external-secret.yaml.template` with Vault ESO, then add the generated file to `deploy/k8s/kustomization.yaml`.

### 2. Wire `jithendar.com/rag` and `jithendar.com/chat` in the Edge Gateway

In your cluster edge chart (`k8s-deployments/deploy/minikube/istio/edge/values.yaml`), add five delegates under `root.delegates` so the shared `istio-ingress/jithendar-com` root VirtualService delegates `/rag`, `/chat`, `/qdrant`, `/docling` and `/llm`, and a gateway login for `/docling` (see [Browsing docling-serve](#browsing-docling-serve)):

```yaml
root:
  delegates:
    - {name: rag, prefix: /rag, to: rag/rag-api}     # <-- add
    - {name: chat, prefix: /chat, to: rag/open-webui} # <-- add
    - {name: qdrant, prefix: /qdrant, to: rag/qdrant} # <-- add
    - {name: docling, prefix: /docling, to: rag/docling-serve} # <-- add
    - {name: llm, prefix: /llm, to: rag/litellm}     # <-- add
basicAuth:
  - {name: docling, prefix: /docling, from: rag/docling-ui-auth} # <-- add
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

## Browsing docling-serve

docling-serve's own Web UI is at **https://jithendar.com/docling** (redirects to `/docling/ui/`). Drop a file or give a URL, pick the output formats and options, press Convert, then read or download the result (Markdown, HTML, JSON, the `.dclx` archive, or open it in the viewer). It is the same service rag-api sends files to, so it shows what `POST /ingest` gets back for a PDF or DOCX. The API reference is at `/docling/docs` (ReDoc) and `/docling/swagger`.

docling-serve has no API key here, so the **gateway asks for a login** before anything under `/docling` (edge chart `basicAuth` entry `docling`). Create that login once, from `k8s-deployments`:

```bash
kubectl -n rag create secret generic docling-ui-auth --type=kubernetes.io/basic-auth \
  --from-literal=username=docling --from-literal=password="$(openssl rand -hex 20)"
kubectl -n rag get secret docling-ui-auth -o jsonpath='{.data.password}' | base64 -d | pbcopy
python3 scripts/istio/gateway-basic-auth.py   # rebuilds istio-ingress/gateway-basic-auth, then terraform apply
```

Until that Secret exists, the gateway keeps `/docling` locked (Terraform refuses to plan without the line), never open. Rerun the script and the apply after changing the password.

Mind what the UI can do: a conversion takes up to the pod's 2 CPUs and 4Gi, so a large PDF slows rag-api's file ingests while it runs; and **URL** sources are fetched from inside the cluster.

How `/docling` works: the `docling-serve` VirtualService in `deploy/k8s/istio-routing.yaml` strips the prefix and sets `Cache-Control: no-store` on everything except the UI's hashed files, so Cloudflare never keeps a converted document; the `docling-serve-ingress` AuthorizationPolicy lets the gateway reach port 5001 only. The UI takes its API address from its own URL (everything before `/ui`), including the status WebSocket, so it needs no patch. `UVICORN_ROOT_PATH=/docling` only fixes the links FastAPI writes (the API docs, the `/ui` redirect); requests still arrive without the prefix, so rag-api's calls are unchanged. CORS is limited to `https://jithendar.com`.

---

## LiteLLM Admin UI

LiteLLM is at **https://jithendar.com/llm**: the Admin UI at `/llm/ui/` (`/llm` redirects there), Swagger at `/llm/docs` (`/llm/openapi.json`), and the OpenAI-compatible API at `/llm/v1/...`. In the cluster, callers keep using `http://litellm.rag.svc.cluster.local:4000/v1` (no `/llm`).

Models, provider credentials and virtual keys live in PostgreSQL: database `litellm` on the shared CloudNativePG cluster (`postgres-db/postgres-cluster`, through PgBouncer), with its own login role. Changes made in the UI take effect without restarting LiteLLM or its callers.

| Kind of model | Where it is defined | Edit it in |
| --- | --- | --- |
| `claude-sonnet-5-5`, `claude-haiku-5-5` (rag-api's `DEFAULT_LLM_MODEL` / `FAST_LLM_MODEL`) | `litellm_config.yaml` in `deploy/k8s/litellm.yaml` | Git (the UI shows them as *Defined in config*, read-only) |
| `moks-cms-default` | created once by the Job `litellm-seed-models` from `deploy/k8s/litellm.yaml` | the Admin UI |
| everything you add in the UI | PostgreSQL | the Admin UI |

Never give a UI model the name of a config model: LiteLLM would then serve both under that name, in turn.

### First deployment

Order matters. Each step leaves the running LiteLLM untouched until the last one.

1. **Database** (k8s-deployments): merge the `postgres_databases.litellm = { namespace = "rag" }` entry in `minikube-infrastructure/vars/mk4.tfvars` (with the `/llm` edge delegate). Terraform creates database and role `litellm`, and Secret `rag/postgres-db-credentials`:
   ```bash
   kubectl -n postgres-db get database postgres-cluster-litellm          # APPLIED true
   kubectl -n rag get secret postgres-db-credentials -o jsonpath='{.data}' | python3 -c 'import json,sys; print(sorted(json.load(sys.stdin)))'
   ```
2. **Secrets**: add LiteLLM's three new keys to `rag-secrets`. Values go through stdin, never the command line:
   ```bash
   read -r -p "LiteLLM admin user name (not 'admin'): " UIUSER
   SALT="sk-$(openssl rand -hex 32)"; PASS="$(openssl rand -hex 24)"
   kubectl -n rag patch secret rag-secrets --type merge --patch-file /dev/stdin <<EOF
   {"stringData": {"LITELLM_SALT_KEY": "$SALT", "LITELLM_UI_USERNAME": "$UIUSER", "LITELLM_UI_PASSWORD": "$PASS"}}
   EOF
   unset SALT PASS UIUSER
   ```
   Then copy `LITELLM_SALT_KEY` and `LITELLM_UI_PASSWORD` into your password manager (`kubectl -n rag get secret rag-secrets -o jsonpath='{.data.LITELLM_SALT_KEY}' | base64 -d | pbcopy`). The salt key is the only way to read what LiteLLM stores; a database backup is useless without it.
3. **Merge this repository's change.** The `deploy-argocd-apps` workflow applies the AppProject (it now allows `Job`), and Argo CD rolls LiteLLM: the new pod applies the database migrations (`prisma migrate deploy`, about 260 on an empty database) before serving, while the old pod keeps serving. Then the PostSync hook creates `moks-cms-default`:
   ```bash
   kubectl -n rag rollout status deploy/litellm --timeout=10m
   kubectl -n rag logs job/litellm-seed-models        # moks-cms-default: created -> anthropic/claude-sonnet-5-5
   ```

Until steps 1 and 2 are done, the new LiteLLM pod waits for its Secret keys (`CreateContainerConfigError`) and the old pod keeps serving rag-api. If Argo CD synced before the workflow applied the AppProject, the sync fails on the Job (`batch:Job is not permitted`) and its retries pick it up a minute later.

### Accessing the Admin UI

Open **https://jithendar.com/llm/ui/** and sign in with `LITELLM_UI_USERNAME` / `LITELLM_UI_PASSWORD` from `rag-secrets` (proxy admin). Read them with `kubectl -n rag get secret rag-secrets -o jsonpath='{.data.LITELLM_UI_PASSWORD}' | base64 -d | pbcopy`. A session lasts 24 hours (`LITELLM_UI_SESSION_DURATION` shortens it).

The red **Environment-credential login is enabled** banner refers to exactly this sign-in. To retire it, create a proxy-admin user with a password in the UI (its users page, `/llm/ui/users/`), sign in as that user, then set `general_settings.disable_env_credential_login: true` in `litellm_config.yaml`. Do it in that order: the other way round nobody can sign in (the master key still works over the API).

The master key (`LITELLM_MASTER_KEY`) is for administration over the API only. Applications get virtual keys.

### Adding a provider and model

1. *Models + Endpoints* → **LLM Credentials**: add one for the provider (pick it, paste its API key, name it, e.g. `openai`). LiteLLM encrypts it with `LITELLM_SALT_KEY` before storing it.
2. *Models + Endpoints* → **Add Model**: provider, the provider's model (e.g. `openai/gpt-...`), the **public model name** callers will use, and the credential from step 1 (*Existing credentials*), or a key typed in directly.
3. *Add Model* saves it to PostgreSQL. It is served within seconds (at once by the pod that saved it; the others reload from the database every 30 s).
4. Test it: the model's page → **Test Connection**, or *Playground*, or over the API with a key allowed that model:
   ```bash
   curl -s https://jithendar.com/llm/v1/chat/completions -H "Authorization: Bearer $KEY" \
     -H 'Content-Type: application/json' -d '{"model":"<public name>","messages":[{"role":"user","content":"ping"}]}'
   ```

### Creating a virtual key for moks-cms

1. *Virtual Keys* → **Create New Key**. Key name `moks-cms`; **Models**: only `moks-cms-default`. Budget, rate limits (TPM/RPM) and expiry are optional: leave them empty for no limit, or set your own.
2. **Create Key** shows the key once. Copy it.
3. Put it in moks-cms's Secret, through stdin:
   ```bash
   K="$(pbpaste)"
   kubectl -n jithendar patch secret jithendar-com-env --type merge --patch-file /dev/stdin <<EOF
   {"stringData": {"LITELLM_API_KEY": "$K"}}
   EOF
   unset K; pbcopy </dev/null
   ```
   The site loads `jithendar-com-env` with `envFrom`, so new site pods see `LITELLM_API_KEY`; running pods keep their environment until they are replaced. Today nothing in moks-cms reads it: moks-cms has no LiteLLM client yet. Wiring it means a client in moks-cms calling `http://litellm.rag.svc.cluster.local:4000/v1` with model `moks-cms-default`, and an egress rule to `rag`/`app=litellm` port 4000 in jithendar.com's `kubernetes/30-jithendar-com.yaml` NetworkPolicy. The `litellm-callers` AuthorizationPolicy here already admits namespace `jithendar`.

That key can call `moks-cms-default` and nothing else (another model answers 403). It is not the master key, and must never be: a leaked master key can create keys and change every model and credential.

### Switching the model live

The callers' model name stays `moks-cms-default`; only what it points at changes.

1. Open https://jithendar.com/llm/ui/ → **Models + Endpoints**.
2. Click the model ID of `moks-cms-default` (its *Model Information* shows the current provider model).
3. **Edit Settings** → change **LiteLLM Model Name** (e.g. `anthropic/claude-sonnet-5-5` → `anthropic/claude-haiku-5-5`). For another provider, also set its credential with **Re-use Credentials** or **Update API Key**.
4. **Save** (the UI sends `PATCH /llm/model/<id>/update`; LiteLLM writes PostgreSQL and reloads that model at once).
5. Check the alias from the API:
   ```bash
   curl -s https://jithendar.com/llm/model/info -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
     | python3 -c 'import json,sys; [print(m["model_name"], "->", m["litellm_params"]["model"]) for m in json.load(sys.stdin)["data"]]'
   ```
6. Send a request with the moks-cms virtual key; the response headers name the model that answered:
   ```bash
   curl -s -D - -o /dev/null https://jithendar.com/llm/v1/chat/completions -H "Authorization: Bearer $MOKS_CMS_KEY" \
     -H 'Content-Type: application/json' -d '{"model":"moks-cms-default","messages":[{"role":"user","content":"ping"}]}' \
     | grep -i -E 'x-litellm-model-(group|name)'
   # x-litellm-model-name: anthropic/claude-haiku-5-5
   # x-litellm-model-group: moks-cms-default
   ```
7. Every later request uses the new model. Neither LiteLLM nor moks-cms restarts (`kubectl -n rag get pod -l app=litellm` shows the same pod and restart count).

With more than one LiteLLM pod, the others pick the change up from PostgreSQL within 30 s. A sync of `moks-rag-stack` never undoes the switch: the seed Job only creates names that do not exist.

What still needs Git and a restart: the config models (`claude-sonnet-5-5`, `claude-haiku-5-5`), `general_settings` / `litellm_settings` in `litellm_config.yaml`, and the environment (`SERVER_ROOT_PATH`, keys, `UI_PASSWORD`). Retiring `moks-cms-default` for good: remove it from `litellm-seed-models` first, or the next sync creates it again.

### Redis (Valkey)

LiteLLM uses database **8** of the shared Valkey (`redis/valkey-master`) as ACL user `litellm`, which can open no other database and run no admin command. k8s-deployments (`redis_tenants.litellm` in `vars/mk4.tfvars`) writes Secret `rag/redis-litellm-credentials`; its `uri` is LiteLLM's `REDIS_URL`.

What goes there: the virtual keys' rate-limit and spend counters (so limits hold across pods and restarts), the broadcast that tells every pod at once that a model changed (channel `litellm_proxy.config_change`; without Redis the other pods notice within 30 s), and a few small config-cache keys. Counters are keyed by a hash of the virtual key, never the key itself. **No response cache:** `litellm_settings.cache` stays off, so prompts and answers never reach Valkey. Turning it on is a separate decision: they would sit in the shared Valkey, where RedisInsight's console login reads every database. Give it its own database number if you do.

Valkey is optional for LiteLLM. If it does not answer when LiteLLM starts, LiteLLM runs on in-memory counters until its next restart; if it goes away later, requests go on. Tried on 1.112.0.dev2 with an ACL user like the tenant's (`-@admin -flushall -swapdb`): only `SELECT 8`, no refused command, chat answered with Redis stopped and when started without it.

Check it:
```bash
kubectl -n rag logs deploy/litellm -c litellm | grep -i "stays in-memory"   # nothing = LiteLLM is on Valkey
```
and in RedisInsight (https://jithendar.com/redis/, database 8): `litellm_config:param:*` and, after traffic, `{api_key:<hash>}:tokens`.

### Operational warnings

- **`LITELLM_SALT_KEY` must never change.** It encrypts every model parameter and credential LiteLLM stores. With another salt key LiteLLM starts, but every UI-added model and credential silently disappears (they cannot be decrypted; tried on 1.112.0.dev2: `moks-cms-default` became "Invalid model name"), and the seed Job would create a second `moks-cms-default`. Virtual keys survive (they are hashed, not encrypted). Putting the old salt key back restores everything. Keep a copy outside the cluster; losing it means re-entering every model and credential.
- **Rotating secrets:** `LITELLM_MASTER_KEY` signs UI sessions and is rag-api's LiteLLM key today, so rotating it signs everyone out and needs rag-api's restart too (`kubectl -n rag rollout restart deploy/litellm deploy/rag-api`). `LITELLM_UI_PASSWORD`: patch the Secret, restart LiteLLM. A virtual key: create a new one with the same models, patch the caller's Secret, then delete the old key. The database password belongs to Terraform (k8s-deployments); LiteLLM reads it at start.
- **Backups:** the shared cluster has **no backups yet** (`postgres_backups` unset in k8s-deployments: no ScheduledBackup, the replica is not a backup). Until then, dump before upgrades and now and then:
  ```bash
  kubectl -n postgres-db exec postgres-cluster-1 -c postgres -- pg_dump -Fc -d litellm > litellm-$(date +%F).dump
  ```
  (run it on the current primary: `kubectl -n postgres-db get cluster postgres-cluster -o jsonpath='{.status.currentPrimary}'`). Restore: `kubectl -n rag scale deploy/litellm --replicas=0`, then `kubectl -n postgres-db exec -i <primary> -c postgres -- pg_restore --clean --if-exists --no-owner --role=litellm -d litellm < litellm-<date>.dump`, then scale back to 1. What it restores is only readable with the same `LITELLM_SALT_KEY`.
- **Storage:** the data sits on the cluster's two 10Gi `local-path` volumes (one per instance, pinned to its node), shared with every tenant. LiteLLM's share is small: no per-request rows (`disable_spend_logs: true`), only models, keys, users and daily spend totals.
- **Upgrading LiteLLM:** change the image digest. Each start runs `prisma migrate deploy`; the schema only moves forward, so going back to an older image after a migration is not supported: dump first, and roll back by restoring the dump with the old image. During the rolling update the old pod keeps serving on the migrated schema; LiteLLM's default (v2) migration resolver is meant for exactly that overlap. Watch `kubectl -n rag logs deploy/litellm | grep -i migrat`.
- **Database down = LiteLLM unready.** `/health/readiness` answers 503 while PostgreSQL cannot be reached (virtual keys cannot be checked), so rag-api's answers stop too. LiteLLM's `allow_requests_on_db_unavailable` would keep serving, but would also stop checking keys; it stays off because `/llm` is public. The cluster fails over to its replica on its own.
- **Memory:** 1.16 GiB working set before the database; requests 1Gi, limit 2Gi. The previous 1.5 GiB limit killed the container when a second process (the prisma CLI) started beside it. A rollout runs a second LiteLLM pod until the new one is ready, and starting is the expensive part: on 2026-11-28 two LiteLLM pods starting on worker1 at once (a pending rollout plus a `rollout restart`) used up that node's memory, which was already ~80% used against ~69% requested; the node went NotReady for 11 minutes and the kernel killed docling-serve. Before `kubectl -n rag rollout restart deploy/litellm`, wait for any rollout in progress and look at `kubectl top nodes`.

### Shared origin: the UI's session cookie

LiteLLM keeps the UI session in a cookie named `token`, on path `/` of jithendar.com, readable by JavaScript (the UI reads it, so it cannot be HttpOnly). It is a signed JWT that **contains a proxy-admin API key**, valid for the session (24 h). jithendar.com is one origin for the site and every admin tool, so any script running on any jithendar.com page, including the site's third-party scripts, can read it while you are signed in. Also, Open WebUI (`/chat`) uses a cookie of the same name and path: signing in to one replaces the other's.

Until the UI moves to its own host name (e.g. `llm.jithendar.com`, the lasting fix), sign out of the LiteLLM UI when done, use a private window for it, or shorten sessions with `LITELLM_UI_SESSION_DURATION`. Cloudflare Access (edge `cloudflare.access.prefixes` lists `/llm`) guards who reaches `/llm`, not the cookie.

### Troubleshooting

| Symptom | Check |
| --- | --- |
| Pod `CreateContainerConfigError` | `kubectl -n rag describe pod -l app=litellm`: Secret `postgres-db-credentials` (k8s-deployments not applied) or a `rag-secrets` key missing |
| Pod not Ready, `/health/readiness` 503 `"db":"disconnected"` | `kubectl -n postgres-db get cluster,pooler`; `kubectl -n postgres-db get database postgres-cluster-litellm`; LiteLLM's log for `P1001` (cannot reach) or `P1000` (password) |
| Start-up exits after "Database migration cannot proceed" | `kubectl -n rag logs deploy/litellm --previous`; usually permissions: the role must own database `litellm` |
| Sign-in says invalid credentials | the values in `rag-secrets` (`LITELLM_UI_USERNAME`/`PASSWORD`) as of the pod's start: patching the Secret needs `kubectl -n rag rollout restart deploy/litellm`. Repeated failures block that user name for a while (HTTP 429) |
| UI blank, or `/llm/ui/` shows the site's 404 | `curl -sI https://jithendar.com/llm/ui/` must carry `cache-control: no-store, no-transform` (set by the `litellm` VirtualService); if not, the edge delegate `llm` (k8s-deployments) is missing. LiteLLM's log must not say `Cannot apply server_root_path replacements` |
| A model added in the UI is missing | `kubectl -n rag logs deploy/litellm | grep -i decrypt` (salt key changed?); a name equal to a config model? |
| Seed Job failed | `kubectl -n rag logs job/litellm-seed-models` |
| Log says `cross-pod coordination stays in-memory` | Valkey did not answer at start-up, or Secret `redis-litellm-credentials` is missing (`kubectl -n rag get secret redis-litellm-credentials`): fix that, then restart LiteLLM (see Memory above) |

How `/llm` works: the `litellm` VirtualService in `deploy/k8s/istio-routing.yaml` keeps the prefix (LiteLLM runs with `SERVER_ROOT_PATH=/llm` and rewrites its UI files for it at start-up), redirects `/llm`, `/llm/` and `/llm/ui` to `/llm/ui/`, and sets `Cache-Control: no-store` on everything except the UI's hashed files under `/llm/_next/static/`. It matches only `/llm` and `/llm/...`, so the site keeps `/llms.txt`. `PROXY_BASE_URL=https://jithendar.com/llm` gives the UI its API base; `FORWARDED_ALLOW_IPS=127.0.0.6` (the sidecar) lets LiteLLM's own redirects say `https://`. The `litellm-callers` AuthorizationPolicy admits the gateway and namespace `jithendar` on port 4000. Two of the UI's requests leave `/llm` and get the site's 404 (`/get_favicon`, `/ui/litellm-runtime-config.json`); the UI does not need them.

## Indexing the GitHub repositories

The chat model **`mrjithendar-repos`** answers from the documentation of the GitHub repositories (collection `github_repos`). CronJob `repo-index` (`deploy/k8s/repo-index.yaml`) fills it every night at 02:00 UTC with `python -m rag_api.repo_index` from the rag-api image:

1. It lists every repository of `mrjithendar` that is not archived (one GitHub API call), so **a new repository is indexed by the next run without any change**.
2. For each, it downloads the default branch as one tarball and takes the files that match `include` and no `exclude` glob in the config: Markdown, MDX, reStructuredText, text, PDF, DOCX and PPTX, without `jithendar.com/content/` (the blog posts: moks-cms indexes them into `cms_content`), test fixtures, diagram captions, `node_modules` and the like.
3. Each file is one document, `doc_id` `owner/name/path`, sent to rag-api one at a time: text as text, PDF/DOCX/PPTX as an upload that rag-api converts with docling-serve. Its metadata has `repo`, `path`, `branch`, `url` (the file on GitHub) and `blob_sha`, so an unchanged file is answered `unchanged` without embedding anything, and answers can link to the file.
4. Then it deletes what is gone: files removed from a repository or no longer matching, and every document of a repository that was deleted, archived or excluded. Nothing is deleted when listing GitHub failed or a repository could not be read.

On 2026-11-26 that was 11 repositories, 419 files, 3.6 MB of text (one PDF); the first run embeds all of it, later runs only what changed.

**Adding or removing repositories and files:** edit the `config.json` in ConfigMap `repo-index` (PR, or `kubectl -n rag edit configmap repo-index`):

| Key | Meaning |
| --- | --- |
| `orgs` | every non-archived repository of these orgs |
| `extra_repos` | more repositories, `"owner/name"` (public, or reachable with `GITHUB_TOKEN`) |
| `exclude_repos` | `"owner/name"` never indexed |
| `include`, `exclude` | path globs: `**/` any directories (none included), `*` anything but `/`; `"owner/name:glob"` applies to one repository |
| `max_file_bytes` | skip larger files; `null` = no limit |

**Once, before the first run:**

```bash
scripts/issue-service-token.sh repo-indexer            # Secret rag/repo-indexer-rag; restarts rag-api
# optional, raises the GitHub API limit (60 calls an hour without it, shared by the cluster's address):
kubectl -n rag create secret generic repo-index-github --from-literal=GITHUB_TOKEN=<token, no scopes needed>
kubectl -n rag create job --from=cronjob/repo-index repo-index-first     # or wait for 02:00 UTC
kubectl -n rag logs -f job/repo-index-first -c repo-index
```

Each repository logs a `repo_index_repo` line (files, indexed, unchanged, rejected, failed, deleted) and the run ends with `repo_index_done`. The job fails (exit 1) when anything failed; the next run sends it again. `rejected` (rag-api answered 422: an empty file, a chunk too long, a PDF without text) is logged, not failed. In Open WebUI the model appears as `mrjithendar-repos`; if a user does not see it, check its visibility in Admin Panel → Settings → Models.

---

## API contract (0.2.0)

| Endpoint | Scope | Notes |
| --- | --- | --- |
| `POST /ingest` | write on `collection` | JSON or multipart (`file` via docling-serve). **`doc_id` is required**: sending the same `doc_id` again replaces the document. The answer has `unchanged: true` when the stored version was identical (nothing embedded or written). A chunk too long for bge-m3's context is refused with 422 instead of being embedded from its start only. |
| `DELETE /documents/{doc_id}?collection=…` | write | **`collection` is required** (`all` = every collection the caller may write). `doc_id` may contain `/`. Also `DELETE /documents?doc_id=…&collection=…`. |
| `POST /documents/list` | read | `{collection, filters}` → `{doc_ids}`: every stored doc_id, or those whose metadata matches every filter exactly (a syncing caller finds what it no longer has). |
| `POST /query` | read | Each source has `score` (rank fusion: orders one answer only), **`dense_score`** (cosine similarity, comparable across queries) and `rerank_score` when the reranker ran. `min_dense_score` drops chunks below a cosine floor. `filters` match payload fields exactly (not `text` or `document*`; the first filter on a field indexes it). `model` must be `DEFAULT_LLM_MODEL` or `FAST_LLM_MODEL`. |
| `GET /v1/models`, `POST /v1/chat/completions` | read on the model's collection | OpenAI-compatible, for Open WebUI. Follow-ups are rewritten into a standalone search query with `FAST_LLM_MODEL` (`CONDENSE_FOLLOWUPS`). Open WebUI's background tasks (title, tags, follow-ups: messages starting `### Task:`) skip retrieval and use `FAST_LLM_MODEL`. |
| `GET /admin/collections`, `POST /admin/collections/{name}/rebuild`, `…/promote`, `DELETE /admin/physical/{name}` | admin | [Index versions](#index-versions). |

Prompts: the system prompt is fixed per model; retrieved chunks go in the user turn inside `<documents><document doc_id=… title=… section=… url=…>` (`url` when the document's metadata has an `https://` one) and the system prompt says they are data, not instructions (retrieved text can come from third-party pages, e.g. `seo_intel`).

Errors name the dependency that failed: `503` unreachable, `504` too slow, `429` rate-limited (with `Retry-After`), `502` anything else, body `{"detail", "dependency", "kind", "request_id"}`. A stream that fails after it started ends with an SSE `{"error": …}` event and `[DONE]`.

## Access: per-service tokens

Every caller can get its own token, limited by the access policy in `deploy/k8s/rag-api.yaml` (ConfigMap `rag-api-access`): `moks-cms` writes `cms_content`; `moks-seo-agent` writes `seo_intel` and reads `cms_content`; `moks-sre-agent` writes `sre_knowledge`; `repo-indexer` writes `github_repos`; `open-webui` reads everything; `rag-admin` may use `/admin/*`. With its own token, a caller's logs and metrics carry the token's service, whatever `X-Service-Name` says.

The 0.1.0 shared `RAG_BEARER_TOKEN` keeps full access while `RAG_SHARED_TOKEN_ENABLED` is `"true"`, so nothing breaks during the move. To move one caller (no code change in the caller: each one already reads its token from its own Secret):

```bash
scripts/issue-service-token.sh moks-cms         # Secret jithendar/jithendar-com-rag (key token)
scripts/issue-service-token.sh moks-seo-agent   # Secret jithendar/seo-agent-rag (RAG_BEARER_TOKEN)
scripts/issue-service-token.sh moks-sre-agent   # Secret moks-sre/moks-rag-client (RAG_BEARER_TOKEN)
scripts/issue-service-token.sh open-webui       # rag-secrets OPEN_WEBUI_RAG_TOKEN
scripts/issue-service-token.sh repo-indexer     # Secret rag/repo-indexer-rag (RAG_BEARER_TOKEN), for CronJob repo-index
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

Ingests of one document (same collection and `doc_id`) take turns, in arrival order; different documents still run side by side. A caller that gives up before rag-api answers and sends the same document again therefore waits for the first send and then gets `"unchanged": true`, instead of embedding it a second time beside the first (that is how moks-cms's 10-second retries used to queue bge-m3 into 504s). The `rag_ingest` log line carries `waited_for_same_document_s`; `rag_api_ingests_waiting_for_document` counts the ones waiting.

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
| `rag_api_ingests_total{outcome}`, `rag_api_ingest_chunks_total`, `rag_api_ingests_in_progress`, `rag_api_ingests_waiting`, `rag_api_ingests_waiting_for_document` | the write path |
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
