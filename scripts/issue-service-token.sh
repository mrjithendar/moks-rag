#!/usr/bin/env bash
# Issue (or rotate) one caller's own rag-api token. README "Access".
#
#   scripts/issue-service-token.sh <service>
#
# <service> is a key of deploy/k8s/rag-api.yaml's ConfigMap rag-api-access. The script:
#   1. generates a random token,
#   2. adds it to RAG_SERVICE_TOKENS (JSON) in Secret rag/rag-secrets (open-webui: also OPEN_WEBUI_RAG_TOKEN),
#   3. writes it into the caller's own Secret, keeping that Secret's other keys,
#   4. restarts rag-api (and Open WebUI for open-webui), then tells you which caller to restart.
# The token is never printed. The shared RAG_BEARER_TOKEN keeps working until RAG_SHARED_TOKEN_ENABLED=false.
#
# Uses your current kubectl context. To guard against the wrong cluster, export KUBE_CONTEXT=<name>: the script
# then refuses to run when the current context is a different one.
set -euo pipefail

service="${1:-}"
case "$service" in
  moks-cms)       ns=jithendar; secret=jithendar-com-rag; key=token;            restart="deploy/jithendar-com -n jithendar" ;;
  moks-seo-agent) ns=jithendar; secret=seo-agent-rag;     key=RAG_BEARER_TOKEN; restart="deploy/seo-agent -n jithendar (Recreate: check that no flow is running first)" ;;
  moks-sre-agent) ns=moks-sre;  secret=moks-rag-client;   key=RAG_BEARER_TOKEN; restart="the MOKS api, worker and crewai Deployments in namespace moks-sre" ;;
  open-webui)     ns=rag;       secret=rag-secrets;       key=OPEN_WEBUI_RAG_TOKEN; restart="" ;;
  rag-admin)      ns=rag;       secret=rag-admin-token;   key=token;            restart="" ;;
  *)
    echo "usage: $0 <moks-cms|moks-seo-agent|moks-sre-agent|open-webui|rag-admin>" >&2
    exit 2
    ;;
esac

for bin in kubectl python3 openssl; do
  command -v "$bin" >/dev/null || { echo "$bin is required" >&2; exit 1; }
done
if [[ -n "${KUBE_CONTEXT:-}" ]]; then
  current="$(kubectl config current-context)"
  [[ "$current" == "$KUBE_CONTEXT" ]] || { echo "current context is $current, not $KUBE_CONTEXT" >&2; exit 1; }
fi

token="$(openssl rand -hex 32)"

# Merge one key into a Secret (creating the Secret if needed) without touching its other keys.
put_secret_key() { # namespace secret key value
  local n="$1" s="$2" k="$3" v="$4" b64
  b64="$(printf '%s' "$v" | base64 | tr -d '\n')"
  if kubectl -n "$n" get secret "$s" >/dev/null 2>&1; then
    kubectl -n "$n" patch secret "$s" --type merge -p "{\"data\":{\"$k\":\"$b64\"}}" >/dev/null
  else
    kubectl -n "$n" create secret generic "$s" --from-literal="$k=$v" >/dev/null
  fi
}

# Every service's token is registered in RAG_SERVICE_TOKENS (rag-api reads only that and RAG_BEARER_TOKEN), Open
# WebUI's too. rag-secrets is read once and must be read: a failed read stops here, it never looks like "no tokens yet"
# (which would overwrite every other service's token).
secret_json="$(kubectl -n rag get secret rag-secrets -o json)"
merged="$(SECRET_JSON="$secret_json" SERVICE="$service" TOKEN="$token" python3 -c '
import base64, json, os
data = json.loads(os.environ["SECRET_JSON"]).get("data") or {}
raw = data.get("RAG_SERVICE_TOKENS")
try:
    tokens = json.loads(base64.b64decode(raw)) if raw else {}
except ValueError:
    raise SystemExit("rag-secrets RAG_SERVICE_TOKENS is not valid JSON; fix it by hand first")
if not isinstance(tokens, dict):
    raise SystemExit("rag-secrets RAG_SERVICE_TOKENS is not a JSON object; fix it by hand first")
tokens[os.environ["SERVICE"]] = os.environ["TOKEN"]
print(json.dumps(tokens, sort_keys=True))
')"
unset secret_json
put_secret_key rag rag-secrets RAG_SERVICE_TOKENS "$merged"
unset merged
put_secret_key "$ns" "$secret" "$key" "$token"
unset token
# seo-agent and MOKS read RAG_API_URL from the same Secret and treat rag as off without it.
case "$service" in
  moks-seo-agent|moks-sre-agent)
    if [[ -z "$(kubectl -n "$ns" get secret "$secret" -o jsonpath='{.data.RAG_API_URL}')" ]]; then
      put_secret_key "$ns" "$secret" RAG_API_URL "http://rag-api.rag.svc.cluster.local:8000"
    fi
    ;;
esac

kubectl -n rag rollout restart deploy/rag-api >/dev/null
kubectl -n rag rollout status deploy/rag-api --timeout=180s
if [[ "$service" == "open-webui" ]]; then
  kubectl -n rag rollout restart deploy/open-webui >/dev/null
  echo "Open WebUI restarted. If its connection was saved in Admin -> Settings -> Connections, the saved key"
  echo "wins over the environment: paste the new token there (kubectl -n rag get secret rag-secrets"
  echo "-o jsonpath='{.data.OPEN_WEBUI_RAG_TOKEN}' | base64 -d)."
fi

echo "Issued a token for $service into $ns/$secret ($key)."
[[ -n "$restart" ]] && echo "Restart $restart so it reads the new token."
echo "Check: rag-api logs show \"auth\": \"service-token\" and \"service\": \"$service\" for its requests."
