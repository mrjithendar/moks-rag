#!/usr/bin/env bash
# Create jithendar.com's LiteLLM team and its virtual keys, and write the keys into the site's Secrets (moks-cms D-109,
# moks-cms docs/LLM_GATEWAY.md; README "LiteLLM as jithendar.com's gateway").
#
#   scripts/issue-litellm-keys.sh            # create the team and the keys that are missing
#   scripts/issue-litellm-keys.sh --rotate   # replace every key (the old ones stop working at once)
#
# Team "jithendar.com", models moks-cms-default and moks-cms-fast. One key per caller, each in the Secret that caller
# reads:
#   jithendar-agent    → jithendar/seo-agent-env       LITELLM_API_KEY           the SEO agent's flows and crews
#   jithendar-ask      → jithendar/seo-agent-env       LITELLM_ASK_API_KEY       Ask the Blog (readers)
#   jithendar-web      → jithendar/jithendar-com-env   LITELLM_API_KEY           the web app's drafts and enrichment
#   jithendar-practice → jithendar/jithendar-com-env   LITELLM_PRACTICE_API_KEY  the Python practice tutor
# No budget is set (nothing is preset): set the team's and each key's in the Admin UI (https://jithendar.com/llm/ui →
# Teams / Virtual Keys). A key that exists is kept (its value cannot be read back) unless --rotate is given.
# Keys are never printed: they go from LiteLLM's answer to `kubectl patch` through stdin.
#
# Uses your current kubectl context (KUBE_CONTEXT=<name> refuses any other) and reaches LiteLLM through a port-forward,
# with the master key read from Secret rag/rag-secrets. For a test against another LiteLLM: LITELLM_URL and
# LITELLM_MASTER_KEY, and DRY_SECRETS=1 to name the Secret keys it would write instead of writing them.
set -euo pipefail

rotate=0
case "${1:-}" in
  "") ;;
  --rotate) rotate=1 ;;
  *) echo "usage: $0 [--rotate]" >&2; exit 2 ;;
esac
for bin in python3 curl; do command -v "$bin" >/dev/null || { echo "$bin is required" >&2; exit 1; }; done

url="${LITELLM_URL:-}"
master="${LITELLM_MASTER_KEY:-}"
if [[ -z "$url" ]]; then
  command -v kubectl >/dev/null || { echo "kubectl is required" >&2; exit 1; }
  if [[ -n "${KUBE_CONTEXT:-}" ]]; then
    current="$(kubectl config current-context)"
    [[ "$current" == "$KUBE_CONTEXT" ]] || { echo "current context is $current, not $KUBE_CONTEXT" >&2; exit 1; }
  fi
  master="$(kubectl -n rag get secret rag-secrets -o jsonpath='{.data.LITELLM_MASTER_KEY}' | base64 -d)"
  [[ -n "$master" ]] || { echo "rag/rag-secrets has no LITELLM_MASTER_KEY" >&2; exit 1; }
  port=14000
  kubectl -n rag port-forward svc/litellm "$port:4000" >/dev/null 2>&1 &
  pf=$!
  trap 'kill "$pf" 2>/dev/null || true' EXIT
  url="http://127.0.0.1:$port"
  for _ in $(seq 1 30); do
    curl -fsS -o /dev/null "$url/health/liveliness" 2>/dev/null && break
    sleep 1
  done
fi
[[ -n "$master" ]] || { echo "LITELLM_MASTER_KEY is required with LITELLM_URL" >&2; exit 1; }

# Team and keys through LiteLLM's API; stdout: one line per Secret, "<secret>\t<JSON patch>" (holds new keys only).
patches="$(LITELLM_URL="$url" MASTER="$master" ROTATE="$rotate" python3 - <<'PY'
import json, os, sys, urllib.parse, urllib.request

URL, MASTER, ROTATE = os.environ["LITELLM_URL"].rstrip("/"), os.environ["MASTER"], os.environ["ROTATE"] == "1"
TEAM, MODELS = "jithendar.com", ["moks-cms-default", "moks-cms-fast"]
KEYS = [  # alias, Secret, Secret key
    ("jithendar-agent", "seo-agent-env", "LITELLM_API_KEY"),
    ("jithendar-ask", "seo-agent-env", "LITELLM_ASK_API_KEY"),
    ("jithendar-web", "jithendar-com-env", "LITELLM_API_KEY"),
    ("jithendar-practice", "jithendar-com-env", "LITELLM_PRACTICE_API_KEY"),
]

def call(method, path, body=None, query=None):
    q = f"?{urllib.parse.urlencode(query)}" if query else ""
    req = urllib.request.Request(f"{URL}{path}{q}", method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": f"Bearer {MASTER}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        sys.exit(f"LiteLLM {method} {path}: HTTP {e.code} {e.read()[:300].decode(errors='replace')}")

teams = call("GET", "/team/list")
teams = teams if isinstance(teams, list) else teams.get("teams", [])
team = next((t for t in teams if t.get("team_alias") == TEAM), None)
if team is None:
    team = call("POST", "/team/new", {"team_alias": TEAM, "models": MODELS, "metadata": {"app": "jithendar.com", "decision": "moks-cms D-109"}})
    print(f"created team {TEAM}", file=sys.stderr)
else:
    print(f"team {TEAM} exists", file=sys.stderr)
team_id = team["team_id"]

patches = {}
for alias, secret, field in KEYS:
    found = call("GET", "/key/list", query={"key_alias": alias, "return_full_object": "true"}).get("keys") or []
    if found and not ROTATE:
        print(f"key {alias} exists: kept (--rotate replaces it)", file=sys.stderr)
        continue
    if found:
        call("POST", "/key/delete", {"key_aliases": [alias]})
    caller = alias.split("-", 1)[1]
    key = call("POST", "/key/generate", {"key_alias": alias, "team_id": team_id, "models": MODELS, "metadata": {"app": "jithendar.com", "caller": caller}})["key"]
    patches.setdefault(secret, {})[field] = key
    print(f"key {alias}: {'rotated' if found else 'created'} → {secret} {field}", file=sys.stderr)
for secret, fields in patches.items():
    print(f"{secret}\t{json.dumps({'stringData': fields})}")
PY
)"
unset master

if [[ -z "$patches" ]]; then
  echo "Nothing to write: every key exists. Budgets: https://jithendar.com/llm/ui → Teams / Virtual Keys."
  exit 0
fi
while IFS=$'\t' read -r secret patch; do
  if [[ "${DRY_SECRETS:-}" == "1" ]]; then
    printf 'would write %s: %s\n' "$secret" "$(PATCH="$patch" python3 -c 'import json,os; print(", ".join(json.loads(os.environ["PATCH"])["stringData"]))')"
  else
    printf '%s' "$patch" | kubectl -n jithendar patch secret "$secret" --type merge --patch-file /dev/stdin >/dev/null
    echo "wrote jithendar/$secret"
  fi
done <<< "$patches"
unset patches
echo "Restart the site so both containers read their keys: kubectl -n jithendar rollout restart deploy/jithendar-com"
echo "Then set the budgets in LiteLLM (team jithendar.com, and Ask the Blog's key at least): https://jithendar.com/llm/ui"
