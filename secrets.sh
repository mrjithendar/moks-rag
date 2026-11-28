# 1. Guard: the salt key must never be replaced once it exists
if kubectl -n rag get secret rag-secrets -o jsonpath='{.data.LITELLM_SALT_KEY}' | grep -q .; then
  echo "LITELLM_SALT_KEY already exists - stop, do not replace it"
else
  # 2. Add the three keys; the values go through stdin, never the command line
  read -r -p "LiteLLM admin user name (not 'admin'): " UIUSER
  SALT="sk-$(openssl rand -hex 32)"
  PASS="$(openssl rand -hex 24)"
  kubectl -n rag patch secret rag-secrets --type merge --patch-file /dev/stdin <<EOF
{"stringData": {"LITELLM_SALT_KEY": "$SALT", "LITELLM_UI_USERNAME": "$UIUSER", "LITELLM_UI_PASSWORD": "$PASS"}}
EOF
  unset SALT PASS UIUSER
fi

kubectl -n rag rollout status deploy/litellm --timeout=10m