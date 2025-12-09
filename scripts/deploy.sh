#!/bin/bash
set -euo pipefail

if [ -z "${DEPLOY_HOST:-}" ] || [ -z "${DEPLOY_USER:-}" ] || [ -z "${DEPLOY_PATH:-}" ]; then
  echo "DEPLOY_HOST, DEPLOY_USER and DEPLOY_PATH must be set" >&2
  exit 1
fi

ssh_cmd=(ssh)
if [ -n "${DEPLOY_KEY:-}" ]; then
  ssh_cmd+=( -i "$DEPLOY_KEY" )
fi

"${ssh_cmd[@]}" "$DEPLOY_USER@$DEPLOY_HOST" <<'REMOTE'
cd "$DEPLOY_PATH"
if [ "${LOCAL_BUILD:-false}" = "true" ]; then
  docker compose build
else
  docker compose pull
fi
docker compose up -d
REMOTE

