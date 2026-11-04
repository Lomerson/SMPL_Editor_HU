#!/usr/bin/env bash
# Force-release the current participant lease from the local viewer.
set -euo pipefail
STATE_DIR="${HOME}/.config/smplx_experiment"
TOKEN_FILE="${SMPLX_ADMIN_TOKEN_FILE:-${STATE_DIR}/admin_token}"
PORT="${SMPLX_PORT:-8766}"
if [[ ! -f "$TOKEN_FILE" ]]; then
  echo "admin token not found: $TOKEN_FILE" >&2
  exit 1
fi
TOKEN="$(cat "$TOKEN_FILE")"
curl -fsS -X POST "http://127.0.0.1:${PORT}/api/admin/reset" \
  -H 'Content-Type: application/json' \
  -d "{\"token\":\"${TOKEN}\"}"
echo
