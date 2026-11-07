#!/usr/bin/env bash
# Kick one account (argument) or all active accounts (no argument).
set -euo pipefail
STATE_DIR="${HOME}/.config/smplx_experiment"
TOKEN_FILE="${SMPLX_ADMIN_TOKEN_FILE:-${STATE_DIR}/admin_token}"
PORT="${SMPLX_PORT:-8766}"
USERNAME="${1:-}"
if [[ ! -f "$TOKEN_FILE" ]]; then
  echo "admin token not found: $TOKEN_FILE" >&2
  exit 1
fi
TOKEN="$(cat "$TOKEN_FILE")"
if [[ -n "$USERNAME" ]]; then
  curl -fsS -X POST "http://127.0.0.1:${PORT}/api/admin/kick" \
    -H 'Content-Type: application/json' \
    -d "{\"token\":\"${TOKEN}\",\"username\":\"${USERNAME}\"}"
else
  curl -fsS -X POST "http://127.0.0.1:${PORT}/api/admin/reset" \
    -H 'Content-Type: application/json' \
    -d "{\"token\":\"${TOKEN}\"}"
fi
echo
