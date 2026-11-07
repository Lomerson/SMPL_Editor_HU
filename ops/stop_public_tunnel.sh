#!/usr/bin/env bash
# Stop the Cloudflare Quick Tunnel and remove its published URL.
set -uo pipefail
PORT="${SMPLX_PORT:-8766}"
STATE_DIR="${HOME}/.config/smplx_experiment"
URL_FILE="${STATE_DIR}/public_url.txt"
tmux kill-session -t smplx-tunnel >/dev/null 2>&1 || true
# Only match the quick tunnel for our local port; do not kill other tunnels.
pids="$(pgrep -f "cloudflared.*127\.0\.0\.1:${PORT}" 2>/dev/null || true)"
if [[ -n "$pids" ]]; then
  # shellcheck disable=SC2086
  kill $pids >/dev/null 2>&1 || true
  sleep 1
  # shellcheck disable=SC2086
  kill -9 $pids >/dev/null 2>&1 || true
fi
rm -f "$URL_FILE"
echo "[tunnel] stopped"
