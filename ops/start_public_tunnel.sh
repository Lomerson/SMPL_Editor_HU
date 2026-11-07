#!/usr/bin/env bash
# Start a Cloudflare Quick Tunnel for the local viewer (no domain required).
set -euo pipefail
PROJECT_DIR="/home/lomerson/MyProjects/smplx_tpose_viewer_package"
PORT="${SMPLX_PORT:-8766}"
BIN="${CLOUDFLARED_BIN:-$HOME/.local/bin/cloudflared}"
STATE_DIR="${HOME}/.config/smplx_experiment"
LOG="${STATE_DIR}/cloudflared.log"
URL_FILE="${STATE_DIR}/public_url.txt"
SESSION="smplx-tunnel"

if [[ ! -x "$BIN" ]]; then
  echo "cloudflared not found: $BIN" >&2
  echo "Install: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/" >&2
  exit 1
fi
if ! curl -fsS --max-time 3 "http://127.0.0.1:${PORT}/healthz" >/dev/null 2>&1; then
  echo "viewer is not healthy on 127.0.0.1:${PORT}" >&2
  exit 1
fi

"$PROJECT_DIR/ops/stop_public_tunnel.sh" >/dev/null 2>&1 || true
mkdir -p "$STATE_DIR"
: > "$LOG"
rm -f "$URL_FILE"
tmux new-session -d -s "$SESSION" \
  "'$BIN' tunnel --url 'http://127.0.0.1:${PORT}' --no-autoupdate >>'$LOG' 2>&1"

for _ in $(seq 1 30); do
  url="$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG" 2>/dev/null | tail -1 || true)"
  if [[ -n "$url" ]]; then
    printf '%s\n' "$url" > "$URL_FILE"
    echo "[tunnel] $url"
    exit 0
  fi
  sleep 1
done
echo "[tunnel] no URL after 30s; see $LOG" >&2
"$PROJECT_DIR/ops/stop_public_tunnel.sh" >/dev/null 2>&1 || true
exit 1
