#!/usr/bin/env bash
# Start the experimental viewer in WSL if it is not already healthy.
set -euo pipefail

PROJECT_DIR="/home/lomerson/MyProjects/smplx_tpose_viewer_package"
PORT="${SMPLX_PORT:-8766}"
STATE_DIR="${HOME}/.config/smplx_experiment"
LOG="${STATE_DIR}/server.log"
mkdir -p "$STATE_DIR"

if curl -fsS --max-time 2 "http://127.0.0.1:${PORT}/healthz" >/dev/null 2>&1; then
  exit 0
fi

cd "$PROJECT_DIR"
export SMPLX_BIND_HOST=127.0.0.1
export SMPLX_PORT="$PORT"
export SMPLX_EXPERIMENT=1
export SMPLX_COOKIE_SECURE="${SMPLX_COOKIE_SECURE:-1}"
export SMPLX_ACCESS_FILE="${SMPLX_ACCESS_FILE:-${STATE_DIR}/participants.json}"
if [[ -f "${STATE_DIR}/admin_token" ]]; then
  export SMPLX_ADMIN_TOKEN="$(cat "${STATE_DIR}/admin_token")"
fi

nohup ./run_local.sh >>"$LOG" 2>&1 &
for _ in $(seq 1 40); do
  if curl -fsS --max-time 2 "http://127.0.0.1:${PORT}/healthz" >/dev/null 2>&1; then
    exit 0
  fi
  sleep 1
done
echo "viewer did not become healthy; see $LOG" >&2
exit 1
