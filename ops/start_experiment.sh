#!/usr/bin/env bash
# Start the experiment from a clean state, with a WSL-only keepalive
# (a tmux supervisor loop, not a visible Windows scheduled task).
set -euo pipefail

PROJECT_DIR="/home/lomerson/MyProjects/smplx_tpose_viewer_package"
STATE_DIR="${HOME}/.config/smplx_experiment"
PORT="${SMPLX_PORT:-8766}"
TAILSCALE="${TAILSCALE_EXE:-/mnt/d/ForProgrammingOnly/Tailscale/tailscale.exe}"
LOG="${STATE_DIR}/server.log"
SESSION="smplx-expt"
MAX_ACTIVE="${SMPLX_MAX_ACTIVE:-5}"
TORCH_THREADS="${SMPLX_TORCH_THREADS:-4}"

cd "$PROJECT_DIR"
echo "[start] step 1/6: clean previous experiment"
"$PROJECT_DIR/ops/stop_experiment.sh"

echo "[start] step 2/6: check credentials and tools"
[[ -f "${STATE_DIR}/participants.json" ]] || {
  echo "missing ${STATE_DIR}/participants.json; generate accounts first" >&2
  exit 1
}
[[ -f "${STATE_DIR}/admin_token" ]] || {
  echo "missing ${STATE_DIR}/admin_token" >&2
  exit 1
}
command -v tmux >/dev/null || { echo "tmux is required" >&2; exit 1; }
[[ -x "$TAILSCALE" ]] || { echo "Tailscale CLI not found: $TAILSCALE" >&2; exit 1; }
mkdir -p "$STATE_DIR"
touch "$LOG"

echo "[start] step 3/6: start WSL viewer + supervisor (no Windows task)"
# The loop is the WSL-side keepalive. It survives a server crash; the
# supervisor itself is stopped by stop_experiment.sh.
tmux new-session -d -s "$SESSION" \
  "cd '$PROJECT_DIR' && export SMPLX_EXPERIMENT=1 SMPLX_ACCESS_FILE='${STATE_DIR}/participants.json' SMPLX_COOKIE_SECURE=1 SMPLX_ADMIN_TOKEN=\"\$(cat '${STATE_DIR}/admin_token')\" SMPLX_BIND_HOST=127.0.0.1 SMPLX_PORT=${PORT} SMPLX_MAX_ACTIVE=${MAX_ACTIVE} SMPLX_TORCH_THREADS=${TORCH_THREADS} SMPLX_HTTP_KEEPALIVE=1 SMPLX_GZIP=1 SMPLX_MAX_CONCURRENT_AUTOFIT=1; while true; do ./run_local.sh >>'${LOG}' 2>&1; echo \"[supervisor] server exited at \$(date -Is); restarting\" >>'${LOG}'; sleep 2; done"
for _ in $(seq 1 40); do
  if curl -fsS --max-time 2 "http://127.0.0.1:${PORT}/healthz" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
curl -fsS --max-time 3 "http://127.0.0.1:${PORT}/healthz" >/dev/null || {
  echo "viewer did not become healthy; see ${LOG}" >&2
  "$PROJECT_DIR/ops/stop_experiment.sh" >/dev/null 2>&1 || true
  exit 1
}
tmux has-session -t "$SESSION"

echo "[start] step 4/6: check popup tasks are gone"
if /mnt/c/Windows/System32/schtasks.exe /Query /TN "SMPLX Viewer Health" >/dev/null 2>&1; then
  echo "SMPLX Viewer Health task still exists; removing it" >&2
  /mnt/c/Windows/System32/schtasks.exe /End /TN "SMPLX Viewer Health" >/dev/null 2>&1 || true
  /mnt/c/Windows/System32/schtasks.exe /Delete /TN "SMPLX Viewer Health" /F >/dev/null 2>&1 || true
fi
if /mnt/c/Windows/System32/schtasks.exe /Query /TN "SMPLX Viewer Start" >/dev/null 2>&1; then
  echo "SMPLX Viewer Start task still exists; removing it" >&2
  /mnt/c/Windows/System32/schtasks.exe /End /TN "SMPLX Viewer Start" >/dev/null 2>&1 || true
  /mnt/c/Windows/System32/schtasks.exe /Delete /TN "SMPLX Viewer Start" /F >/dev/null 2>&1 || true
fi

echo "[start] step 5/6: verify Funnel capability and apply public broadcast"
funnel_status="$("$TAILSCALE" funnel status 2>&1 || true)"
if grep -q "Funnel on" <<<"$funnel_status"; then
  echo "[start] Funnel already on"
else
  funnel_output="$(timeout 15 "$TAILSCALE" funnel --bg "$PORT" 2>&1 || true)"
  if grep -q "not enabled" <<<"$funnel_output"; then
    echo "[start] Funnel permission is OFF on this tailnet." >&2
    echo "$funnel_output" | sed 's/^/  /' >&2
    echo "[start] Enable Funnel at https://console.tailscale.com/admin/dns" >&2
    echo "[start] or run 'tailscale funnel --bg ${PORT}' in a terminal and open the printed link." >&2
    echo "[start] stopping server; then rerun this script." >&2
    "$PROJECT_DIR/ops/stop_experiment.sh" >/dev/null 2>&1 || true
    exit 1
  fi
  # The CLI can print "Funnel started" before status catches up; accept either
  # the success banner or a short poll of status.
  if ! grep -qE "Funnel started|Available on the internet|Funnel on" <<<"$funnel_output"; then
    for _ in $(seq 1 5); do
      if "$TAILSCALE" funnel status 2>&1 | grep -q "Funnel on"; then
        break
      fi
      sleep 1
    done
  fi
  if ! grep -qE "Funnel started|Available on the internet|Funnel on" <<<"$funnel_output" \
     && ! "$TAILSCALE" funnel status 2>&1 | grep -q "Funnel on"; then
    echo "[start] could not confirm Funnel is on." >&2
    echo "$funnel_output" | sed 's/^/  /' >&2
    "$PROJECT_DIR/ops/stop_experiment.sh" >/dev/null 2>&1 || true
    exit 1
  fi
fi

echo "[start] step 6/6: verify local and public state"
curl -fsS --max-time 3 "http://127.0.0.1:${PORT}/healthz" >/dev/null
"$TAILSCALE" funnel status 2>&1 | sed 's/^/  /'
echo
echo "[start] experiment is up (login required, WSL supervisor running)"
echo "[start] public: https://$( "$TAILSCALE" status --json 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))' 2>/dev/null || echo 'see funnel status' )"
echo "[start] admin : same URL + /admin"
echo "[start] stop  : $PROJECT_DIR/ops/stop_experiment.sh"
