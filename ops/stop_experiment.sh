#!/usr/bin/env bash
# Fully stop the experiment: public broadcast, viewer, WSL keepalive, and the
# Windows scheduled tasks that used to launch a visible PowerShell window.
# This does NOT delete participant credentials or experiment data.
set -uo pipefail

PROJECT_DIR="/home/lomerson/MyProjects/smplx_tpose_viewer_package"
PORT="${SMPLX_PORT:-8766}"
TAILSCALE="${TAILSCALE_EXE:-/mnt/d/ForProgrammingOnly/Tailscale/tailscale.exe}"
SCHTASKS="/mnt/c/Windows/System32/schtasks.exe"

echo "[stop] removing Windows scheduled tasks (popup source)"
for task in "SMPLX Viewer Health" "SMPLX Viewer Start"; do
  if [[ -x "$SCHTASKS" ]]; then
    "$SCHTASKS" /End /TN "$task" >/dev/null 2>&1 || true
    "$SCHTASKS" /Delete /TN "$task" /F >/dev/null 2>&1 || true
  fi
done

echo "[stop] stopping WSL sessions/processes"
for session in smplx-expt smplx-supervisor smplx; do
  tmux kill-session -t "$session" >/dev/null 2>&1 || true
done
pids="$(pgrep -f 'smplx_tpose_server.py' 2>/dev/null || true)"
if [[ -n "$pids" ]]; then
  # shellcheck disable=SC2086
  kill $pids >/dev/null 2>&1 || true
  sleep 1
  # shellcheck disable=SC2086
  kill -9 $pids >/dev/null 2>&1 || true
fi
# Also catch a server that inherited a different command line.
if command -v fuser >/dev/null 2>&1; then
  fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
fi

echo "[stop] removing Tailscale Serve/Funnel"
if [[ -x "$TAILSCALE" ]]; then
  "$TAILSCALE" funnel reset >/dev/null 2>&1 || true
  "$TAILSCALE" serve reset >/dev/null 2>&1 || true
fi

echo "[stop] remaining state"
if [[ -x "$TAILSCALE" ]]; then
  "$TAILSCALE" funnel status 2>&1 | sed 's/^/  funnel: /' | head -6 || true
  "$TAILSCALE" serve status 2>&1 | sed 's/^/  serve : /' | head -6 || true
fi
if ss -ltn 2>/dev/null | grep -q ":${PORT}\b"; then
  echo "[stop] WARNING: something is still listening on ${PORT}" >&2
  exit 1
fi
echo "[stop] experiment is down; no public broadcast and no Windows popup tasks"
