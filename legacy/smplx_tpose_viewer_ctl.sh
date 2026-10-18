#!/usr/bin/env bash
set -eu

PROJECT=/home/hgc/project/AnimeAvatarGen
PIDFILE=/tmp/anime_smplx_tpose_viewer.pid
LOGFILE=/tmp/smplx_tpose_viewer.log
PORT=8766

start() {
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "Already running, PID $(cat "$PIDFILE")"
    return
  fi
  set +u
  source /home/hgc/miniconda3/etc/profile.d/conda.sh
  conda activate mars39
  set -u
  cd "$PROJECT"
  nohup python -u tools/smplx_tpose_server.py --host 0.0.0.0 --port "$PORT" >"$LOGFILE" 2>&1 &
  echo $! >"$PIDFILE"
  sleep 1
  echo "Started PID $(cat "$PIDFILE")"
  echo "URL: http://100.95.255.117:$PORT"
}

stop() {
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    kill "$(cat "$PIDFILE")"
    echo "Stopped PID $(cat "$PIDFILE")"
  else
    echo "Not running"
  fi
  rm -f "$PIDFILE"
}

status() {
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then echo "Running PID $(cat "$PIDFILE")"; else echo "Stopped"; fi
}

case "${1:-status}" in
  start) start ;; stop) stop ;; restart) stop; start ;; status) status ;;
  *) echo "Usage: $0 {start|stop|restart|status}"; exit 1 ;;
esac
