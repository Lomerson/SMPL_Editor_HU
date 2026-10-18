#!/usr/bin/env bash
# Local launcher for this machine (WSL Ubuntu, conda env "torch3d").
# It does NOT modify the original scripts, which are hardcoded to the author's
# machine (/home/hgc/..., conda env "mars39"). All paths can be overridden by
# environment variables:
#   SMPLX_MODEL_DIR  directory containing SMPLX_{NEUTRAL,MALE,FEMALE}.npz
#   SMPLX_CACHE_DIR  directory for generated T-pose OBJ files
#   SMPLX_BIND_HOST / SMPLX_PORT
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SMPLX_MODEL_DIR:-$(cd "$PROJECT_DIR/.." && pwd)/models/smplx}"
CACHE_DIR="${SMPLX_CACHE_DIR:-$PROJECT_DIR/outputs/smplx_tpose_viewer}"
BIND_HOST="${SMPLX_BIND_HOST:-127.0.0.1}"
BIND_PORT="${SMPLX_PORT:-8766}"

CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
# conda's activate.d scripts touch LD_LIBRARY_PATH, which is unset in a clean
# shell, so nounset has to be relaxed around the activation.
set +u
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate torch3d
set -u

export PYTHONNOUSERSITE=1
echo "python : $(which python)"
echo "models : $MODEL_DIR"
echo "cache  : $CACHE_DIR"
echo "url    : http://$BIND_HOST:$BIND_PORT"

exec python -u "$PROJECT_DIR/smplx_tpose_server.py" \
  --model-dir "$MODEL_DIR" \
  --cache-dir "$CACHE_DIR" \
  --host "$BIND_HOST" \
  --port "$BIND_PORT" \
  "$@"
