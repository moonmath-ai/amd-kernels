#!/usr/bin/env bash
# Hardware counter sample for libattention.so — run on the MI300 host with ROCm.
# Usage (from repo root):
#   ./tools/rocprof_attn.sh
# Tweak ROCPROF_OPTS for your ROCm version (see `rocprof --help`).

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

: "${ROCPROF_OPTS:=--stats}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"

if ! command -v rocprof >/dev/null 2>&1; then
  echo "rocprof not in PATH; load ROCm (e.g. source /opt/rocm*/bin/rocprof-env or use module load rocm)." >&2
  exit 1
fi

make -s all
exec rocprof ${ROCPROF_OPTS} python3 runner.py --warmup-iters 2 --benchmark-iters 5 \
  --batch 2 --heads 24 --seq-len 8192 --head-dim 128
