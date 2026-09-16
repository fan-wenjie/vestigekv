#!/usr/bin/env bash
# Shared launch prelude for Kimi-Linear-48B-A3B (bf16) on 2x RTX PRO 6000, engine branch
# vestigekv (the Kimi line). Quality/RULER line: CUDA graph ON (decode bs <= GRAPH_BS),
# radix cache OFF, fixed --random-seed, serial clients. GRAPH=0 turns the graph off.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export CUDA_HOME=/usr/local/cuda PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 HF_HUB_OFFLINE=1
export PYTHONPATH=${ENGINE:-$ROOT/engine}/python  # ENGINE=<checkout dir> serves another engine tree (bisection)
PY=${PYTHON:-$HOME/.conda/envs/sglang-dev/bin/python}
MODEL=${MODEL:-moonshotai/Kimi-Linear-48B-A3B-Instruct}
COMMON=(--model-path $MODEL --trust-remote-code --tp-size 2
  --context-length "${CTX:-73728}" --max-running-requests "${MAX_REQS:-4}"
  --max-mamba-cache-size "${MAMBA_SLOTS:-4}" --chunked-prefill-size "${CHUNK:-4096}"
  --disable-radix-cache --disable-custom-all-reduce --sampling-backend pytorch
  --random-seed "${SEED:-0}" --host 127.0.0.1 --port "${PORT:-30000}")
[ -n "${MEM_FRAC:-}" ] && COMMON+=(--mem-fraction-static "$MEM_FRAC")
if [ "${GRAPH:-1}" = 1 ]; then COMMON+=(--cuda-graph-max-bs-decode "${GRAPH_BS:-4}"); else COMMON+=(--disable-cuda-graph); fi
# Weight-cache daemon (mexp/kimi/weight_daemon.sh): when its ready files name live pids,
# the server loads the weights over CUDA IPC (WEIGHT_CACHE=off forces disk loading).
daemon_up() {
  local f pid
  for f in /tmp/sglang_weight_cache_*.ready; do
    [ -e "$f" ] || return 1
    pid=$(sed -n 's/^pid=//p' "$f"); [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null || return 1
  done
}
if [ "${WEIGHT_CACHE:-auto}" != off ] && daemon_up; then COMMON+=(--weight-cache-mode client); fi
guard() {
  # Without the daemon the GPUs must be empty; with it, only another server counts as busy.
  if daemon_up; then
    if pgrep -f "^[^ ]*python[^ ]* -m sglang.launch_server" > /dev/null; then echo "ABORT: a server is running"; exit 1; fi
  elif nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000{f=1} END{exit !f}'; then echo "ABORT: GPU busy"; exit 1; fi
  if ss -ltn | grep -q ":${PORT:-30000} "; then echo "ABORT: port ${PORT:-30000} busy"; exit 1; fi
}
