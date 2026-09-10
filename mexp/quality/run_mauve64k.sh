#!/bin/bash
# Long-context MAUVE (replacement for the retired continuation-agreement
# gate, ERRATA #20): T=65536 prefill, 256-token generations, 16 fineweb-edu
# contexts, per-context sampling_seed shared across arms. 4 cells
# (Base/Instruct x vestigekv/dense), radix OFF, serial, then GPU scoring.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=/home/user/.conda/envs/sglang/bin/python
export PYTHONPATH=$PWD/engine/python CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH NCCL_P2P_DISABLE=1 M7_T=65536 M7_N=64
DATE=$(date +%F)
mkdir -p "runs/$DATE"
CTX=results/quality_mauve64k64_ctx.json

run_cell() {
  local model=$1 arm=$2 tag=$3
  local backend=triton
  [ "$arm" = vestigekv ] && backend=vestigekv_mla
  pkill -9 -f "sglang[.]launch_server" 2>/dev/null; sleep 3
  local log="runs/$DATE/server_mauve64k_${tag}.log"
  if [ "$arm" = vestigekv ]; then
    export SGLANG_VESTIGEKV_ACTIVATION_MIN_TOKENS=0
  else
    unset SGLANG_VESTIGEKV_ACTIVATION_MIN_TOKENS
  fi
  echo "== CELL $tag: server ($backend) =="
  nohup $PY -m sglang.launch_server --model-path "$model" --trust-remote-code \
    --attention-backend "$backend" --tp-size 2 --context-length 139264 \
    --cuda-graph-max-bs 2 --disable-radix-cache \
    --disable-custom-all-reduce --sampling-backend pytorch > "$log" 2>&1 &
  local spid=$!
  local ok=0
  for i in $(seq 1 90); do
    sleep 10
    if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:30000/health_generate 2>/dev/null | grep -q 200; then ok=1; break; fi
    if ! kill -0 $spid 2>/dev/null; then echo "CELL $tag: SERVER DIED"; tail -5 "$log"; return 1; fi
  done
  [ "$ok" = 1 ] || { echo "CELL $tag: health timeout"; return 1; }
  echo "== CELL $tag: MAUVE 64k gen =="
  $PY mexp/m7_mauve_serving.py gen "$CTX" \
    "results/quality_mauve64k64_${tag}.json" > "runs/$DATE/mauve64k64_${tag}.log" 2>&1
  echo "CELL $tag gen exit=$?"
  pkill -9 -f "sglang[.]launch_server" 2>/dev/null; sleep 3
}

run_cell moonshotai/Kimi-Linear-48B-A3B-Base vestigekv vestigekv
run_cell moonshotai/Kimi-Linear-48B-A3B-Base dense dense
run_cell moonshotai/Kimi-Linear-48B-A3B-Instruct vestigekv vestigekv_instruct
run_cell moonshotai/Kimi-Linear-48B-A3B-Instruct dense dense_instruct

echo "== SCORE =="
for sfx in "" "_instruct"; do
  $PY mexp/m7_mauve_serving.py score "$CTX" \
    "results/quality_mauve64k64_vestigekv${sfx}.json" \
    "results/quality_mauve64k64_dense${sfx}.json" \
    "results/quality_mauve64k64_verdict${sfx}.json"
done
echo ALL_DONE
