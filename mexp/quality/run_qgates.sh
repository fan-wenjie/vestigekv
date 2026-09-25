#!/bin/bash
# Quality gates: continuation agreement + serving needle, 4 cells
# (Base/Instruct x vestigekv/dense). One server per cell with ctx 131072,
# radix OFF (quality line), graph bs<=2, vk arm threshold 0. Client serial.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=/home/user/.conda/envs/sglang/bin/python
export PYTHONPATH=$PWD/engine/python CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH NCCL_P2P_DISABLE=1
DATE=$(date +%F)
mkdir -p "runs/$DATE"

run_cell() {
  local model=$1 arm=$2 tag=$3
  local backend=triton
  [ "$arm" = vestigekv ] && backend=vestigekv_mla
  pkill -9 -f "sglang[.]launch_server" 2>/dev/null; sleep 3
  local log="runs/$DATE/server_qgate_${tag}.log"
  echo "== CELL $tag: server ($backend, ctx 131072, radix off) =="
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

  echo "== CELL $tag: continuation agreement =="
  $PY mexp/quality/continuation_agreement.py gen \
    results/quality_cont64k_contexts_tokens.json \
    "results/quality_continuation_${tag}.json" > "runs/$DATE/qgate_cont_${tag}.log" 2>&1
  echo "CELL $tag continuation exit=$?"

  echo "== CELL $tag: serving needle =="
  $PY mexp/quality/needle_serving.py gen 131072 8 \
    "results/quality_needle_${tag}.json" > "runs/$DATE/qgate_needle_${tag}.log" 2>&1
  echo "CELL $tag needle exit=$?"

  pkill -9 -f "sglang[.]launch_server" 2>/dev/null; sleep 3
}

run_cell moonshotai/Kimi-Linear-48B-A3B-Base vestigekv vestigekv
run_cell moonshotai/Kimi-Linear-48B-A3B-Base dense dense
run_cell moonshotai/Kimi-Linear-48B-A3B-Instruct vestigekv vestigekv_instruct
run_cell moonshotai/Kimi-Linear-48B-A3B-Instruct dense dense_instruct

echo "== COMPARE =="
for sfx in "" "_instruct"; do
  $PY mexp/quality/continuation_agreement.py compare \
    "results/quality_continuation_vestigekv${sfx}.json" \
    "results/quality_continuation_dense${sfx}.json" \
    "results/quality_continuation_verdict${sfx}.json"
  $PY mexp/quality/needle_serving.py compare \
    "results/quality_needle_vestigekv${sfx}.json" \
    "results/quality_needle_dense${sfx}.json" \
    "results/quality_needle_verdict${sfx}.json"
done
echo ALL_DONE
