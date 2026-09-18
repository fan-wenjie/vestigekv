#!/bin/bash
# P1 rerun, 4 arms (Base/Instruct x vestigekv/dense): bs=1, 4k prefill,
# continuous decode to 512k (README frozen rule). New protocol vs. the Sep-9
# published run: radix cache ON (production parity), vestigekv activation
# threshold 0 (compressed path at every length), --disable-custom-all-reduce
# (this box's CUDA P2P hangs), graph capture bs<=2.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=/home/user/.conda/envs/sglang/bin/python
export PYTHONPATH=$PWD/engine/python CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH NCCL_P2P_DISABLE=1
DATE=$(date +%F)
mkdir -p "runs/$DATE"

run_arm() {
  local model=$1 arm=$2 tag=$3
  local backend=triton
  [ "$arm" = vestigekv ] && backend=vestigekv_mla
  pkill -9 -f "sglang[.]launch_server" 2>/dev/null; sleep 3
  local log="runs/$DATE/server_p1_${tag}.log"
  echo "== ARM $tag: starting server ($backend) =="
  nohup $PY -m sglang.launch_server --model-path "$model" --trust-remote-code \
    --attention-backend "$backend" --tp-size 2 --context-length 524288 \
    --max-total-tokens 589824 --cuda-graph-max-bs 2 \
    --disable-custom-all-reduce --sampling-backend pytorch > "$log" 2>&1 &
  local spid=$!
  local ok=0
  for i in $(seq 1 90); do
    sleep 10
    if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:30000/health_generate 2>/dev/null | grep -q 200; then ok=1; break; fi
    if ! kill -0 $spid 2>/dev/null; then echo "ARM $tag: SERVER DIED"; tail -5 "$log"; return 1; fi
  done
  [ "$ok" = 1 ] || { echo "ARM $tag: health timeout"; return 1; }
  grep -o "max_total_num_tokens=[0-9]*" "$log" | head -1
  echo "== ARM $tag: bench =="
  $PY -m sglang.benchmark.serving --backend sglang --model "$model" \
    --num-prompts 1 --dataset-name random --random-input-len 4096 \
    --random-output-len 520192 --random-range-ratio 1 --max-concurrency 1 \
    --warmup-requests 0 --output-details \
    --output-file "results/latency_stream_4k-512k_${tag}.jsonl" \
    > "runs/$DATE/p1_${tag}.log" 2>&1
  echo "ARM $tag bench exit=$?"
  cp "$log" "results/latency_stream_serverlog_${tag}.log"
  pkill -9 -f "sglang[.]launch_server" 2>/dev/null; sleep 3
}

run_arm moonshotai/Kimi-Linear-48B-A3B-Base vestigekv vestigekv
run_arm moonshotai/Kimi-Linear-48B-A3B-Base dense dense
run_arm moonshotai/Kimi-Linear-48B-A3B-Instruct vestigekv vestigekv_instruct
run_arm moonshotai/Kimi-Linear-48B-A3B-Instruct dense dense_instruct
echo ALL_DONE
