#!/usr/bin/env bash
# PREREG: the bs=1 latency curve, both arms, one tree, one sitting, server log kept.
#
# WHAT IS WRONG TODAY. The paper's two headline speed numbers -- 1.28x at 256k
# and 1.53x at 508k -- are hand-maintained values whose comments name
# "server-side gen-throughput logs". Those logs are in no surviving tree:
# results/README.md records that results/kimi/server_vestigekv_*.log "were
# never tracked by git and were deleted in this reduction". So the numbers
# cannot be re-derived from anything we ship.
#
# Worse, what we DO ship contradicts them. Reducing the surviving client-side
# JSONL (latency_stream_4k-512k_{dense,vestigekv}.jsonl) gives dense/vk window
# medians of 1.049 at 64k, 1.134 at 128k, 0.979 at 256k and 0.066 at 508k --
# the last meaning VestigeKV 15x SLOWER exactly where the abstract claims
# 1.53x faster. The project's own ERRATA (runs/2026-09-11, item 19) explains
# why: above ~80-180 tok/s the streaming client's ITL is contaminated and the
# contamination penalises the faster arm. But that errata is not in
# results.zip either, so a reviewer who unzips the archive finds the
# contradiction and no explanation of it.
#
# And the vk arm at 512k has no matched dense partner at all: queue.jsonl
# contains exactly one job past 300k output (td-stream-512k, arm=vestigekv, on
# engine-fused), and no baseline job at that length. The dense 7.120 ms/token
# the macro names has no identified producing run.
#
# WHAT THIS RUN FIXES. Both arms, back to back, on the RELEASED tree, with the
# server log retained rather than reduced away. That gives the figure a pair
# that (a) exists, (b) shares a tree, (c) ships, and (d) is measured by the
# instrument the paper already declares authoritative for this metric.
#
# CONTROL:
#   varies:  the attention backend, and nothing else -- triton (the wrapped
#            base, which is the paper's dense arm) against vestigekv_mla.
#   fixed:   tree $HOME/vestigekv/engine (released, v0.5.20 + the backend);
#            4096-token prefill; 520192 decoded tokens; bs=1; max-concurrency
#            1; no warmup request; CTX, MAX_REQS, MAMBA_SLOTS, CHUNK, GRAPH_BS;
#            one script, one sitting, no server reuse between arms.
#
# HOW TO READ IT. The client JSONL is written too, but it is NOT the metric:
# see the errata above. The server log is. Whatever ratio this produces is the
# number the paper prints, including if it is below 1.53x -- the point of the
# run is that the printed number and the shipped record agree.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

MODEL="moonshotai/Kimi-Linear-48B-A3B-Instruct"
OUTLEN=520192
INLEN=4096

require_free_gpu
for arm in dense vestigekv; do
  refuse_overwrite "$ROOT/results/kimi/serverlog_paired512k_${arm}.log"
done
cd "$ROOT"

export CTX=528384 MAX_REQS=1 MAMBA_SLOTS=1 CHUNK=4096 GRAPH_BS=2
export ENGINE="$ROOT/engine"
[ -d "$ENGINE/python/sglang" ] || die "released engine tree missing: $ENGINE"

run_arm() {   # arm, launcher script
  local arm="$1" launcher="$2"
  local srv="$ROOT/results/kimi/serverlog_paired512k_${arm}.log"
  local out="$ROOT/results/kimi/latency_paired512k_${arm}.jsonl"
  banner "latency pair: ${arm} (${launcher})"
  bash "mexp/kimi/${launcher}" > "$srv" 2>&1 &
  local pid=$!
  trap 'kill -TERM '"$pid"' 2>/dev/null || true; pkill -f "[s]glang.launch_server" || true' EXIT
  local up=0
  for _ in $(seq 480); do
    sleep 5
    if grep -q "fired up" "$srv"; then up=1; break; fi
    kill -0 $pid 2>/dev/null || break
  done
  [ "$up" = 1 ] || die "server did not come up for ${arm}; see $srv"
  grep -m1 "VestigeKV:" "$srv" || true

  env OPENAI_API_KEY=dummy PYTHONPATH="$ENGINE/python" CUDA_VISIBLE_DEVICES="" \
    "$PY" -m sglang.benchmark.serving --backend sglang --model "$MODEL" \
      --port 30000 --num-prompts 1 --dataset-name random \
      --random-input-len "$INLEN" --random-output-len "$OUTLEN" \
      --random-range-ratio 1 --max-concurrency 1 --warmup-requests 0 \
      --output-details --output-file "$out" \
    2>&1 | tail -30
  kill -TERM $pid 2>/dev/null || true
  pkill -f "[s]glang.launch_server" || true
  sleep 30
  trap - EXIT
  # the server log is the metric and it is what went missing last time
  echo "kept $(wc -l < "$srv") server log lines for ${arm}"
}

run_arm dense     baseline.sh
run_arm vestigekv vestigekv.sh

echo "wrote results/kimi/serverlog_paired512k_{dense,vestigekv}.log"
echo "next: mexp/kimi/make_serving_speed_numbers.py reduces them to the macros"
