#!/usr/bin/env bash
# PREREG: the control arm for the two no-fallback RULER runs.
#
# The review asks how much of the RULER accuracy is sustained by the dense
# fallback rather than by the sparse trigger. Two arms that answer it already
# exist and are matched to each other -- trunc-prefix-nofb and trunc-spread-nofb,
# both launched with --disable-vestigekv-recall-overflow-fallback on the
# engine-fused tree, seed 0, n=10, three tasks, three lengths:
#
#     niah_multikey_2, niah_multikey_3, ruler_qa_hotpot at 16k/32k/65k
#
# What is missing is their control. The obvious candidate on disk,
# results_vestigekv_n10_16384-32768-65536.json, is UNTAGGED: no queue entry in
# any surviving backup produced it, so the tree and flags it ran under are
# unknown. Comparing a no-fallback arm against a run of unknown provenance is
# the arm-mismatch defect this project has already shipped once, in the
# dual-arm RoPE table. So the control is re-run here, deliberately, with the
# fallback ON -- which is the default, hence no server flag -- and every other
# knob copied from the trunc-prefix-nofb queue line character for character.
#
# The launch is a hand-rolled copy of what mexp/glm53/queue_runner.py does for
# that job, because the runner reads a shared queue that currently holds
# eleven undrained jobs, several of them multi-hour. Nothing here writes to
# that queue or its state file.
#
# Read the result as a three-way comparison at n=10 on the three affected
# tasks only. It is NOT the n=50 grid the paper's RULER table reports, and the
# cells are 10 prompts wide, so single cells move by 0.1 at a time. The
# direction across nine cells is what the comparison can support.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

TAG="fallback-on-control"
OUT="results/kimi/ruler/results_vestigekv_n10_16384-32768-65536_${TAG}.json"
SRV_LOG="$ROOT/results/kimi/server_vestigekv_${TAG}.log"
CLI_LOG="$ROOT/results/kimi/ruler_vestigekv_${TAG}.log"
MODEL="moonshotai/Kimi-Linear-48B-A3B-Instruct"

require_free_gpu
refuse_overwrite "$ROOT/$OUT"
cd "$ROOT"

# env: the trunc-prefix-nofb line, minus nothing.
export CTX=73728 MAX_REQS=4 MAMBA_SLOTS=4 CHUNK=4096 GRAPH_BS=4
export ENGINE="$HOME/vestigekv-wt/engine-fused"
export SGLANG_DEBUG_VESTIGEKV_STATS=1
[ -d "$ENGINE/python/sglang" ] || die "engine-fused missing: $ENGINE"

banner "RULER fallback-ON control, 3 tasks x 3 lengths, n=10, seed 0"
bash mexp/kimi/vestigekv.sh > "$SRV_LOG" 2>&1 &
SRV=$!
trap 'kill -TERM $SRV 2>/dev/null || true; pkill -f "[s]glang.launch_server" || true' EXIT

for _ in $(seq 360); do
  sleep 5
  grep -q "fired up" "$SRV_LOG" && break
  kill -0 $SRV 2>/dev/null || die "server died; see $SRV_LOG"
done
grep -q "fired up" "$SRV_LOG" || die "server never came up; see $SRV_LOG"
grep -m1 "VestigeKV:" "$SRV_LOG" || true

env OPENAI_API_KEY=dummy PYTHONPATH="$ROOT/engine/python" CUDA_VISIBLE_DEVICES="" \
  "$PY" mexp/glm53/run_ruler.py \
    --arm vestigekv --port 30000 --model "$MODEL" \
    --n 10 --out "$ROOT/results/kimi/ruler" \
    --lengths 16384,32768,65536 \
    --tasks niah_multikey_2,niah_multikey_3,ruler_qa_hotpot \
    --tag "$TAG" 2>&1 | tee "$CLI_LOG"

echo "wrote $OUT"
