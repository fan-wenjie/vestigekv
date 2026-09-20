#!/usr/bin/env bash
# PREREG: what the overflow fallback contributes, measured by deleting it.
#
# The review asks how much of the RULER accuracy is sustained by the dense
# fallback rather than by the sparse trigger. The answer needs three arms that
# differ in one thing, so all three are run here, back to back, from one
# script -- the previous attempt at this comparison had two matched arms and a
# control of unknown provenance, which is the defect that cost this paper its
# dual-arm RoPE table.
#
#   fb-on          the deployed algorithm. Nothing set.
#   fb-off-prefix  SGLANG_DEBUG_VESTIGEKV_NO_OVERFLOW_FALLBACK=1. On overflow,
#                  attend the first W fired rows instead of the full row set.
#   fb-off-spread  the same, plus SGLANG_DEBUG_VESTIGEKV_SPREAD_TRUNCATE=1,
#                  which keeps the truncated rows spread over the archive
#                  rather than taking a prefix of it.
#
# WHY THE ABLATION IS NOT A CONFIGURATION. The admission rule is uncapped: a
# row fires iff its certified score beats its own head's kept maximum. The
# 4096-row buffer is a CUDA-graph width sitting BEHIND that rule, not an
# admission cap. So an overflow is not a miss -- it is the certificate
# reporting that this step cannot be served sparsely at the recall target, and
# the fallback is the method's answer to that report. Deleting it does not buy
# a cheaper configuration; it leaves an incomplete algorithm. That is why the
# server flag is gone from the release branch and only a debug key reaches it.
#
# WHY THIS TREE. engine-fused is where the paper's n=50 RULER arms were
# measured (queue ids ruler-vestigekv-n50-tgt/-rest, ruler-baseline-n50). An
# ablation that sits beside that table has to run on the same tree or it
# becomes the third tree the review already complained about.
#
# WHY THE OLD RECORDS ARE NOT REUSED. trunc-prefix-nofb and trunc-spread-nofb
# measured the same thing, but through the server flag that no longer exists.
# Re-run through the key so the records say what the paper says.
#
# HOW TO READ IT. Three tasks, three lengths, 10 prompts per cell, seed 0.
# A cell moves by 0.1 at a time, so no single cell is evidence; the paired
# count across all nine is what the comparison supports. This is not the n=50
# grid the RULER table reports.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

MODEL="moonshotai/Kimi-Linear-48B-A3B-Instruct"
TASKS="niah_multikey_2,niah_multikey_3,ruler_qa_hotpot"
LENGTHS="16384,32768,65536"
STEM="results/kimi/ruler/results_vestigekv_n10_16384-32768-65536"

require_free_gpu
for tag in fb-on fb-off-prefix fb-off-spread; do
  refuse_overwrite "$ROOT/${STEM}_${tag}.json"
done
cd "$ROOT"

# env: the trunc-prefix-nofb queue line, which is how the n=50 arms were served
export CTX=73728 MAX_REQS=4 MAMBA_SLOTS=4 CHUNK=4096 GRAPH_BS=4
export ENGINE="$HOME/vestigekv-wt/engine-fused"
export SGLANG_DEBUG_VESTIGEKV_STATS=1
[ -d "$ENGINE/python/sglang" ] || die "engine-fused missing: $ENGINE"
grep -q "SGLANG_DEBUG_VESTIGEKV_NO_OVERFLOW_FALLBACK" \
  "$ENGINE/python/sglang/srt/environ.py" \
  || die "this tree has no NO_OVERFLOW_FALLBACK key; the ablation would silently run WITH the fallback"

run_arm() {  # tag, then any extra env assignments
  local tag="$1"; shift
  local srv="$ROOT/results/kimi/server_vestigekv_${tag}.log"
  local cli="$ROOT/results/kimi/ruler_vestigekv_${tag}.log"
  banner "RULER arm ${tag}: ${*:-no ablation key}"
  env "$@" bash mexp/kimi/vestigekv.sh > "$srv" 2>&1 &
  local pid=$!
  trap 'kill -TERM '"$pid"' 2>/dev/null || true; pkill -f "[s]glang.launch_server" || true' EXIT
  local up=0
  for _ in $(seq 360); do
    sleep 5
    if grep -q "fired up" "$srv"; then up=1; break; fi
    kill -0 $pid 2>/dev/null || break
  done
  [ "$up" = 1 ] || die "server did not come up for ${tag}; see $srv"
  # the backend logs its resolved config; overflow_fallback there is the
  # ground truth for which arm actually ran
  grep -m1 "VestigeKV:" "$srv" || true
  env OPENAI_API_KEY=dummy PYTHONPATH="$ROOT/engine/python" CUDA_VISIBLE_DEVICES="" \
    "$PY" mexp/glm53/run_ruler.py \
      --arm vestigekv --port 30000 --model "$MODEL" \
      --n 10 --out "$ROOT/results/kimi/ruler" \
      --lengths "$LENGTHS" --tasks "$TASKS" --tag "$tag" 2>&1 | tee "$cli"
  kill -TERM $pid 2>/dev/null || true
  pkill -f "[s]glang.launch_server" || true
  sleep 20
  trap - EXIT
}

run_arm fb-on
run_arm fb-off-prefix  SGLANG_DEBUG_VESTIGEKV_NO_OVERFLOW_FALLBACK=1
run_arm fb-off-spread  SGLANG_DEBUG_VESTIGEKV_NO_OVERFLOW_FALLBACK=1 \
                       SGLANG_DEBUG_VESTIGEKV_SPREAD_TRUNCATE=1

echo "wrote ${STEM}_{fb-on,fb-off-prefix,fb-off-spread}.json"
