#!/usr/bin/env bash
# PREREG: KVzip at the compression it was built for, plus the timing reference.
#
# kvzip-needle-8k.sh measured 32x and 128x, which is where THIS paper operates.
# KVzip's own README claims "a 3-4x reduction in KV cache size ... with minimal
# performance degradation". Reporting only 32x/128x would repeat the exact
# criticism the PAT review levelled at the paper's H2O table: a method evaluated
# far outside its stated range, collapsing by construction, which says nothing
# about the method. This script runs 2x, 4x and 8x -- its claimed range and the
# first step past it -- so the pair of runs answers two separate questions:
#
#   at 4x    does KVzip do what its authors claim, on this checkpoint?
#   at 32x+  does either method still retrieve where this paper operates?
#
# The comparison is only honest if both are reported. Whichever way it comes out.
#
# TIMING, and what the number is worth. The harness records the wall time of
# KVzip's scoring pass and of one base forward on the same prompt, in the same
# process. That ratio is a fair reference BECAUSE both arms are our own
# unoptimised HF-forward implementations -- implementation quality is matched,
# which is exactly what makes it comparable. It is NOT a deployment number:
#   * VestigeKV's serving figures come from a fused sglang backend, seven Triton
#     kernels inside a decode CUDA graph. Nothing here runs that path, so this
#     timing must never be compared against those figures.
#   * KVzip's own implementation is not this one; the authors report a 2x
#     decoding speedup we make no attempt to reproduce or dispute.
#   * The scoring cost is the one quantity that survives the caveat, because a
#     second pass over the context is ALGORITHMIC: it is work the method must do
#     whoever writes the kernel. That is the number to quote.
#
# Per-step read volume needs no timing at all and is the cleaner engineering
# comparison: at the same rho KVzip reads only its kept rows, while VestigeKV
# also scans the archive index (260 B/row over (1-rho)T rows). VestigeKV reads
# MORE per step at equal rho. That is the trade being made, and it should be
# stated plainly rather than buried under a wall-clock number.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

OUT="results/harness_kvzip_inrange_8192.json"
MODEL_DIR="$HOME/.cache/vestigekv/instruct_plain"

require_free_gpu
require_model_dir "$MODEL_DIR"
refuse_overwrite "$ROOT/$OUT"

banner "kvzip in its claimed range: 2x, 4x, 8x, 24 trials at 8k"
cd "$ROOT"
"$PY" harness/e2e.py \
  --arch kimi_instruct \
  --seq-len 8192 \
  --n-docs 0 \
  --needle-trials 24 \
  --gpu-expert-layers 18 \
  --seed 0 \
  --ops kvzip,twotier,digk64 \
  --rhos 2,4,8 \
  --out "$OUT"

echo "wrote $OUT"
