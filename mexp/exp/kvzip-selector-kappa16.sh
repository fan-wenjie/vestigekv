#!/usr/bin/env bash
# PREREG: the selector-only row of the KVzip table, at the DEPLOYED bandwidth.
#
# kvzip-needle-8k.sh and kvzip-inrange-8k.sh ran the selector arm as `digk64`.
# The `k64` in that op name is the detector BANDWIDTH (kappa=64), not a read
# width -- a naming trap this repo has already been caught by once. The shipped
# constant is kappa=16, and the paper's Section 6.1 selector numbers come from
# `dig_r64`, which takes the default kappa=16. Printing digk64's row next to
# them puts two different numbers in the paper for one quantity (0.75 vs 0.89
# at 32x, 0.25 vs 0.69 at 128x) with nothing saying why.
#
# So this run measures the selector arm the paper actually deploys: r=64 sketch
# rank, kappa=16 bins, same seed, same 24 needles, same five ratios, same
# harness. Its output replaces the digk64 row in the table; digk64's record
# stays on disk as the off-spec bandwidth it is.
#
# What must NOT be read from this: a rerun of the OTHER two arms. kvzip and
# twotier are not in --ops here because their records already exist under the
# same seed and protocol; re-measuring them would only invite a reader to ask
# which of two identical runs the paper printed.
#
# Two ratios of this grid (32x, 128x) are the ones the paper cites in prose.
# If they disagree with Section 6.1's pooled seeds 11+12 values, THAT is the
# finding: the pooled numbers come from records no surviving tree can rebuild
# (see mexp/kimi/make_selector_numbers.py), and this run, which is
# reproducible, is what the paper should print.
#
# CONTROL:
#   varies:  the detector bandwidth, kappa=16 against the kappa=64 the two
#            scripts above ran. One constant, nothing else.
#   fixed:   seed 0; 24 needles; L=8192; the same five ratios; checkpoint
#            Kimi-Linear-48B-A3B-Instruct; --gpu-expert-layers 18.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

OUT="results/harness_selector_k16_8192.json"
MODEL_DIR="$HOME/.cache/vestigekv/instruct_plain"

require_free_gpu
require_model_dir "$MODEL_DIR"
refuse_overwrite "$ROOT/$OUT"

banner "tier-1 selector at the deployed kappa=16: 2x..128x, 24 trials at 8k"
cd "$ROOT"
"$PY" harness/e2e.py \
  --arch kimi_instruct \
  --seq-len 8192 \
  --n-docs 0 \
  --needle-trials 24 \
  --gpu-expert-layers 18 \
  --seed 0 \
  --ops dig_r64 \
  --rhos 2,4,8,32,128 \
  --out "$OUT"

echo "wrote $OUT"
