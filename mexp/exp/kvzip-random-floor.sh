#!/usr/bin/env bash
# PREREG: the floor row of the KVzip table -- what a budget-matched selector
# that does not know what to keep scores on this metric.
#
# The KVzip comparison has one weak point and it is not the finding. Our row
# for KVzip is a TRANSCRIPTION of the authors' scoring rule, not their code,
# and it reads 0.00 at 8x and beyond. The cheapest attack on the table is
# therefore "your port is broken", and a unit test against their _get_score
# answers only for the scoring function, not for the eviction path around it.
#
# This run answers it with the table itself. `imp_random` keeps top-m rows by a
# random score -- `_imp` adds no sinks and no recent window, so it is a clean
# uniform draw at exactly the same budget. The needle's answer spans several
# tokens and all of them must survive, so a random keeper should be at or near
# 0.00 even at 2x, where our KVzip row is 24/24.
#
# That pair is the evidence: a broken transplant cannot score 24/24 at 2x on a
# metric where blind selection scores 0. Whatever the reader concludes about
# KVzip at 8x, they cannot conclude it from a claim that the port does not run.
#
# What must NOT be read from this: a baseline anyone proposed. Nobody has ever
# suggested selecting a KV cache at random. It is a floor, printed so the other
# rows have a scale, and the caption says so.
#
# Reproducibility note: before this run the harness seeded only the needle
# placement, not the global torch stream that imp_random draws from, so a
# random control could not be re-derived from its own record. e2e.py now seeds
# the global stream from --seed as well. That makes this run reproducible and
# leaves every deterministic op unaffected.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

OUT="results/harness_random_floor_8192.json"
MODEL_DIR="$HOME/.cache/vestigekv/instruct_plain"

require_free_gpu
require_model_dir "$MODEL_DIR"
refuse_overwrite "$ROOT/$OUT"

banner "random floor at the KVzip table's five ratios, 24 trials at 8k"
cd "$ROOT"
"$PY" harness/e2e.py \
  --arch kimi_instruct \
  --seq-len 8192 \
  --n-docs 0 \
  --needle-trials 24 \
  --gpu-expert-layers 18 \
  --seed 0 \
  --ops imp_random \
  --rhos 2,4,8,32,128 \
  --out "$OUT"

echo "wrote $OUT"
