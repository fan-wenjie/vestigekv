#!/usr/bin/env bash
# PREREG: head-to-head against a QUERY-INDEPENDENT baseline, which is the gap the
# PAT review named three times. The paper's existing baseline table transplants
# H2O/SnapKV/StreamingLLM into the pre-query order, where their statistic is
# undefined and they collapse by construction -- the review calls that
# uninformative, and it is right. KVzip (Kim et al., NeurIPS 2025) is in the same
# quadrant as VestigeKV: training-free, query-independent, decided before any
# query. Running it here is therefore a fair comparison, unlike the transplants.
#
# Arms, all at a matched total budget on one model, one harness, one protocol:
#   kvzip    our transcription of the authors' scoring rule (see below)
#   twotier  VestigeKV as shipped, both tiers
#   digk64   VestigeKV's selector alone, recall tier off
#
# READ IT AS: needle intact rate and answer-NLL delta at 32x and 128x. If KVzip
# matches or beats twotier, the paper's contribution narrows to cost, not
# retrieval, and Section 8 must say so. If it does not, the comparison the
# review asked for exists and the "no head-to-head" paragraph is replaced by a
# measured one.
#
# HONESTY, to be repeated verbatim in the paper: this is OUR re-implementation of
# KVzip's scoring rule inside OUR harness, transcribed from the authors'
# attention/score.py::_get_score and model/wrapper.py::self_task. It is not the
# authors' optimized code, and no throughput or latency claim is made from it --
# only retrieval accuracy at a matched budget, which is what a re-implementation
# can honestly carry. Their repo targets standard-attention checkpoints; Kimi
# Linear is NoPE-MLA, so their code cannot run here unported, and porting their
# kernels would measure our port, not their method.
#
# TWO DECLARED DEVIATIONS:
#  1. The authors keep a per-(layer, head) budget. An MLA cache row is ONE latent
#     shared by every head, so per-head eviction does not exist in this
#     architecture; the score is reduced over heads with amax, the granularity
#     this cache manages. This is a property of MLA, not a shortcut.
#  2. Their global cross-layer threshold (_threshold) is preserved exactly, so a
#     layer's budget is whatever its scores earn. Total kept over all layers is
#     the same rho*L*T as every other arm: budget-matched.
#
# CONTROL:
#   varies:  the operator (kvzip / the selector / both tiers) and the
#            compression ratio. Nothing else.
#   fixed:   seed 0; 24 needles; L=8192; checkpoint
#            Kimi-Linear-48B-A3B-Instruct; --gpu-expert-layers 18; one
#            harness process, one sitting, so every arm sees the same needles.
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

OUT="results/harness_kvzip_needle_8192.json"
MODEL_DIR="$HOME/.cache/vestigekv/instruct_plain"

require_free_gpu
require_model_dir "$MODEL_DIR"
refuse_overwrite "$ROOT/$OUT"

banner "kvzip vs vestigekv, needle at 8k, 24 trials, 32x and 128x"
cd "$ROOT"
"$PY" harness/e2e.py \
  --arch kimi_instruct \
  --seq-len 8192 \
  --n-docs 0 \
  --needle-trials 24 \
  --gpu-expert-layers 18 \
  --seed 0 \
  --ops kvzip,twotier,digk64 \
  --rhos 32,128 \
  --out "$OUT"

echo "wrote $OUT"
