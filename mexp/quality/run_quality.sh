#!/bin/bash
# Quality reproduction: gsm8k 64-shot (n=800) + MAUVE, per arm, on the live
# server. Run once per arm ($1 = vestigekv | dense); the server must already
# be up at ctx 16384 (see the launch note below), and the arm is selected by
# whether /tmp/vestigekv_full is present (dense) or absent (vestigekv).
#
#   # launch (both nodes; PP split is deployment-specific, not part of the
#   # method): --context-length 16384 --cuda-graph-max-bs 4, backend
#   # vestigekv_mla for both arms; dense = touch /tmp/vestigekv_full first.
#   bash mexp/quality/run_quality.sh vestigekv
#   bash mexp/quality/run_quality.sh dense
#   bash mexp/quality/run_quality.sh score   # after both arms; needs the GPU
set -euo pipefail
ARM=${1:?arm: vestigekv | dense | score}
PORT=${PORT:-30000}
R=results
PY=python

if [ "$ARM" = score ]; then
  $PY mexp/m7_mauve_serving.py score \
    $R/quality_mauve_contexts_tokens.json \
    $R/quality_mauve_texts_vestigekv.json \
    $R/quality_mauve_texts_dense.json \
    $R/quality_mauve_scores.json
  exit 0
fi

# arm select
if [ "$ARM" = dense ]; then touch /tmp/vestigekv_full
else rm -f /tmp/vestigekv_full; fi

echo "== gsm8k 64-shot, n=800 ($ARM) =="
$PY -m sglang.test.few_shot_gsm8k --num-shots 64 --num-questions 800 \
  --parallel 4 --port "$PORT" | tee "$R/quality_gsm8k_64shot_n800_${ARM}.txt"

echo "== MAUVE generations ($ARM) =="
$PY mexp/m7_mauve_serving.py gen \
  "$R/quality_mauve_contexts_tokens.json" \
  "$R/quality_mauve_texts_${ARM}.json"

echo "DONE $ARM (run 'score' after both arms)"
