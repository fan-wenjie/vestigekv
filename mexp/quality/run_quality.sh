#!/bin/bash
# Quality reproduction: gsm8k-platinum 64-shot (n=1209, full cleaned set)
# + MAUVE, per arm, on the live server. Run once per arm
# ($1 = vestigekv | dense); the server must already be up at ctx 16384
# (see the launch note in README.md). Dense = stock triton backend on the
# same tree; the /tmp/vestigekv_full flag is also touched for the dense
# arm (inert under triton, kept for the legacy same-backend protocol).
# MODEL_TAG=_instruct (etc.) separates per-checkpoint output files.
#
#   bash mexp/quality/run_quality.sh vestigekv
#   bash mexp/quality/run_quality.sh dense
#   bash mexp/quality/run_quality.sh score   # after both arms; needs the GPU
set -euo pipefail
ARM=${1:?arm: vestigekv | dense | score}
TAG=${MODEL_TAG:-}
PORT=${PORT:-30000}
R=results
# The engine is imported via PYTHONPATH (never installed); make sure it is
# on the path even when the caller forgot to set it.
if [ -d engine/python ] && [[ ":${PYTHONPATH:-}:" != *":$PWD/engine/python:"* ]]; then
  export PYTHONPATH="$PWD/engine/python${PYTHONPATH:+:$PYTHONPATH}"
fi
# Prefer $PY; otherwise pick a python that has the client deps (numpy +
# sglang), so the script works even without an activated conda env.
PY=${PY:-python}
if ! $PY -c "import numpy, sglang" 2>/dev/null; then
  for c in "$HOME/.conda/envs/sglang/bin/python"; do
    if [ -x "$c" ] && "$c" -c "import numpy, sglang" 2>/dev/null; then PY="$c"; break; fi
  done
fi

if [ "$ARM" = score ]; then
  $PY mexp/m7_mauve_serving.py score \
    $R/quality_mauve_contexts_tokens.json \
    $R/quality_mauve_texts_vestigekv${TAG}.json \
    $R/quality_mauve_texts_dense${TAG}.json \
    $R/quality_mauve_scores${TAG}.json
  exit 0
fi

# arm select
if [ "$ARM" = dense ]; then touch /tmp/vestigekv_full
else rm -f /tmp/vestigekv_full; fi

echo "== gsm8k-platinum 64-shot, n=1209 full set (${ARM}${TAG}) =="
# GSM8K-Platinum (madrylab/gsm8k-platinum, arXiv:2502.03461): the cleaned
# revision of the FULL gsm8k test set (1209 rows, label errors fixed,
# ill-posed questions removed). Replaces the legacy n=800 subset of the
# noisy original test set. 64-shot prefix = first 64 rows of the file.
# --parallel 1 on purpose: the quality gate is a REPRODUCIBILITY target, not
# a throughput target. Serial requests pin the server-side batch to 1, so
# kernel reduction order (and hence the logits) cannot drift with timing;
# combined with --disable-radix-cache on the quality server this makes every
# run bitwise repeatable on the same hardware.
$PY -m sglang.test.few_shot_gsm8k --num-shots 64 --num-questions 1209 \
  --data-path mexp/quality/gsm8k_platinum.jsonl \
  --parallel 1 --port "$PORT" | tee "$R/quality_gsm8kplatinum_64shot_n1209_${ARM}${TAG}.txt"

echo "== MAUVE generations (${ARM}${TAG}) =="
$PY mexp/m7_mauve_serving.py gen \
  "$R/quality_mauve_contexts_tokens.json" \
  "$R/quality_mauve_texts_${ARM}${TAG}.json"

echo "DONE ${ARM}${TAG} (run 'score' after both arms)"
