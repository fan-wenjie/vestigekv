#!/usr/bin/env bash
# vestigekv: the split pair, DSA prefill + vestigekv_dsa decode (the DSA-model copy of the
# backend); DSA_PREFILL=0 is the single-backend vestigekv_mla shape over the dense substrate.
# Side pool quantized like the DSA index cache (fp8 e4m3 + ue8m0 scale; BF16=1 for the exact
# ablation); every other --vestigekv-* flag at its default (overflow fallback on, activation
# threshold 0 = compression from token 0).
#
# Recall capacity 2048, not the 4096 default: this checkpoint's DSA attends
# index_topk = 2048 rows, and on this line the overflow fence falls back to DSA
# rather than to dense. A 4096-row recall budget against a 2048-row baseline is
# two rows of ours per row of theirs, which is a difference in budget rather
# than in method. It also halves a [max_reqs, W] int32 buffer held per layer,
# on a box that has 2.73 GB left after the pool.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"; guard
SIDE=fp8; [ "${BF16:-0}" = 1 ] && SIDE=bf16
# Split pair: DSA computes prefill (sparse, the indexer's top-k on), VestigeKV
# owns decode. Measured before the split: VestigeKV over a dense base attended
# densely at prefill and paid +5.6 s of TTFT at 32k against DSA's sparse
# prefill -- the whole gap between the arms, and none of it decode. DSA_PREFILL=0
# restores the single-backend shape (DSA off) for the ablation.
if [ "${DSA_PREFILL:-1}" = 1 ]; then
  # DSA resolves the pool to page 64 (its KPool path requires it); VestigeKV
  # addresses token slots and runs at whatever page the pool has.
  BACKENDS=(--prefill-attention-backend dsa --decode-attention-backend vestigekv_dsa
    --dsa-prefill-backend triton --dsa-decode-backend triton)
else
  BACKENDS=("${NO_DSA[@]}" --attention-backend vestigekv_mla --page-size 1)
fi
cd "$ROOT" && exec $PY -m sglang.launch_server "${COMMON[@]}" "${BACKENDS[@]}" \
  --vestigekv-side-pool-dtype "$SIDE" \
  --vestigekv-recall-capacity "${RECALL_CAP:-2048}" "$@"
