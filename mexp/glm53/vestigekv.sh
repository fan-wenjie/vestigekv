#!/usr/bin/env bash
# vestigekv: DSA off, vestigekv_mla over the dense MLA substrate (page size 1 is a precondition).
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
cd "$ROOT" && exec $PY -m sglang.launch_server "${COMMON[@]}" "${NO_DSA[@]}" \
  --attention-backend vestigekv_mla --page-size 1 --vestigekv-side-pool-dtype "$SIDE" \
  --vestigekv-recall-capacity "${RECALL_CAP:-2048}" "$@"
