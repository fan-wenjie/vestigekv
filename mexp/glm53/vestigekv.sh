#!/usr/bin/env bash
# vestigekv: DSA off, vestigekv_mla over the dense MLA substrate (page size 1 is a precondition).
# Side pool quantized like the DSA index cache (fp8 e4m3 + ue8m0 scale; BF16=1 for the exact
# ablation); every other --vestigekv-* flag at its default (capacity 4096, overflow fallback
# on, activation threshold 0 = compression from token 0).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"; guard
SIDE=fp8; [ "${BF16:-0}" = 1 ] && SIDE=bf16
cd "$ROOT" && exec $PY -m sglang.launch_server "${COMMON[@]}" "${NO_DSA[@]}" \
  --attention-backend vestigekv_mla --page-size 1 --vestigekv-side-pool-dtype "$SIDE" "$@"
