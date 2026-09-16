#!/usr/bin/env bash
# baseline: GLM-5.3-Flash as shipped -- DSA (indexer top-k 2048, KPool 4:1) on the Triton DSA
# kernels (the SM120 patch allows them). THE baseline for every GLM comparison.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"; guard
cd "$ROOT" && exec $PY -m sglang.launch_server "${COMMON[@]}" --dsa-prefill-backend triton --dsa-decode-backend triton "$@"
