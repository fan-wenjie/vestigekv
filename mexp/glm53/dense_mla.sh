#!/usr/bin/env bash
# dense MLA substrate: DSA off, stock triton MLA backend -- the arm vestigekv_mla wraps.
# NOT the baseline (that is DSA, baseline.sh); an ablation of the substrate only.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"; guard
cd "$ROOT" && exec $PY -m sglang.launch_server "${COMMON[@]}" "${NO_DSA[@]}" --attention-backend triton "$@"
