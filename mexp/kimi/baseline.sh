#!/usr/bin/env bash
# Baseline arm on Kimi Linear = dense MLA: the same tree with --attention-backend triton.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"; guard
cd "$ROOT" && exec $PY -m sglang.launch_server "${COMMON[@]}" --attention-backend triton "$@"
