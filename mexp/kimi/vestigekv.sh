#!/usr/bin/env bash
# VestigeKV arm: every --vestigekv-* flag at its default (capacity 4096, overflow fallback
# on, activation threshold 0, sketch rank 64); extra sglang flags pass through.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"; guard
cd "$ROOT" && exec $PY -m sglang.launch_server "${COMMON[@]}" --attention-backend vestigekv_mla "$@"
