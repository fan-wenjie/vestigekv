#!/bin/bash
# Canonical VestigeKV serving launch. OPERATIONAL values come from
# config/recommended.json (single source, no magic numbers here); ALGORITHM
# values (the cap above all) are NOT auto-applied -- type them explicitly:
#
#   usage: launch_vestige_server.sh <node_rank 0|1> <dist_init_addr> [extra sglang args...]
#   cap:   SGLANG_VESTIGE_TOPJ=16 launch_vestige_server.sh ...   # bounded fetch (recommended cap; see README)
#          (omit for the uncapped default; the choice is yours to state)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
CFG=$HERE/../config/recommended.json
PY=${PYTHON:-/home/user/.conda/envs/sglang/bin/python}
MODEL=${MODEL_PATH:?set MODEL_PATH to the Kimi Linear checkpoint dir}
RANK=${1:?node rank (0 or 1)}; ADDR=${2:?dist-init-addr host:port}; shift 2

cfg() { $PY -c "import json,sys; print(json.load(open('$CFG'))['$1']['value'])"; }
# operational values: read from the JSON (auto-applied by design)
export SGLANG_PP_LAYER_PARTITION=$(cfg pp_partition)
while IFS='=' read -r k v; do export "$k=$v"; done < <(
  $PY -c "import json; [print(f'{k}={v}') for k,v in json.load(open('$CFG'))['nccl_env']['value'].items()]")

exec $PY -m sglang.launch_server \
  --model-path "$MODEL" --trust-remote-code \
  --attention-backend "$(cfg attention_backend)" \
  --sampling-backend "$(cfg sampling_backend)" \
  --mem-fraction-static "$(cfg mem_fraction_static)" \
  --context-length "$(cfg context_length)" \
  --chunked-prefill-size "$(cfg chunked_prefill_size)" \
  --tp-size 1 --pp-size 2 --nnodes 2 --node-rank "$RANK" \
  --dist-init-addr "$ADDR" --host 0.0.0.0 --port 30000 \
  "$@"
