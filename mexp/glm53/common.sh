#!/usr/bin/env bash
# Shared launch prelude for GLM-5.3-Flash-NVFP4 on 2x RTX PRO 6000 Blackwell (branch
# vestigekv-pro6000x2). Quality/RULER line: CUDA graph ON (decode bs <= GRAPH_BS), radix
# cache OFF, fixed --random-seed, serial clients. GRAPH=0 turns the graph off (the
# decode step is the same either way, only the launch mechanism differs).
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export CUDA_HOME=/usr/local/cuda PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 HF_HUB_OFFLINE=1
export PYTHONPATH=${ENGINE:-$ROOT/engine}/python  # ENGINE=<checkout dir> serves another engine tree
PY=${PYTHON:-$HOME/.conda/envs/sglang-dev/bin/python}
MODEL=nvidia/GLM-5.3-Flash-NVFP4
COMMON=(--model-path $MODEL --quantization modelopt_fp4 --tp-size 2
  --kv-cache-dtype bfloat16 --moe-runner-backend flashinfer_cutlass
  --mem-fraction-static "${MEM_FRAC:-0.955}" --context-length "${CTX:-73728}" --max-running-requests "${MAX_REQS:-4}"
  --max-mamba-cache-size "${MAMBA_SLOTS:-4}" --chunked-prefill-size "${CHUNK:-1024}" --language-model-only
  --disable-radix-cache --disable-custom-all-reduce --sampling-backend pytorch
  --random-seed "${SEED:-0}" --reasoning-parser glm45 --host 127.0.0.1 --port "${PORT:-30000}")
if [ "${GRAPH:-1}" = 1 ]; then COMMON+=(--cuda-graph-max-bs-decode "${GRAPH_BS:-4}"); else COMMON+=(--disable-cuda-graph); fi
# DSA off: the indexer's top-k is switched off at the config level, so pools, arg
# resolution and the model all see a plain NoPE-MLA hybrid (what vestigekv_mla wraps).
NO_DSA=(--json-model-override-args '{"text_config": {"index_topk": null}}')
guard() {
  if nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000{f=1} END{exit !f}'; then echo "ABORT: GPU busy"; exit 1; fi
  if ss -ltn | grep -q ":${PORT:-30000} "; then echo "ABORT: port ${PORT:-30000} busy"; exit 1; fi
}
