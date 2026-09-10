#!/bin/bash
# Resident weight daemons (one per node, PP=2). The layer-partition env MUST
# be passed explicitly -- the daemon decides which layers each rank loads;
# omitting it means an even split, which does not fit the 5090 (a loud
# failure, but do not step on it).
set -e
SP=/tmp/claude-1001/-home-user-fft/766c391a-452a-4b35-80a1-f74918c8a077/scratchpad
MODEL=/home/user/.cache/huggingface/hub/models--moonshotai--Kimi-Linear-48B-A3B-Base/snapshots/3b171c17bfc4ee348599b6781a2ca8715c21c8dc
PY=/home/user/.conda/envs/sglang/bin/python
ssh gpu5090 "SGLANG_PP_LAYER_PARTITION=23,4 nohup $PY -m sglang.srt.weight_cache.daemon \
  --model-path $MODEL --trust-remote-code --tp-size 1 --pp-size 2 \
  --nnodes 2 --node-rank 1 --dist-init-method tcp://172.20.56.38:29600 \
  --weight-cache-socket /tmp/vkv_weights.sock > /home/user/fft/weight_daemon1.log 2>&1 &"
SGLANG_PP_LAYER_PARTITION=23,4 nohup $PY -m sglang.srt.weight_cache.daemon \
  --model-path $MODEL --trust-remote-code --tp-size 1 --pp-size 2 \
  --nnodes 2 --node-rank 0 --dist-init-method tcp://172.20.56.38:29600 \
  --weight-cache-socket /tmp/vkv_weights.sock > $SP/weight_daemon0.log 2>&1 &
echo "daemons launching; verify: once both logs print serving, launch with"
echo "  --weight-cache-mode client --weight-cache-socket /tmp/vkv_weights.sock"
