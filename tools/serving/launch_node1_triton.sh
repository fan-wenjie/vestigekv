#!/bin/bash
PART=$1; CTX=$2; MAXREQ=$3; MAXBS=$4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SGLANG_PP_LAYER_PARTITION=$PART
export CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda
export PATH=/home/user/.conda/envs/sglang/bin:/usr/local/cuda/bin:$PATH
export NCCL_SOCKET_IFNAME=eth0 NCCL_IB_DISABLE=1 NCCL_SOCKET_NTHREADS=4 NCCL_NSOCKS_PERTHREAD=4 NCCL_P2P_NET_CHUNKSIZE=2097152 NCCL_BUFFSIZE=8388608
cd /home/user/fft/sglang

# env preflight audit: the resolved SGLANG_*/NCCL_* env is part of the run
# record (a knob nobody wrote down polluted the authoritative vk curve:
# DEBUG_STATS cost ~15%/step on one arm only). Debug/test instrumentation
# ABORTS an authoritative launch unless explicitly allowed.
echo "== env audit $(hostname) $(date +%H:%M:%S) =="
env | grep -E "^(SGLANG_|NCCL_|PYTORCH_)" | sort
BAD=$(env | grep -E "^SGLANG_(DEBUG|TEST)_" | grep -v "FULL_ARM_FLAG" || true)
if [ -n "$BAD" ] && [ "${ALLOW_INSTRUMENTED:-0}" != "1" ]; then
  echo "ABORT: instrumentation env set on an authoritative launch:"; echo "$BAD"
  echo "(export ALLOW_INSTRUMENTED=1 for a diagnostic run)"; exit 1
fi
nohup /home/user/.conda/envs/sglang/bin/python -m sglang.launch_server   --model-path /home/user/.cache/huggingface/hub/models--moonshotai--Kimi-Linear-48B-A3B-Base/snapshots/3b171c17bfc4ee348599b6781a2ca8715c21c8dc --trust-remote-code --attention-backend triton   --tp-size 1 --pp-size 2 --nnodes 2 --node-rank 1 --dist-init-addr 172.20.56.38:29500   --mem-fraction-static 0.88 --context-length $CTX --chunked-prefill-size 8192   --max-running-requests $MAXREQ --cuda-graph-max-bs $MAXBS --sampling-backend pytorch --host 0.0.0.0 --port 30000   > /home/user/fft/part_node1.log 2>&1 &
echo "node1 launched pid $! part=$PART ctx=$CTX maxreq=$MAXREQ maxbs=$MAXBS"
