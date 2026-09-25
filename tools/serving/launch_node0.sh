#!/bin/bash
PART=$1; CTX=$2; MAXREQ=$3; MAXBS=$4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda
export PATH=/home/user/.conda/envs/sglang/bin:/usr/local/cuda/bin:$PATH
export NCCL_SOCKET_IFNAME=eth0 NCCL_IB_DISABLE=1 NCCL_SOCKET_NTHREADS=4 NCCL_NSOCKS_PERTHREAD=4 NCCL_P2P_NET_CHUNKSIZE=2097152 NCCL_BUFFSIZE=8388608
export SGLANG_PP_LAYER_PARTITION=$PART SGLANG_TEST_VESTIGEKV_FULL_ARM_FLAG=/tmp/vestigekv_full
# code-sync guard: git identity of the WHOLE sglang tree must match across
# nodes -- HEAD (committed state, every file incl. stock ports) plus an md5 of
# working-tree diff + porcelain status (uncommitted edits and untracked files).
# A backend-only content hash once missed a batched_step.py fix and node1 ran
# stale code (the dual-machine code-sync trap); git identity closes that.
SGL=/home/user/fft/sglang
FP='git -C '"$SGL"' rev-parse HEAD^{tree}; { git -C '"$SGL"' diff HEAD; git -C '"$SGL"' status --porcelain=v1 -uall; } | md5sum | cut -d" " -f1'
L=$(eval "$FP")
R=$(ssh gpu5090 "$FP" 2>/dev/null)
[ -n "$L" ] && [ -n "$R" ] || { echo "ABORT: git fingerprint empty -- guard would be checking nothing"; exit 1; }
[ "$L" = "$R" ] || { echo "ABORT: sglang git state differs across nodes (run sync_node1.sh first)"; exit 1; }
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
nohup /home/user/.conda/envs/sglang/bin/python -m sglang.launch_server   --model-path /home/user/.cache/huggingface/hub/models--moonshotai--Kimi-Linear-48B-A3B-Base/snapshots/3b171c17bfc4ee348599b6781a2ca8715c21c8dc --trust-remote-code --attention-backend vestigekv_mla   --tp-size 1 --pp-size 2 --nnodes 2 --node-rank 0 --dist-init-addr 172.20.56.38:29500   --mem-fraction-static 0.88 --context-length $CTX --chunked-prefill-size 8192   --max-running-requests $MAXREQ --cuda-graph-max-bs $MAXBS --sampling-backend pytorch --host 0.0.0.0 --port 30000   > /tmp/claude-1001/-home-user-fft/766c391a-452a-4b35-80a1-f74918c8a077/scratchpad/part_node0.log 2>&1 &
echo "node0 launched pid $! part=$PART ctx=$CTX maxreq=$MAXREQ maxbs=$MAXBS"
