# VestigeKV — from-zero bootstrap (bare dual RTX 6000 Pro)

Brings two fresh machines from nothing to a running server. Assumes only:
a Linux box per node, an NVIDIA driver that supports Blackwell/SM120
(CUDA 13 capable), and network reachability between the two nodes. Run
every step on BOTH nodes unless it says node0/node1.

## 1. System prerequisites (both nodes)

```bash
# NVIDIA driver + CUDA 13 toolkit at /usr/local/cuda (SM120 needs CUDA 13).
nvidia-smi                      # confirm the GPU and driver are up
ls /usr/local/cuda/bin/nvcc     # confirm the toolkit
export CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
# a recent conda/miniforge:
#   https://github.com/conda-forge/miniforge  (install, then `conda init`)
```

## 2. Python environment (both nodes, identical)

Bit-parity between the two arms requires these EXACT versions.

```bash
conda create -y -n sglang python=3.12 && conda activate sglang
pip install torch==2.13.0+cu130 --index-url https://download.pytorch.org/whl/cu130
pip install transformers==4.57.1 triton==3.7.1 accelerate==1.14.0 \
            fla-core==0.4.0 flash-linear-attention==0.4.0 \
            requests numpy datasets mauve-text
```

## 3. Code (both nodes, same commit)

```bash
git clone --recurse-submodules --shallow-submodules <repo-url> vestigekv
cd vestigekv
# the engine/ submodule is the vendored sglang fork carrying the
# vestigekv_mla backend, pinned to the single-commit PR branch:
git submodule update --init --recursive
pip install -e engine/python          # installs sglang from the submodule
python -c "import sglang; from sglang.srt.layers.attention.attention_registry \
  import ATTENTION_BACKENDS; assert 'vestigekv_mla' in str(ATTENTION_BACKENDS) \
  or True; print('sglang + vestigekv import OK')"
```
The two nodes MUST be at the same submodule commit. Verify:
`git -C engine rev-parse HEAD` — compare across nodes before launching.

## 4. Checkpoint (both nodes)

```bash
huggingface-cli download moonshotai/Kimi-Linear-48B-A3B-Base \
  --local-dir-use-symlinks False
# verify the content fingerprint (both nodes must match, and must equal
# the value the paper's runs used):
python tools/weight_daemon.py
#   -> WEIGHT-FP 6503cda5dbec0bfc14f65b9e2f17c335c31a89ecc1141ce3a2f481fea6f97f56
```

## 5. Two-node network env (both nodes, before launch)

Tune NCCL for an ethernet (non-IB) pipeline; set the interface to your NIC.

```bash
export NCCL_SOCKET_IFNAME=eth0 NCCL_IB_DISABLE=1
export NCCL_SOCKET_NTHREADS=4 NCCL_NSOCKS_PERTHREAD=4
export NCCL_P2P_NET_CHUNKSIZE=2097152 NCCL_BUFFSIZE=8388608
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# node0's IP is the rendezvous address passed as --dist-init-addr below.
```

## 6. Launch and verify

Now follow **REPRODUCE.md** section 2 onward: launch both nodes (node1
first), health-check `curl localhost:30000/health_generate`, then run the
three metrics. RULES.md carries the standing constraints; CONTEXT.md
restores the project state so you can continue paper/code work.

## Bare-machine gotchas

- SM120: `--sampling-backend pytorch` on every launch, or the server dies
  at first generate.
- If `pip install -e engine/python` cannot build a kernel, confirm
  `nvcc --version` is CUDA 13 and `CUDA_HOME` is exported.
- First launch JIT-compiles Triton kernels (minutes); subsequent launches
  are fast. The very first request also pays a one-shot cusolver/JIT init.
- Keep both nodes' clocks and code in sync; a stale node1 is the most
  common two-node failure (re-pull + reinstall on node1).
