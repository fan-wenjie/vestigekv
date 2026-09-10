# VestigeKV — self-contained reproduction (dual RTX 6000 Pro)

This one document reproduces every headline number in the paper from a
fresh two-machine setup. Follow it top to bottom; each metric ends with the
bar it must clear.

## 0. Context (what you are verifying)

VestigeKV is a training-free KV-cache compression backend for NoPE-MLA
models (Kimi Linear 48B-A3B), ported into sglang as attention backend
`vestigekv_mla`. It evicts cache rows by a query-independent signal the
NoPE cache already carries (the un-roped 64-dim branch) into a
GPU-resident archive, and keeps them reachable with a certified per-step
recall tier. Claim: **decode speed, not memory** — bit-exact bf16 rows,
both tiers on GPU. The dense control is the *same code tree* with
`--attention-backend triton`.

Three headline metrics, all at bs=1 unless noted:
1. **Latency**: 4k prefill -> 512k continuous decode; per-token latency vs S.
   Bar: crossover ~48k; **1.20x at 272k, 1.39x at 496k**; vestigekv slope
   ~0.5 ns/token vs dense ~6.3 ns/token.
2. **Throughput**: 64k prefill + 4k decode, batch 1..16. Bar: within
   launch-noise at bs=1 (~0.98x), **+8% at bs=12** (attention cannot batch,
   so the advantage grows with batch).
3. **Quality**: 64-shot gsm8k (n=800) and MAUVE. Bar: gsm8k
   **0.834 (vk) vs 0.821 (dense)** — statistically indistinguishable;
   MAUVE within 0.10; teacher-forced dCE <= 0.05 nats.

Why bs=1 throughput sits near dense at 64k and only wins at long context:
attention is ~7% of the decode step at 64k (MoE + linear layers + the
two-node pipeline dominate), so compression's saving is small there; it
grows with context length (latency curve) and with batch (throughput).
This is deployment-dependent in the honest direction — a single-device or
NVLink box lifts every number.

## 1. Machine + environment

Two nodes, one RTX 6000 Pro (Blackwell, SM120) each, on a fast network.
**SM120 note**: Blackwell needs `--sampling-backend pytorch` (in every
launch below) — the default sampling kernel fails its arch check on SM120.

```bash
# clone with the engine submodule (the vendored sglang fork)
git clone --recurse-submodules <repo-url> vestigekv && cd vestigekv
# pinned deps (exact versions matter for bit-parity):
pip install -r <(cat <<'REQ'
torch==2.13.0+cu130
transformers==4.57.1
triton==3.7.1
accelerate==1.14.0
fla-core==0.4.0
flash-linear-attention==0.4.0
REQ
)
pip install -e engine/python   # the vendored sglang with the vestigekv backend
```

Checkpoint: `moonshotai/Kimi-Linear-48B-A3B-Base`. Verify the content
fingerprint before trusting any number (both arms must load the same
weights):

```bash
python tools/weight_daemon.py   # prints WEIGHT-FP; must equal
# 6503cda5dbec0bfc14f65b9e2f17c335c31a89ecc1141ce3a2f481fea6f97f56
```

Pipeline partition for this model on two nodes: `SGLANG_PP_LAYER_PARTITION=23,4`
(node0 = 23 layers, node1 = 4). This is the ONLY machine-specific parameter;
everything else below is verbatim from the paper protocol.

## 2. Launch (both nodes; node1 first)

```bash
# node1:
SGLANG_PP_LAYER_PARTITION=23,4 python -m sglang.launch_server \
  --model-path moonshotai/Kimi-Linear-48B-A3B-Base --trust-remote-code \
  --attention-backend vestigekv_mla --tp-size 1 --pp-size 2 \
  --nnodes 2 --node-rank 1 --dist-init-addr <node0-ip>:29500 \
  --context-length 524288 --max-total-tokens 589824 \
  --cuda-graph-max-bs 2 --sampling-backend pytorch --disable-radix-cache
# node0: identical, --node-rank 0.
# DENSE baseline: the SAME command with --attention-backend triton
#   (same tree — this proves the modifications do not affect stock behavior).
```

## 3. Metric 1 — latency (per arm)

```bash
python -m sglang.benchmark.serving --backend sglang \
  --model moonshotai/Kimi-Linear-48B-A3B-Base --num-prompts 1 \
  --dataset-name random --random-input-len 4096 --random-output-len 520192 \
  --random-range-ratio 1 --max-concurrency 1 --warmup-requests 0 \
  --output-details --output-file results/latency_stream_4k-512k_<arm>.jsonl
# ALSO keep the node-0 server log: above ~240k the client inter-token
# stamps degrade under stream batching, so the server decode log is the
# per-token authority. Plot: python mexp/bench/plot_curve_jsonl.py (see README).
```
Bar: crossover ~48k; 1.20x at 272k; 1.39x at 496k.

## 4. Metric 2 — throughput (per arm, relaunch with ctx 73728, maxbs 16)

```bash
for BS in 1 2 4 8 12 16; do
  python -m sglang.benchmark.serving --backend sglang \
    --model moonshotai/Kimi-Linear-48B-A3B-Base --num-prompts $((BS*2)) \
    --dataset-name random --random-input-len 65536 --random-output-len 4096 \
    --random-range-ratio 1 --max-concurrency $BS --warmup-requests 0 \
    --output-details --output-file results/throughput_64k+4k_bs${BS}_<arm>.jsonl
done
```
Bar: ~0.98x at bs=1, +8% at bs=12.

## 5. Metric 3 — quality (relaunch with ctx 16384, maxbs 4)

```bash
bash mexp/quality/run_quality.sh vestigekv
bash mexp/quality/run_quality.sh dense
bash mexp/quality/run_quality.sh score      # after both arms; needs the GPU
```
Bar: gsm8k 0.834 vs 0.821; MAUVE within 0.10.

## 6. Algorithm claims (single machine, no serving)

The needle-retrieval, baseline (H2O/SnapKV/StreamingLLM), RoPE-collapse,
ablation, and bits-per-byte numbers come from the HF-forward harness (no
two-node serving needed):

```bash
python harness/e2e.py --arch kimi --seq-len 8192 --n-docs 0 \
  --needle-trials 24 --gpu-expert-layers 18 --seed 0 --ops baselines \
  --out results/harness_baselines_8192.json
```
See `harness/README.md` and `EXPERIMENTS.md` for the full `--ops` set.

## 7. If a number disagrees

- Re-check the WEIGHT-FP on both nodes (step 1) — a mismatched checkpoint
  is the most common cause.
- Confirm `--sampling-backend pytorch` on SM120 (else the server dies at
  first generate).
- Above ~240k, read the SERVER decode log, not the client itls.
- Network variance is +/-15% on the fixed per-step terms; collect each
  arm's curve inside one launch and read cross-arm claims as
  delta-vs-4k where the fixed terms cancel.
