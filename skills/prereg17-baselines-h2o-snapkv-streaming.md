---
name: prereg17-baselines-h2o-snapkv-streaming
description: Compares observed-attention KV baselines (H2O, SnapKV, StreamingLLM, recent-protected H2O) against VestigeKV on the NoPE model; run to establish the baseline parity/collapse story.
---

# PREREG17 — Baselines: H2O and SnapKV vs VestigeKV on the NoPE model

**PREREG**: PREREG17.md | **Output**: out/baselines_8192.json | **Phase**: A

## What it measures
Needle-retrieval of observed-attention eviction methods — H2O (accumulated attention), SnapKV (observation-window pooled attention), StreamingLLM (`recent`), and a recent-protected H2O (`h2o_recent`) — against VestigeKV (dig_r64 / digk64 / sel_k16) at matched budgets, at L=8192 and L=32768.

## Frozen decision rule (if pre-registered)
Parity bar: dig_r64 within 0.05 of the best baseline in every cell. If a baseline beats VestigeKV by >0.05 anywhere, the paper reports it plainly and the pitch narrows to the cost advantage.

## Reproduce
```bash

# 8k arm: H2O, SnapKV vs VestigeKV
python harness/e2e.py --arch kimi --seq-len 8192 --n-docs 0 --needle-trials 24 \
  --gpu-expert-layers 18 --seed 11 --rhos 8,32,128 \
  --ops h2o,snapkv,dig_r64,sel_k16,recent --out out/baselines_8192.json
# 32k arm
python harness/e2e.py --arch kimi --seq-len 32768 --n-docs 0 --needle-trials 12 \
  --gpu-expert-layers 16 --seed 11 --rhos 32,128 \
  --ops h2o,snapkv,digk64,sel_k64,recent --out out/baselines_32768.json
# fairness supplement: recent-protected H2O at 8k
python harness/e2e.py --arch kimi --seq-len 8192 --n-docs 0 --needle-trials 24 \
  --gpu-expert-layers 18 --seed 11 --rhos 8,32,128 \
  --ops h2o_recent,dig_r64,recent --out out/h2orecent_8192.json
```
Requires: the mini-sglang package importable (see project README "Setup"), Kimi Linear checkpoint in the HF cache.

## Verdict
Parity bar exceeded — the baselines structurally collapse. At 8k: h2o 0.00/0.00/0.00, snapkv 0.33/0.04/0.00 vs dig_r64 1.00/0.88/0.67 (24 trials, gate 24/24). Replicated at 32k (h2o/snapkv 0.00/0.00 vs digk64 0.92/0.83, 12 trials). The recent-protected H2O supplement changes nothing (0.00 at every ratio). The failure is informational, not budgetary: the needle receives its attention only from queries after the compression point, so observed-attention statistics cannot see it — H2O/SnapKV are built for compress-with-query-in-prompt, while this setting is compress-before-the-query-is-known. VestigeKV's query-independent vestige signal is unaffected.
