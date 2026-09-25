---
name: prereg00-premise-merge-vs-evict
description: Tests whether removing positional encoding (NoPE-MLA) makes the KV cache more compressible by merge-class operators than eviction; the founding premise. Run first, it grounds everything downstream.
---

# Premise: does NoPE unlock merge-class KV compression?

**PREREG**: PREREGISTRATION.md | **Output**: out/kimi_metrics.json, out/dsv2_metrics.json (compare.py prints the 2x2) | **Phase**: A

## What it measures
Per MLA layer x head at matched compression rho, G(rho) = min_err(no-merge {stride,recent}) - min_err(merge {avgpool,fft,svd}), on true-query functional out_err. Asks whether "merging" cache entries itself pays off, and whether it pays off more under NoPE (Kimi Linear) than RoPE (DeepSeek-V2-Lite), plus fft-vs-svd (Fourier vs linear-optimal basis).

## Frozen decision rule (if pre-registered)
Support: for rho <= 1/8, median G_NoPE > median G_RoPE and the difference exceeds half the per-head IQR. Negation: G_NoPE ~= G_RoPE (within noise) -> removing PE buys no extra compression. Reverse-null: G_NoPE <= 0 -> even under NoPE the merge class loses to eviction, and the whole direction dies independent of RoPE. Abort gates: MLA layers must == 7, heads == 32, kv_lora_rank == 512, >= 512 query positions, and every op's out_err < 1e-3 at rho=1 (identity check).

## Reproduce
```bash

# 1. build the per-layer latent dumps (NoPE Kimi, RoPE DSV2)
python extract.py --arch kimi --seq-len 8192 --n-docs 8 --n-queries 512 \
  --gpu-expert-layers 14 --max-lm-loss 5 --seed 1 --out out/main2
python extract.py --arch dsv2 --seq-len 8192 --n-docs 8 --n-queries 512 \
  --gpu-expert-layers 0 --max-lm-loss 5 --seed 1 --out out/dsv2
# 2. compute merge-vs-evict out_err per op (PREREGISTRATION analysis)
python analyze.py --data out/main2 --out out/kimi_metrics.json
python analyze.py --data out/dsv2  --out out/dsv2_metrics.json
# 3. the 2x2 native/counterfactual table with G = no-merge - merge
python compare.py out/kimi_metrics.json out/dsv2_metrics.json
```
Requires: the mini-sglang package importable (see project README "Setup"), Kimi Linear checkpoint in the HF cache.

## Verdict
Reverse-null fired. The merge class (fft / low-pass, svd, avgpool) scores 0.00 across the board — under NoPE just as under RoPE — so merging buys nothing and G_NoPE <= 0. Only exact keep/evict (stride/recent-style selection) survives. The central "NoPE makes MLA more compressible by merging" premise is not supported; the gap that remains is a selection gap, not a merge gap.
