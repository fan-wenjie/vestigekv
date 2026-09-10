---
name: prereg23-streaming-iterated-compression
description: Tests whether blockwise streaming compression (and constant-m global rebalance) matches one-shot global compression; run to validate the compress-as-you-generate deployment path.
---

# PREREG23 — Streaming (iterated) compression: the compound-loss question

**PREREG**: PREREG23.md | **Output**: out/stream_32768.json | **Phase**: A

## What it measures
Whether compressing block-by-block as context closes (`stream_dig`: blocks of 4096, per-block sigma top-m_b) — and a constant-absolute-budget global rebalance (`stream_rebal`: per-block sigma, global top-m) — retrieve as well as the one-shot global selector (digk64) at matched total budget, at L=32768.

## Frozen decision rule (if pre-registered)
stream_dig within 0.09 (1 trial) of digk64 at each rho -> streaming validated, deployment caveat removed; worse than 0.09 -> reported as streaming tax and Limitations row updated; worse than 0.25 anywhere -> must re-rank globally at each event (cost note to ENGINEERING.md). Amendment: stream_rebal within 0.09 of digk64 at each rho AND >= stream_dig -> constant-m schedule validated end to end.

## Reproduce
```bash

# blockwise local streaming vs one-shot global
python harness/e2e.py --arch kimi --seq-len 32768 --n-docs 0 --needle-trials 12 \
  --gpu-expert-layers 16 --seed 11 --rhos 32,128 \
  --ops stream_dig,digk64,recent --out out/stream_32768.json
# constant-m global rebalance
python harness/e2e.py --arch kimi --seq-len 32768 --n-docs 0 --needle-trials 12 \
  --gpu-expert-layers 16 --seed 11 --rhos 32,128 \
  --ops stream_rebal,recent --out out/rebal_32768.json
```
Requires: the mini-sglang package importable (see project README "Setup"), Kimi Linear checkpoint in the HF cache.

## Verdict
Both variants are tax-free — trial-exact ties with the global selector. stream_dig equals one-shot global at both ratios (0.92/0.92 at 1/32, 0.83/0.83 at 1/128; frozen 0.09 bar met with zero gap), because kept rows are never revisited (blockwise keeps ARE the streaming outcome) and salience is per-token, so local ranking does not misallocate against the global ranking. stream_rebal also ties digk64 (0.92/0.83, gap 0.00 at both rhos, 12/12 gate); the mid-run trial-8 displacement lands inside the same miss quota as fixed-rho. The compress-as-you-generate caveat is removed; the constant-m (dynamic-rho) rebalance becomes the mini-sglang M4 default, with fixed-rho streaming the fallback.
