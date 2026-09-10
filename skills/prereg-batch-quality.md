---
name: prereg-batch-quality
description: Checks that evaluating multiple operators per forward (batch>1) does not degrade compression quality; run to confirm the operator is correct and non-degrading under batching.
---

# batch>1 quality note

**PREREG**: PREREG_batch.md | **Output**: out/batch2_8192.json | **Phase**: A

## What it measures
Whether running the compressor with batch>1 (operators evaluated per forward, amortising CPU-offloaded expert-weight streaming) degrades quality relative to batch1, at L=8192, rho=1/32.

## Frozen decision rule (if pre-registered)
No formal accept/reject bar. The honest framing: batch2 vs batch1 is NOT a batching quality gain (the two runs use different document seeds; kernel nondeterminism affects the absolute CE, not the compressed-vs-uncompressed delta). What is confirmed is that the operator runs correctly under batching (gate passed, 12/12 baseline-gated) and does not crash or degrade below batch1.

## Reproduce
```bash

python harness/e2e.py --arch kimi --seq-len 8192 --n-docs 0 --needle-trials 12 --batch 2 \
  --gpu-expert-layers 18 --seed 11 --rhos 32 \
  --ops dig_r64,recent --out out/batch2_8192.json
```
Requires: the mini-sglang package importable (see project README "Setup"), Kimi Linear checkpoint in the HF cache. (The task's out/batch_smoke.json is not present in out/ — only out/batch2_8192.json was retained; skip the smoke output.)

## Verdict
Confirmed no degradation. batch2 dig_r64@1/32 = 1.00 vs batch1 = 0.88, but this is not a batching gain: the two runs use different document seeds (base NLL 0.013 vs 0.014) on a 12-trial sample. What batch2 confirms is that the operator runs correctly under batching (gate1/2 passed, 12/12 baseline-gated) and does not degrade below batch1; the kernel-nondeterminism concern (1.18e-3 CE on the uncompressed model) affects the absolute metric, not the needle-intact delta. A batch-matched head-to-head at fixed seed was not run.
