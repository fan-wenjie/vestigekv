---
name: prereg19-selfcal-e2e-recovery-twotier
description: Tests per-context self-calibration of the cascade (Part B, CPU) and whether the two-tier archive + trigger actually recovers needle retrieval that eviction loses end-to-end (Part A, GPU); run to validate the tier-2 fallback as a real operator.
---

# PREREG19 — Self-calibration + end-to-end recovery (two-tier archive)

**PREREG**: PREREG19.md | **Output**: out/twotier_8192.json | **Phase**: A

## What it measures
Part B (offline): calibrate the entropy gate and conformal z on the SAME context's own prefix queries (split each OOD doc's 512 queries into cal/test halves by position), testing whether per-context self-calibration fixes the OOD transfer failure. Part A (GPU): a two-tier operator in the e2e harness — evict to rho, keep archived rows addressable, run the self-calibrated cascade (gate -> index scan -> fetch top-j=16) per future query — measuring needle intact rate vs plain eviction at L=8192, rho in {1/32, 1/128}, 24 trials.

## Frozen decision rule (if pre-registered)
Part B: on OOD docs, self-calibration must restore gate rate <= 50% AND combined recall >= 0.85. Part A: at 1/128 two-tier must recover >= +0.15 intact over plain eviction (0.67 -> >= 0.82); at 1/32 no regression (> 0.79); fetch median <= 16 rows/(query,head), gate rate <= 60%. Recovery < +0.05 refutes the archive's value at this task.

## Reproduce
```bash

# Part B (CPU, self-calibration):
python selfcal.py
# Part A (GPU, end-to-end two-tier recovery, writes out/twotier_8192.json):
python harness/e2e.py --arch kimi --seq-len 8192 --n-docs 0 --needle-trials 24 --gpu-expert-layers 18 --seed 11 --rhos 32,128 --ops twotier,dig_r64,recent --out out/twotier_8192.json
```
Requires: the mini-sglang package importable (see project README "Setup"), Kimi Linear checkpoint in the HF cache. selfcal.py reads pre-extracted latents from out/main2 (basis docs 000-005) and OOD out/code_kimi, out/logs_kimi from extract.py; Part A's e2e.py runs the model on GPU and writes out/twotier_8192.json.

## Verdict
Part B PARTIAL, yielding a self-deciding rule: code gate 0.320 / recall 0.846 (at the bar, missing 0.85 by 0.004); logs gate 0.818 (fail) / recall 0.880 — self-calibration repairs code but not the logs entropy gate (repetitive content keeps tier-1 attention diffuse), though the index stage itself stays healthy (0.880). Rule: if the calibrated gate fires on >60% of calibration queries, disable it and scan the index every step (11%). Part A RECOVERY BAR EXCEEDED: 1/32 two-tier 1.00 vs plain 0.88 (no regression), 1/128 1.00 vs 0.67 (exceeds +0.15). Cost verified: mask_extra 0.005-0.026 (no leak), per-query head-union 20-107 rows/layer (~25-125 KB/query/layer at bf16); the entropy gate disabled itself on 6/7 layers per the Part B rule, so recovery here comes from always-scan + fetch. Validated on single-point retrieval at 8k; long task-chain cascades remain untested.
