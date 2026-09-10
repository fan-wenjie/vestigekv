---
name: prereg04-branch-asymmetric-cache
description: Tests asymmetric budget split between the 512-d content branch and 64-d sidecar, whether detected spans are rank-1 in time, and whether isolated tokens are per-token recoverable. Run to settle the archive/dual-tier design question.
---

# Branch-asymmetric cache: split budget, span rank, recoverability

**PREREG**: PREREG4.md | **Output**: out/span_8192.json, out/lbasis_8192.json, out/recoverable_8192.json | **Phase**: A

## What it measures
Three arms on NoPE-MLA: (1) split the budget unevenly across the 64-d head-shared branch (~68% of score variance) vs the 512-d per-head content branch (`split_r*_c*`); (2) whether detector-selected spans are rank-1 along the time axis (SVD of found spans / local temporal basis `lbasis_w*` vs exact rectangular `span_w*`); (3) per-token recoverability of isolated impulses vs low-pass fill vs low-rank PCA code (`recover.py`, `lowrank_r*_c*`).

## Frozen decision rule (if pre-registered)
Split: at >= 2 of 3 cost-matched points, split's survival must exceed the uniform allocation (or reach the dCE <= 0.02 quality gate at lower cost) to support; inside the IQR -> no gain; worse -> the 67.6% variance measurement cannot drive a budget rule (report as negative). Span-rank: R^2 of a rank-1 (single-peak) fit must exceed 0.8 to call spans rank-1; span-vs-point: a width w>1 must beat w=1 for the bound-phrase hypothesis. Recoverability: the scheme must guarantee per-token recoverability, not just compression ratio.

## Reproduce
```bash

# span vs isolated point (same budget m: m points vs m/w spans of width w)
python harness/e2e.py --arch kimi --seq-len 8192 --n-docs 0 --needle-trials 12 \
  --gpu-expert-layers 18 --seed 11 --rhos 32,128 \
  --ops span_w1,span_w3,span_w5,span_w9,span_w17,sel_k16,recent --out out/span_8192.json
# local temporal basis vs exact spans (needs the offline basis)
python make_tbasis.py out/main2 out/tbasis_kimi.pt
python harness/e2e.py --arch kimi --seq-len 8192 --n-docs 0 --needle-trials 12 \
  --gpu-expert-layers 18 --seed 11 --tbasis out/tbasis_kimi.pt --rhos 32,128 \
  --ops lbasis_w5,lbasis_w9,lbasis_w17,lbasis_w33,span_w5,span_w17,sel_k16,recent \
  --out out/lbasis_8192.json
# per-token recoverable (low-rank+sparse / split) vs smeared, cost-matched
python make_pca.py out/main2 out/pca_kimi.pt 128
python harness/e2e.py --arch kimi --seq-len 8192 --n-docs 0 --needle-trials 12 \
  --gpu-expert-layers 18 --seed 11 --pca out/pca_kimi.pt --rhos 8.4,11.5,17,23 \
  --ops lowrank_r64_c128,lowrank_r32_c32,lowrank_r16_c32,lowrank_r16_c64,split_r1_c128,split_r2_c32,split_r4_c32,sel_k16,recent \
  --out out/recoverable_8192.json
# supporting static analyses (hardcoded to out/main2 / out/dsv2):
python spanrank.py    # time-axis SVD of found spans -> effective rank
python recover.py     # reconstruction error + neighbour-distinguishability
python branch.py      # 64-d vs 512-d branch score-variance split
```
Requires: the mini-sglang package importable (see project README "Setup"), Kimi Linear checkpoint in the HF cache.

## Verdict
Rank-1 refuted: spans have effective rank ~ w/3, far from rank-1 (R^2 0.29-0.65), though a Gaussian kernel is the best single peak and still captures too little span energy. The uncertainty-principle w*k boundary is refuted — the failure region is cross-shaped, not a clean product bound. Recover-vs-retrieve conflict: low-pass fill destroys per-token recovery while selection discards low-attention tokens. Resolution is a dual-tier design — an exact archive kept off the attention path — rather than a single asymmetric split.
