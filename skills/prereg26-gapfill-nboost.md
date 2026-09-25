---
name: prereg26-gapfill-nboost
description: Fills the "---" table cells at 64x (floor/gap/gapfix) and boosts sample size with a second seed (nboost, pooled with seed 11); run for pure measurement and table completion, no acceptance bar.
---

# PREREG26 — Table gap-fill + n-boost

**PREREG**: PREREG26.md | **Output**: out/nboost_8192.json | **Phase**: A

## What it measures
(A) Gap-fill of the "---" cells: full-row/selector retrieval at rho=1/64 across 8k/32k/65k (floor64: dig_r64/digk64; gap64: sel_k16; gapfix64: sel_k64, matching the row's operator per the amendment). (B) N-boost: repeat the main cells with a new seed (12) and pool seed-11 + seed-12 trials for the reported fractions.

## Frozen decision rule (if pre-registered)
No acceptance bar — pure measurement. Gap-fill values are reported as measured. N-boost pooling rule (frozen): pooled counts = seed-11 + seed-12 trials; BOTH seeds enter the pool regardless of outcome (no per-seed selection); the main table and Wilson appendix switch to pooled fractions with n stated in captions; a seed-12 result differing from seed-11 by more than the binomial band is reported as a discrepancy, not averaged away.

## Reproduce
```bash

# --- gap-fill at rho=1/64 (seed 11) ---
python harness/e2e.py --arch kimi --seq-len 8192  --n-docs 0 --needle-trials 12 --gpu-expert-layers 18 --seed 11 --rhos 64 --ops dig_r64,recent  --out out/floor64_8192.json
python harness/e2e.py --arch kimi --seq-len 32768 --n-docs 0 --needle-trials 12 --gpu-expert-layers 18 --seed 11 --rhos 64 --ops digk64,recent  --out out/floor64_32768.json
python harness/e2e.py --arch kimi --seq-len 65536 --n-docs 0 --needle-trials 12 --gpu-expert-layers 18 --seed 11 --rhos 64 --ops digk64,recent  --out out/floor64_65536.json
python harness/e2e.py --arch kimi --seq-len 8192  --n-docs 0 --needle-trials 12 --gpu-expert-layers 18 --seed 11 --rhos 64 --ops sel_k16,recent --out out/gap64_8192.json
python harness/e2e.py --arch kimi --seq-len 65536 --n-docs 0 --needle-trials 12 --gpu-expert-layers 18 --seed 11 --rhos 64 --ops sel_k16,recent --out out/gap64_65536.json
python harness/e2e.py --arch kimi --seq-len 32768 --n-docs 0 --needle-trials 12 --gpu-expert-layers 18 --seed 11 --rhos 64 --ops sel_k64,recent --out out/gapfix64_32768.json
python harness/e2e.py --arch kimi --seq-len 65536 --n-docs 0 --needle-trials 12 --gpu-expert-layers 18 --seed 11 --rhos 64 --ops sel_k64,recent --out out/gapfix64_65536.json
# --- n-boost: seed 12, pooled with seed 11 ---
python harness/e2e.py --arch kimi --seq-len 8192  --n-docs 0 --needle-trials 12 --gpu-expert-layers 18 --seed 12 --rhos 32,64,128 --ops dig_r64,sel_k16,recent --out out/nboost_8192.json
python harness/e2e.py --arch kimi --seq-len 32768 --n-docs 0 --needle-trials 12 --gpu-expert-layers 18 --seed 12 --rhos 32,64,128 --ops digk64,recent         --out out/nboost_32768.json
```
Requires: the mini-sglang package importable (see project README "Setup"), Kimi Linear checkpoint in the HF cache.

## Verdict
Gap-fill: no bar — the 64x cells are reported as measured and slotted into the tables (row operator matched: sel_k64 at 32k/65k per the amendment; the earlier sel_k16 numbers 0.75@32k, 0.58@65k are retained as additional data points, not table cells). N-boost: seed-12 trials are pooled with seed-11 with no per-seed selection, and the pooled n is stated in the table and Wilson-appendix captions.
