---
name: prereg03-decouple-bandwidth-budget
description: Separates the selector's detector bandwidth kappa from the storage budget m and finds the length-dependent optimal kappa; also rejects renorm-fill (H7) end-to-end. Run to establish the kappa=16/64 selection law.
---

# Decouple detector bandwidth kappa from storage budget m

**PREREG**: PREREG3.md | **Output**: out/decouple_{8192,16384,32768,65536}.json | **Phase**: A

## What it measures
The selector scores tokens by ||C - lowpass(C, kappa)|| with the detector bandwidth kappa held fixed and independent of the storage budget m (`sel_k1/4/16/64` vs the old coupled `sel_coupled` = lowpass(C, m/2)). Tests H4 (survival monotone in m once kappa fixed), H5 (a budget-independent small optimal kappa exists), H6 (does optimal kappa grow with context length), and H7 (whitened renorm-fill vs low-pass-residual selection).

## Frozen decision rule (if pre-registered)
H4 support: for each fixed kappa, intact rate is monotone non-decreasing in m (one dip <= 0.08 allowed). H5 support: some fixed kappa is no worse than `sel_coupled` across all (rho, L). H7: at rho in {1/32, 1/128}, whitened top-8 median retention minus the ||delta|| selector's must exceed half the inter-layer IQR to support; inside it -> negation. Needle main-metric bar is 1.0 nat (0.2-nat jitter flips boundary trials).

## Reproduce
```bash

python harness/e2e.py --arch kimi --seq-len 8192  --n-docs 0 --needle-trials 12 \
  --gpu-expert-layers 18 --seed 11 \
  --ops sel_k1,sel_k4,sel_k16,sel_k64,sel_coupled,recent --out out/decouple_8192.json
python harness/e2e.py --arch kimi --seq-len 16384 --n-docs 0 --needle-trials 12 \
  --gpu-expert-layers 18 --seed 11 \
  --ops sel_k1,sel_k4,sel_k16,sel_k64,sel_coupled,recent --out out/decouple_16384.json
python harness/e2e.py --arch kimi --seq-len 32768 --n-docs 0 --needle-trials 12 \
  --gpu-expert-layers 16 --seed 11 \
  --ops sel_k1,sel_k4,sel_k16,sel_k64,sel_coupled,recent --out out/decouple_32768.json
python harness/e2e.py --arch kimi --seq-len 65536 --n-docs 0 --needle-trials 12 \
  --gpu-expert-layers 14 --seed 11 --rhos 8,32,128 \
  --ops sel_k16,sel_k64,sel_coupled,recent --out out/decouple_65536.json
```
Requires: the mini-sglang package importable (see project README "Setup"), Kimi Linear checkpoint in the HF cache. (H7 renorm-fill was tested via `recovery.py`; the branch-variance evidence via `branch.py`.)

## Verdict
H7 rejected end-to-end: renorm/whitened fill is not better than, and far worse below rho=1/32 than, the low-pass-residual selector. H4/H5 supported: survival is monotone in m and there is a small budget-independent optimal bandwidth kappa ~ 4-16. H6 gives the length law: kappa=16 up to 16k, kappa=64 at >= 32k. This kappa=16/64 selection law is the surviving result and feeds the sidecar selector.
