---
name: prereg02-hybrid-impulse-lowpass
description: Tests whether a "sparse exact impulses + low-pass residual" hybrid KV format beats any single-class operator on NoPE-MLA, and whether the low-pass part adds anything over the impulse selector. Run to check if low-pass earns its keep.
---

# Hybrid: sparse impulses + low-pass residual

**PREREG**: PREREG2.md | **Output**: out/hybrid_8192.json (e2e --ops arm) | **Phase**: A

## What it measures
End-to-end (7 MLA layers patched, needle NLL as primary, dCE as gate): does `hybrid` (s = frac*m exact impulses selected by low-pass-residual norm + k = (m-s)/2 complex frequency bins) beat every single-class op (`fft`, `recent`, `stride`, `avgpool`), and does it beat `imp_only`/`sel_coupled` (same budget, no low-pass value)? H3 repeats on RoPE DSV2 as the control.

## Frozen decision rule (if pre-registered)
Primary metric is needle answer-token NLL increment (dCE demoted to a gate: any op with dCE > 0.05 fails regardless of needle). H1 support: `hybrid` beats the best single-class op's needle at >= 2 of rho in {1/32, 1/128} by more than half the trial IQR. H1 negation: hybrid lands inside the best single-class IQR; reverse-null: hybrid is worse. H2 (mechanism): hybrid must beat `imp_only` at matched m, else the low-pass component contributes nothing and the scheme simplifies to pure sparse eviction. Pre-declared: if `imp_only` stably beats all hybrid frac configs, H1 and H2 are both negated. Gates: identity (compression off reproduces CE < 1e-3), rho=1 dCE < 1e-2, intervention-effective (dCE > 1e-6 at rho=1/128), prefix coverage == T x 7 layers.

## Reproduce
```bash

python harness/e2e.py --arch kimi --seq-len 8192 --n-docs 0 --needle-trials 12 \
  --gpu-expert-layers 18 --seed 11 --rhos 8,32,128 \
  --ops hybrid25,hybrid50,hybrid75,fft,recent,stride --out out/hybrid_8192.json
# exploratory sweeps / ablations over the pre-built dump (hardcoded to out/main):
python hybrid.py      # single-config hybrid on out/main
python hybrid2.py out/main   # hybrid over a given dump
python sweep.py       # frac / budget sweep
python ablate.py      # imp_only vs hybrid ablation at rho in {1/8,1/32,1/128}
```
Requires: the mini-sglang package importable (see project README "Setup"), Kimi Linear checkpoint in the HF cache. `hybrid25/50/75` are the frac in {.25,.50,.75} variants in e2e.py's op registry; `--arch dsv2` runs the H3 RoPE control.

## Verdict
The low-pass residual adds nothing on top of selection. Once the exact impulses are chosen, adding frequency bins does not improve needle retrieval over the same-budget pure-selection op (`imp_only`/`sel_coupled`), and dCE is dominated by local context so it cannot resolve the difference. H1/H2 negated: the scheme simplifies to pure exact-keep eviction — the low-pass branch is dropped.
