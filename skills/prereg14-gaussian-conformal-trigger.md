---
name: prereg14-gaussian-conformal-trigger
description: Tests two probabilistic tier-2 triggers — a Gaussian-tail model of the sketch residual and a conformally-calibrated threshold — for validity, in-distribution rate at recall 0.90, and OOD transfer; run to see if a probabilistic trigger beats the sound-but-100% certified one.
---

# PREREG14 — Probabilistic trigger: Gaussian tail + conformal calibration

**PREREG**: PREREG14.md | **Output**: out/gauss_trigger.json | **Phase**: A

## What it measures
(a) Gaussian-tail trigger: models the sketch-residual logit contribution as N(0, ||q_perp||^2 rho_u^2 / (d-r)) — a ~21x shrinkage vs the Cauchy-Schwarz worst case — and sweeps z. (b) Conformal calibration: picks the per-layer threshold z on doc006 for hard-recall 0.90, then reports achieved recall and trigger rate on doc007 and OOD.

## Frozen decision rule (if pre-registered)
Variant (a) is INVALID if realized residual upper-tail quantiles at z in {2,3} exceed the Gaussian prediction by >2x in miss rate. Viability (either variant): at hard-recall >= 0.90 on test, trigger rate <= 20% (held) and <= 35% (OOD), median over layers. Rate > 60% at recall 0.90 rejects the route.

## Reproduce
```bash

python gauss_trigger.py
```
Requires: the mini-sglang package importable (see project README "Setup"), Kimi Linear checkpoint in the HF cache. Reads pre-extracted latents from out/main2 (basis docs 000-005; CAL doc006, TEST doc007; OOD out/code_kimi, out/logs_kimi) produced by extract.py; results print to stdout (no model forward runs here).

## Verdict
(a) Gaussian prior INVALID: realized argmax residuals are 3.5-10.9 sigma at q90 (Gaussian predicts 1.28) — a selection effect, because argmax tokens are precisely the residual outliers. The Gaussian models random tokens well and the tokens that matter wrongly. (b) Conformal PASSES in-distribution strongly (test recall 0.941 at 1.9% trigger rate) but FAILS OOD transfer (recall 0.786 < 0.90, rate 17.8% within bar), so PARTIAL overall. The scored form (index part + z * ||q_perp|| rho_u / sqrt(448)) survives; only its threshold must be empirically calibrated, not derived from a prior. Boundary layers 3/23 dominate the OOD collapse.
