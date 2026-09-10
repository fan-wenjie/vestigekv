---
name: prereg18-entropy-cascade-gate
description: Tests tier-1 attention entropy as a standalone tier-2 trigger and as a free pre-gate cascading into the index scan; run to measure the Kimi researchers' entropy conjecture and how much index-scan cost a gate saves.
---

# PREREG18 — Entropy as trigger + cascade gate

**PREREG**: PREREG18.md | **Output**: out/entropy.json | **Phase**: A

## What it measures
Three arms, calibrated on doc006, tested on doc007 + code/logs OOD: (1) entropy-alone threshold trigger (hard-recall 0.90); (2) entropy + max-logit 2-feature logistic; (3) CASCADE — a stage-1 entropy gate (calibrated for hard-recall 0.97) that lets only gated steps run the PREREG14 conformal index scan, reporting gate rate, final fetch rate, and combined recall.

## Frozen decision rule (if pre-registered)
Entropy-alone VIABLE if test rate <= 25% at achieved recall >= 0.85; REJECTED if rate > 60%. Cascade VALUABLE if gate rate <= 50% (index-scan cost at least halved) with combined recall >= 0.85 in-distribution.

## Reproduce
```bash

python ent_trigger.py     # related: entropy.py, headroom.py, estimator.py
```
Requires: the mini-sglang package importable (see project README "Setup"), Kimi Linear checkpoint in the HF cache. Reads pre-extracted latents from out/main2 (basis docs 000-005; CAL doc006, TEST doc007) and OOD out/code_kimi, out/logs_kimi produced by extract.py; the per-layer table prints to stdout (no model forward runs here).

## Verdict
Entropy-alone VIABLE in-distribution — the conjecture is confirmed and the frozen "lands mid, misses" prediction was wrong: test rate 0.214 at recall 0.931 (bar <=0.25 at >=0.85). Cascade VALUABLE: the entropy gate (recall-0.97 calibration) fires on 33% of steps, only those scan the 11% index, final fetch rate 0.4-3.7% per layer, combined recall 0.905 — a ~3x cut to amortized index-scan cost vs always-scan. Universal caveat (third measurement): calibrated thresholds do not transfer OOD (entropy-alone rate 0.74, cascade gate 0.85 on code/logs; boundary layers 23/26 collapse to recall 0.43-0.54). Emerging design: entropy gate (free, ~33%) -> index scan (11%, only inside the gate) -> fetch top-j exact rows.
