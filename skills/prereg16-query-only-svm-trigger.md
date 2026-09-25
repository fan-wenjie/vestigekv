---
name: prereg16-query-only-svm-trigger
description: Tests whether a classifier on the query alone (no archive scan) can decide the tier-2 trigger, and whether kept cache rows are linearly separable from archived ones; run to check the student's half-space / query-only proposal against the index scan.
---

# PREREG16 — Query-only (SVM) trigger + key-side geometry

**PREREG**: PREREG16.md | **Output**: out/qtrigger.json | **Phase**: A

## What it measures
Q1 (key side): whether kept vs archived cache rows are separable by a hyperplane (linear logistic AUC, then + quadratic ||C_u - mean|| feature), on a doc006->doc007 split. Q2 (query side): whether a logistic classifier on q-only features (strict-q, then + free tier-1 byproducts) can hit hard-recall 0.90 at a competitive trigger rate, against the PREREG14 index trigger as reference.

## Frozen decision rule (if pre-registered)
Viable = trigger rate <= 20% held / <= 35% OOD at recall 0.90 (bars as in PREREG14). If strict-q rate > 60% at recall 0.90, the query-only proposal is REJECTED even in refined form.

## Reproduce
```bash

python qtrigger.py
```
Requires: the mini-sglang package importable (see project README "Setup"), Kimi Linear checkpoint in the HF cache. Reads pre-extracted latents from out/main2 (calibrate doc006, test doc007) and OOD out/code_kimi, out/logs_kimi produced by extract.py; the Q1/Q2 tables print to stdout (no model forward runs here).

## Verdict
Not viable. Q1: the half-space picture is REFUTED — a first run's linear AUC=1.000 was a train-equals-test artifact (576 dims separating 128 positives), caught in self-review; on a proper doc006->doc007 split, generalization AUC is 0.66-0.78 linear (0.96 at layer 26) and 0.69-0.80 with the quadratic feature. Kept tokens do not lie on one side of a hyperplane, because the selector statistic is deviation-from-temporal-trend, not a function of the row alone. Q2: the strict q-only classifier needs 0.416 test rate (with recall broken on layers 11/15/19) and the +free-byproduct variant 0.140 (unstable recall), both dominated ~7-20x by the PREREG14 index trigger's 0.019 rate at 0.941 recall. The proposal is rejected as an index replacement; its quadratic-feature refinement already lives inside PREREG14's score.
