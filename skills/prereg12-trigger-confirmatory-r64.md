---
name: prereg12-trigger-confirmatory-r64
description: Confirmatory re-run of the r=64 trigger v2 on freshly extracted fineweb and a new OOD family, to decide whether PREREG11's r=64 held-out number was fitted to its evaluation set; run before quoting any validated recall@4 for the sketch trigger.
---

# PREREG12 — Confirmatory: trigger v2 at r=64 on fresh data

**PREREG**: PREREG12.md | **Output**: out/trigger12_r64.json | **Phase**: A

## What it measures
Re-runs the frozen r=64 configuration (sidecar + rank-64 query-subspace sketch, index 128/576 = 22%, basis still fitted on out/main2 docs 000-005) on data extracted AFTER the PREREG11 numbers were seen: two new fineweb docs (seed 2) and one new OOD family (concatenated structured JSON/CSV). Decides whether the r=64 held-out claim survives on never-seen data.

## Frozen decision rule (if pre-registered)
recall@4 >= 0.85 on fresh fineweb AND >= 0.70 on fresh OOD (hard set, median over layers) validates trigger v2 at 22% index. Fail means the PREREG11 r=64 numbers were fitted to their evaluation set and the claim stays PARTIAL.

## Reproduce
```bash

# 1. build the fresh evaluation latents first (GPU, one forward pass each):
python extract.py --arch kimi --seq-len 8192 --n-queries 512 --gpu-expert-layers 14 --max-lm-loss 99 --seed 2 --n-docs 2 --out out/fresh_kimi
python extract.py --arch kimi --seq-len 8192 --n-queries 512 --gpu-expert-layers 14 --max-lm-loss 99 --seed 1 --n-docs 1 --corpus-file corpus/struct.txt --out out/struct_kimi
# 2. the confirmatory trigger read (CPU, reads out/fresh_kimi + out/struct_kimi):
python trigger12.py 64
```
Requires: the mini-sglang package importable (see project README "Setup"), Kimi Linear checkpoint in the HF cache. The trigger12.py read consumes pre-extracted latents from out/main2 (basis, docs 000-005), out/fresh_kimi, and out/struct_kimi produced by the extract.py step above; only extract.py runs a model forward.

## Verdict
FAILS the frozen bar; the claim stays PARTIAL. Fresh fineweb (seed 2, never seen) median recall@4 = 0.812/0.811 < 0.85; fresh OOD (structured JSON) 0.767/0.765 passes 0.70. Per the frozen rule the fineweb miss decides — PREREG11's r=64 held-out 0.891 contained ~0.08 of evaluation-set fitting, now removed. The honest thrice-replicated numbers for r=64 (22% index): recall@4 ~0.77-0.82 across three data families, recall@16 ~0.93-0.98. No measured configuration reaches a validated 0.85@4. The mechanism is real, stable, and PARTIAL: usable engineering, not a guaranteed recall claim.
