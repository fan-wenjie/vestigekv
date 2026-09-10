---
name: prereg11-trigger-v2-sketch
description: Sweeps a rank-r query-subspace content sketch added to the sidecar trigger and measures hard-set recall@4 vs index size; run to test whether a cheap content sketch rescues the PREREG10 rejection.
---

# PREREG11 — Trigger v2: sidecar + rank-r query-subspace sketch

**PREREG**: PREREG11.md | **Output**: out/trigger_r{r}.json | **Phase**: A

## What it measures
Adds a rank-r PCA sketch of the content branch, fitted in the query subspace on TRAIN docs 000-005 only, to the exact sidecar. Sweeps r in {0,8,16,32,64} (index 11%-22% of a row) and measures hard-set recall@4 on held-out fineweb and OOD, testing whether the content sketch recovers the queries the sidecar alone misses.

## Frozen decision rule (if pre-registered)
At r <= 32 (index <= 17%): recall@4 >= 0.85 held-out AND >= 0.70 OOD (median over layers) validates trigger v2. recall@4 < 0.6 at r=32 refutes the sketch direction. Layers 3 and 26 stay in the median, no post-hoc exclusion.

## Reproduce
```bash

for r in 0 8 16 32 64; do python trigger2.py $r; done   # r=0 reproduces PREREG10; also: trigger12.py for the fresh-data confirmatory
```
Requires: the mini-sglang package importable (see project README "Setup"), Kimi Linear checkpoint in the HF cache. Reads pre-extracted latents from out/main2 (basis fitted on docs 000-005; held-out 006-007; OOD out/code_kimi, out/logs_kimi), all produced by extract.py — no model forward pass runs here.

## Verdict
PARTIAL at the frozen bar. The rank curve is monotone (recall@4 0.489 -> 0.891 as r goes 0 -> 64). At the pre-registered point r=32 (17% index): held-out 0.818/0.821 misses the 0.85 bar by 0.03; OOD 0.755/0.745 passes the 0.70 bar. r=64 clears both (held 0.891, OOD 0.829) but lies outside the frozen claim and its numbers have been seen, so it cannot be adjudicated post hoc — that is what PREREG12 confirms on fresh data. The d_eff~23 reasoning under-delivered: recall keeps climbing past r=32 because hard queries use content directions outside the bulk query subspace.
