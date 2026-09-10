---
name: prereg10-trigger-recall-sidecar
description: Measures whether the exact sidecar partial logit (q_r . r_u) alone can rank an evicted token into the top-j of the archive on the queries that eviction breaks; run to test the cheapest possible tier-2 recall trigger.
---

# PREREG10 — Archive trigger-recall, sidecar only

**PREREG**: PREREG10.md | **Output**: out/trigger.json | **Phase**: A

## What it measures
On exactly the (query, head) pairs whose true argmax was evicted (the hard set), how often the exact sidecar summand q_r . r_u ranks the needed archived token into the top-j (j in {1,4,16}). Isolates whether a cheap, query-aware trigger exists without any content-branch sketch.

## Frozen decision rule (if pre-registered)
recall@4 >= 0.85 on held-out fineweb AND >= 0.70 on OOD (hard set, median over layers) validates the trigger. recall@4 < 0.5 anywhere rejects it. Between is partial.

## Reproduce
```bash

python trigger.py
```
Requires: the mini-sglang package importable (see project README "Setup"), Kimi Linear checkpoint in the HF cache. This trigger-line script reads pre-extracted latents from out/main2 (and OOD out/code_kimi, out/logs_kimi) produced by extract.py — no model forward pass runs here.

## Verdict
REJECTED per the frozen rule. Held-out median recall@4 = 0.489 (< 0.5), so the "< 0.5 anywhere" clause fires; OOD reaches 0.648-0.652. The signal is real (650x above the ~0.001 random floor) but the sidecar-only partial logit fails exactly on the hard cases whose match lives in the content branch. recall@16 sits near 0.80-0.85 (recorded as data, not adjudicated). Boundary layers 3/26 collapse to 0.17-0.45; middle layers 7-23 reach 0.53-0.73. A cheap 4-row exact trigger does not exist at this bar.
