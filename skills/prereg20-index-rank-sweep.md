---
name: prereg20-index-rank-sweep
description: Sweeps the sketch rank r of the tier-2 index against fire rate, fetched rows per query, and hard recall, weighing the trigger-rate win against the shrinking A+ bandwidth advantage; run to pick the deployment sweet-spot rank.
---

# PREREG20 — Index rank vs trigger-rate sweep

**PREREG**: PREREG20.md | **Output**: out/ranksweep.json | **Phase**: A

## What it measures
Sweeps sketch rank r in {64,128,192,256,448} (index = 64 sidecar + r sketch; r=448 = exact content, the floor). Per r: calibrate z per layer on doc006 for hard-recall 0.90, then on doc007 measure fire-any rate per (q,h), fired rows per query (head-union), and hard recall — testing whether more index complexity buys a lower tier-2 trigger probability, and at what bandwidth cost.

## Frozen decision rule (if pre-registered)
The useful operating point is where fired-rows/query drops below ~8 while recall holds >= 0.88 and the index stays <= 320/576 (56%). If no r achieves that, the trade is recorded as unfavourable. The verdict must weigh rate against bandwidth explicitly (A+ per-step index read grows with r; the win shrinks from 4x at r=64 to ~1.8x at r=256).

## Reproduce
```bash

python ranksweep.py       # related: rho_trigger.py (rho sweep, same index form)
```
Requires: the mini-sglang package importable (see project README "Setup"), Kimi Linear checkpoint in the HF cache. Reads pre-extracted latents from out/main2 (basis docs 000-005; cal doc006, test doc007) produced by extract.py; the r-vs-rate table prints to stdout (no model forward runs here).

## Verdict
The knob pays; sweet spot r=128-192. Fire rate falls monotonically (r=64: 0.028, 15.2 rows/q, recall 0.950; r=128: 0.008, 3.6 rows/q, 0.931; r=192: 0.004, 1.2 rows/q, 0.974; r=256: 0.002, 1.2 rows/q, 0.962). The frozen bar is met from r=128 up, and recall unexpectedly RISES with r (tighter bounds rank better). Cost: the A+ bandwidth cut shrinks 4.1x -> 2.2x at r=192, so the case for larger r is PCIe/CPU jitter and tail-latency isolation, not average cost; the choice is hardware-dependent. These fire rates are corpus-calibrated (doc006->doc007); in-context self-calibration on harder needle contexts lands more conservative z and higher rates.
