---
name: prereg25-intra-index-cascade
description: Tests the intra-index cascade (stage-1 64-dim sidecar shortlist, stage-2 sketch+certificate on the shortlist only) as a cheaper tier-2 scan. Run to check whether sidecar dominance extends to fired-set ranking.
---

# PREREG25: intra-index cascade (sidecar shortlist)

**PREREG**: PREREG25.md | **Output**: out/cascade_8192.json | **Phase**: E

## What it measures
A two-stage tier-2 index scan: stage 1 scans only the 64-dim sidecar summand and
emits a K'-row shortlist; stage 2 computes the sketch+certificate on the
shortlist. The `ttcasc` op is compared against the flat `twotier` scan on the 8k
needle at rho=1/128, alongside offline shortlist-recall telemetry.

## Frozen decision rule (if pre-registered)
Three frozen criteria: (1) stage-1 shortlist recall of full-index fired rows
>= 0.95 at K'=64 (report K'=16/32/64/128); (2) e2e `ttcasc` intact == `twotier`
== 12/12 at 8k (11/12 MARGINAL); (3) scan bytes <= 0.55x flat. If stage-1 recall
< 0.95 at K'=128 the lever is REFUTED — no post-hoc stage stacking.

## Reproduce
```bash

python harness/e2e.py --arch kimi --seq-len 8192 --n-docs 0 --needle-trials 12 \
  --gpu-expert-layers 18 --seed 11 --rhos 128 \
  --ops ttcasc,twotier,recent --out out/cascade_8192.json
```
Requires: Kimi Linear checkpoint in the HF cache. This is an HF-harness script
(`e2e.py`), so it does NOT need the mini-sglang package; single-GPU.

## Verdict
REFUTED under the frozen rule. (1) shortlist recall FAILED catastrophically —
per-layer rec@64 0.10-0.79, max 0.90 even at K'=128 (gate-off layers 23/26 at
0.10-0.28). (2) e2e `ttcasc` 1.00 == `twotier` 1.00 PASSED. (3) scan bytes
0.504x <= 0.55 PASSED. The twist: the 11th offline/e2e divergence, direction
inverted — the offline proxy fails while e2e holds (spurious fires dominate the
fired set; the sidecar shortlist keeps the true targets and cuts noise). That
hypothesis is PREREG27's to test; here the lever is refuted and not shipped.
ICBINB Class C, 11th exhibit, first in the favorable direction.
