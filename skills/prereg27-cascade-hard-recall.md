---
name: prereg27-cascade-hard-recall
description: Tests whether the sidecar stage-1 shortlist preserves the hard targets (full-cache argmax rows outside tier-1) at 8k and 32k. Run as the second (closing) strike on the uniform cascade lever.
---

# PREREG27: cascade hard-target recall

**PREREG**: PREREG27.md | **Output**: out/casct_{8192,32768,smoke}.json | **Phase**: E

## What it measures
Whether the sidecar stage-1 top-K' shortlist preserves the rows that matter —
the hard targets (full-cache causal-argmax rows outside tier-1 that tier-2
exists to protect, already labeled by z-calibration) — rather than the whole
(spurious) fired set. Same `ttcasc` vs `twotier` arms at 8k and 32k.

## Frozen decision rule (if pre-registered)
PASS iff hard-target recall >= 0.90 at K'=64 on BOTH 8k and 32k, AND e2e
`ttcasc` intact == `twotier` at both lengths, AND scan bytes <= 0.55x. REFUTED
if hard-target recall < 0.90 at K'=128 on either length — no further metric
substitution (two strikes closes the lever). Per-layer, not aggregate (audit
discipline: freeze hardest what you want true).

## Reproduce
```bash

# smoke (3 trials) then the two full legs:
python harness/e2e.py --arch kimi --seq-len 8192  --n-docs 0 --needle-trials 3  \
  --gpu-expert-layers 18 --seed 11 --rhos 128 --ops ttcasc,twotier,recent --out out/casct_smoke.json
python harness/e2e.py --arch kimi --seq-len 8192  --n-docs 0 --needle-trials 12 \
  --gpu-expert-layers 18 --seed 11 --rhos 128 --ops ttcasc,twotier,recent --out out/casct_8192.json
python harness/e2e.py --arch kimi --seq-len 32768 --n-docs 0 --needle-trials 12 \
  --gpu-expert-layers 18 --seed 11 --rhos 128 --ops ttcasc,twotier,recent --out out/casct_32768.json
```
Requires: Kimi Linear checkpoint in the HF cache. HF-harness (`e2e.py`) — does
NOT need the mini-sglang package; single-GPU.

## Verdict
REFUTED on both legs. 8k: per-layer trec@64 mid-layers 7-23 pass 0.92-0.99 but
layer 3 = 0.663 and layer 26 = 0.842 FAIL and stay <0.90 at K'=128
(0.773/0.839); aggregate mean@64 0.894 < 0.90; e2e still 1.00 == 1.00. 32k: same
signature deeper (trec@64 layer3=0.505, layer26=0.625; both <0.90 at K'=128);
e2e 1.00 == 1.00 a third time. Two-strikes clause -> the cascade lever is
CLOSED. The no-quantization-anywhere rule (cache AND index/metadata, fp8 route
closed by rule) was frozen in this PREREG. The per-layer-enable variant is left
for PREREG28.
