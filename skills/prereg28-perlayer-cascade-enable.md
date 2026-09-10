---
name: prereg28-perlayer-cascade-enable
description: Tests the per-layer self-calibrated cascade enable (v3) — cascade only on layers whose own calibration hard-target recall clears 0.90, flat scan elsewhere. Run to see if the cascade family survives per-layer.
---

# PREREG28: per-layer self-calibrated cascade enable (v3)

**PREREG**: PREREG28.md | **Output**: out/cascl_{8192,32768,smoke}.json | **Phase**: E

## What it measures
The per-layer variant of the closed uniform cascade: calibration queries are
split even/odd; on the CAL half each layer computes its own stage-1
trec_cal@64, and cascade is ENABLED on a layer iff trec_cal >= 0.90 (disabled
layers keep the flat 129-dim scan). The HOLDOUT half then measures
trec_holdout@64 on enabled layers. Op `ttcascl` vs `twotier` at 8k and 32k.

## Frozen decision rule (if pre-registered)
(1) e2e: `ttcascl` intact == `twotier` at rho=1/128, both 8k and 32k (12/12
PASS; 11/12 MARGINAL). (2) self-calibration soundness: every ENABLED layer must
have trec_holdout@64 >= 0.85 — any enabled layer below 0.85 -> REFUTED. Report
(no bar) enabled-layer count, per-layer trec_cal/holdout, and mean scan-bytes
ratio. One strike closes v3.

## Reproduce
```bash

python harness/e2e.py --arch kimi --seq-len 8192  --n-docs 0 --needle-trials 3  \
  --gpu-expert-layers 18 --seed 11 --rhos 128 --ops ttcascl,recent --out out/cascl_smoke.json
python harness/e2e.py --arch kimi --seq-len 8192  --n-docs 0 --needle-trials 12 \
  --gpu-expert-layers 18 --seed 11 --rhos 128 --ops ttcascl,twotier,recent --out out/cascl_8192.json
python harness/e2e.py --arch kimi --seq-len 32768 --n-docs 0 --needle-trials 12 \
  --gpu-expert-layers 18 --seed 11 --rhos 128 --ops ttcascl,twotier,recent --out out/cascl_32768.json
```
Requires: Kimi Linear checkpoint in the HF cache. HF-harness (`e2e.py`) — does
NOT need the mini-sglang package; single-GPU.

## Verdict
PASS at both lengths. e2e `ttcascl` 1.00 == `twotier` 1.00 at 8k AND 32k
(12/12 each). Soundness held: every enabled layer >= 0.85 (8k min 0.918, 32k min
0.949) — the self-calibration never enables a layer that would lose hard targets
(the frozen core). Economy decays with length: enabled 5/7 @8k -> 3/7 @32k
(layers 7 and 23 self-disable), scan ratio 0.646@8k / 0.787@32k, KV read
reduction 4.0x->5.7x@8k / 4.9x@32k. Sound at any length, value converges to the
flat scan as context grows — graceful self-measuring degradation. Ships as an
optional deployment flag, default off.
