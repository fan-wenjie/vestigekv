---
name: prereg21-bpb-lm-loss
description: Characterizes general continuation loss (delta-CE nats/token, bpb) under KV compression per operator and ratio; run to answer "is compression lossy for LM loss, not just needle retrieval".
---

# PREREG21 — End-to-end LM-loss (bpb) characterization under compression

**PREREG**: PREREG21.md | **Output**: out/bpb_8192.json | **Phase**: A

## What it measures
General continuation loss (delta-CE, nats/token; bits/token = dCE/ln2; bpb further divides by ~4 bytes/token for fineweb prose) per operator and ratio on held-out fineweb docs — the number the "is bpb lossy" question needs, complementing the needle metric which only measures retrieval.

## Frozen decision rule (if pre-registered)
Characterization, not accept/reject. Frozen expectations: dCE(dig_r64) <= 0.01 at 1/8, <= 0.05 at 1/32; twotier <= dig_r64 at every ratio; recent >> both (the floor). Revision clause: if dig_r64 at 1/32 exceeds 0.10 nats/token, the "quality guaranteed" framing must be revised to needle-only, plainly.

## Reproduce
```bash

python harness/e2e.py --arch kimi --seq-len 8192 --n-docs 6 --needle-trials 4 \
  --gpu-expert-layers 18 --seed 11 --rhos 8,32,128 \
  --ops dig_r64,twotier,recent --out out/bpb_8192.json
```
Requires: the mini-sglang package importable (see project README "Setup"), Kimi Linear checkpoint in the HF cache. Trigger telemetry is archived to out/trigger_rates.json.

## Verdict
All frozen expectations met; the standard config is effectively lossless. dCE nats/token (median, 6 docs): twotier +0.0011/+0.0018/+0.0022 at 1/8-1/32-1/128; dig_r64 +0.0055/+0.0090/+0.0104; recent similar on CE but +15 on the needle — confirming CE alone cannot arbitrate retrieval. The revision clause (>0.10) was not remotely approached; bpb conversion at ~4 bytes/token gives twotier@32x ~ 0.0006 bpb.
