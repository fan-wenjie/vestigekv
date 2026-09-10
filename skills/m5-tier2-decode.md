---
name: m5-tier2-decode
description: GPU-resident tier-2 recall in the decode loop — must recover needles tier-1 eviction alone loses, at 8k and 32k context.
---

# M5 tier-2 decode (recall tier in the decode loop)

**PREREG**: none | **Output**: out/m5_needle.json (8k), out/m5_32k.json (32k) | **Phase**: B

## What it measures
GPU-resident tier-2 recall fired per teacher-forced DECODE step (not via prefill logprob). Three arms per trial (needle protocol as M4): base (uncompressed), evict (tier-1 only at rho=1/512), tier2 (same eviction + recall tier r=64 resident, topj=16). Answers scored via decode steps so tier-2 fires each step.

## Frozen decision rule (if pre-registered)
Frozen in the script: 12 trials; tier2 intact = 12/12 -> PASS, 11/12 -> MARGINAL (investigate before M6), <=10 -> FAIL. The evict arm must lose >= 2 needles the tier2 arm recovers, else the run is VOID (proves recovery of nothing, audit rule 1). Baseline gate and drop rules as M4.

## Reproduce
```bash

python mexp/m5_needle.py                                   # 8k -> out/m5_needle.json
M5_L=32768 M5_RHO_DEN=512 M5_OUT=/home/user/fft/nope_kv/out/m5_32k.json \
  python mexp/m5_needle.py                                 # 32k -> out/m5_32k.json
```
The 32k run reads env vars M5_L (context), M5_RHO_DEN (rho = 1/den), M5_OUT (output path) — confirmed in the script. Requires: the mini-sglang package importable (see project README "Setup"), Kimi Linear checkpoint in the HF cache. Phase C (dual-machine) also needs the second box reachable with the gloo transport — see mexp/README.md.

## Verdict
PASS — tier-2 recovers the needles tier-1 eviction alone drops at both 8k and 32k, with narrow topj-capped fetch per step. Results in out/m5_needle.json and out/m5_32k.json.
