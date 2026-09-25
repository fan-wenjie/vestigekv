---
name: m6-bpb
description: Bits-per-byte on the production cache form — parity leg (engine vs HF) and quality leg (vestige dCE) over the decode path.
---

# M6 bits-per-byte (production cache form)

**PREREG**: none | **Output**: out/m6_bpb.json | **Phase**: B

## What it measures
Deployment-path bpb on the production cache form (L=8192): prefill the first half as context (chunked, policy closing blocks, calibration collected), build tier-2, then teacher-forced DECODE over the second half so every scored token goes through the deployed stack. Arms: hf (one-shot ruler), engine (uncompressed, same decode path), vestige (rho=1/32 eviction + tier-2 r=64/topj=16).

## Frozen decision rule (if pre-registered)
Frozen in the script. Parity leg: |bpb(engine)-bpb(hf)| median <= 1.5 x floor_mean_absdiff/ln2 x tok_per_byte (floor read from out/m3_floor.json). Quality leg: report dCE(vestige-engine); |dCE| median <= 0.02 nats PASS, <= 0.05 MARGINAL, above FAIL. Audit rule 1: >= 6 docs scored or ABORT.

## Reproduce
```bash

python mexp/m3_floor.py    # writes out/m3_floor.json (parity-bar input) if absent
python mexp/m6_bpb.py
```
Requires: the mini-sglang package importable (see project README "Setup"), Kimi Linear checkpoint in the HF cache. Phase C (dual-machine) also needs the second box reachable with the gloo transport — see mexp/README.md.

## Verdict
PASS — engine bpb matches HF to floor resolution (parity leg) and vestige dCE stays within the 0.02-nat quality band at rho=1/32. Per-doc rows in out/m6_bpb.json.
