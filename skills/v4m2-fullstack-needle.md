---
name: v4m2-fullstack-needle
description: VestigeKV full stack on the native session (NativeSession + interceptor regate) must reproduce the validated engine's needle behavior.
---

# V4-M2 full-stack needle (native session)

**PREREG**: none | **Output**: out/v4m2_needle.json | **Phase**: B

## What it measures
VestigeKV full stack (tier-1 eviction + tier-2 recall) running on the native NativeSession with the interceptor regate, reproducing the validated KimiEngine's needle behavior. 12 trials, L=8192 (T=4096), passcode needle protocol as M4, rho=1/128 + tier-2 (r=64, topj=16); answers scored via decode steps.

## Frozen decision rule (if pre-registered)
Frozen in the script: tier2 arm intact >= 11/12 (engine/harness measured 12/12; one flake tolerated at this n). Baseline drop rule as M4 (> 2 drops ABORT). Compression witness: attended fraction < 40% of closed rows or ABORT. FAIL blocks downstream.

## Reproduce
```bash

python mexp/v4m2_needle.py
```
Requires: the mini-sglang package importable (see project README "Setup"), Kimi Linear checkpoint in the HF cache. Phase C (dual-machine) also needs the second box reachable with the gloo transport — see mexp/README.md.

## Verdict
PASS — the native-session full stack reproduces the validated engine's needle retrieval at the >= 11/12 bar with genuine compression (attended < 40% of closed rows). Rows in out/v4m2_needle.json.
