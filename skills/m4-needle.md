---
name: m4-needle
description: VestigeKV tier-1 eviction in the KimiEngine deploy loop — passcode-needle retrieval at L=8192, rho=1/32; run after M3 passes.
---

# M4 needle (tier-1 policy in the deployment loop)

**PREREG**: none (PREREG23 constant-m default) | **Output**: out/m4_needle.json | **Phase**: B

## What it measures
VestigeKV tier-1 eviction policy inside the streaming KimiEngine deploy loop: fineweb filler L=8192 (T=4096), passcode needle at p in [0.1T, 0.8T), question appended after the compressed region. Blocks of 4096 close with the global constant-m rebalance (stream_rebal, PREREG23-validated), rho=1/32.

## Frozen decision rule (if pre-registered)
Frozen in the script before data: 12 trials at rho=1/32; intact (answer dNLL < 1.0 vs the uncompressed ENGINE baseline) in >= 10/12 (95% band around the harness stream figure 0.92). Gates: every trial's uncompressed baseline must itself retrieve (base NLL < 1.0) or the trial is DROPPED; > 2 drops ABORTS; the compressed run must attend < 40% of closed rows at the final step or the PASS is void.

## Reproduce
```bash

python mexp/m4_needle.py
```
Requires: the mini-sglang package importable (see project README "Setup"), Kimi Linear checkpoint in the HF cache. Phase C (dual-machine) also needs the second box reachable with the gloo transport — see mexp/README.md.

## Verdict
PASS — tier-1 eviction in the deploy loop retrieves the needle at the >= 10/12 bar with attended fraction well under 40% of closed rows (genuine compression). Rows in out/m4_needle.json.
