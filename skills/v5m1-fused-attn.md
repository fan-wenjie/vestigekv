---
name: v5m1-fused-attn
description: Gate that the SDPA-fused NoPE attention path preserves needle recovery bit-for-bit and speeds the decode step. Run before adopting the fused attention kernel.
---

# V5-M1 gate: SDPA-fused attention preserves recovery + is faster

**PREREG**: none | **Output**: out/v5m1_fused.json | **Phase**: D

## What it measures
Runs the v4m2 8k needle recovery twice in one process — reference attention vs
the SDPA-fused NoPE attention path (`ATT._FUSED` toggle) — and compares intact
rate plus median decode-step ms. Reports the per-step speedup.

## Frozen decision rule (if pre-registered)
No PREREG; the script enforces its own gate: `verdict=PASS` iff
`intact_fused == intact_ref` AND `intact_fused >= TR-1` (i.e. >= 7/8 trials
recovered, and the fused path is bit-for-bit identical in recovery). A FAIL
raises `SystemExit`. TR=8, T=4096, rho=1/128.

## Reproduce
```bash

python mexp/v5m1_fused_gate.py
```
Requires: the mini-sglang package importable (see project README "Setup"; loaded
via `mexp/_bootstrap.py`), Kimi Linear checkpoint in the HF cache. Single-GPU
(`gpu_expert_layers=18`), no CPU offload for the measured path.

## Verdict
Gate: PASS when the fused SDPA attention path leaves the needle intact rate
unchanged versus the reference path and lowers the median decode-step time
(`speedup = step_ms_ref / step_ms_fused > 1`). This qualifies the fused kernel
for the M2 stack; the recovery-equality clause guarantees the speedup is not
bought with quality.
