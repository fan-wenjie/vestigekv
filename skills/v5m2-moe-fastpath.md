---
name: v5m2-moe-fastpath
description: Gate that fused attention plus a gather+bmm MoE decode fast-path preserves 8k needle recovery, and reports the step time against the 82 ms M1 baseline. Run to validate the MoE fast-path.
---

# V5-M2 gate: MoE decode fast-path preserves recovery

**PREREG**: none | **Output**: out/v5m2.json | **Phase**: D

## What it measures
Runs the 8k needle with fused attention + the portable MoE decode fast-path
(gather+bmm, the SM120-portable equivalent of the fused grouped-GEMM kernel),
reporting intact rate and median decode-step ms against the 82 ms M1 baseline.

## Frozen decision rule (if pre-registered)
No PREREG; script-enforced gate: `verdict=PASS` iff `intact >= TR-1` (>= 7/8
trials recovered). FAIL raises `SystemExit`. Reports `step_ms` vs the recorded
`baseline_m1_ms=82.0`. TR=8, T=4096, rho=1/128.

## Reproduce
```bash

python mexp/v5m2_gate.py
```
Requires: the mini-sglang package importable (see project README "Setup"; loaded
via `mexp/_bootstrap.py`), Kimi Linear checkpoint in the HF cache. Single-GPU;
`gpu_expert_layers=24` (24/26 MoE layers on the card, 2 on CPU — same condition
in the compared arms so the ratio is fair, fits 96 GB). Sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## Verdict
Gate: PASS when the fused-attn + MoE fast-path decode leaves the 8k needle
recovery intact (>= 7/8) while cutting the median step well below the 82 ms M1
baseline. Confirms the gather+bmm fast-path is the runnable substitute for the
fused grouped-GEMM kernel unavailable on this SM120 card (see PREREG33 CU129
note).
