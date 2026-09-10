---
name: v5-profile-step
description: Profiles a single 8k NoPE-MLA decode step to show where the ~82 ms goes (MoE GEMM dominates). Run to justify the V5 optimization targets.
---

# V5 decode-step profile (where the 82 ms goes)

**PREREG**: none | **Output**: none (prints a torch-profiler table) | **Phase**: D

## What it measures
A torch.profiler CPU+CUDA breakdown of one 8k-context decode step of the native
Kimi Linear + VestigeKV session, to identify the dominant cost. It confirms the
MoE GEMM (not attention) dominates the step, motivating the M1/M2/M3 work.

## Frozen decision rule (if pre-registered)
None — this is profiling, not a gate. No bar; the script only prints the
`key_averages()` table sorted by `cuda_time_total` (top 12 rows).

## Reproduce
```bash

python mexp/v5_profile_step.py
```
Requires: the mini-sglang package importable (see project README "Setup"; loaded
via `mexp/_bootstrap.py`), Kimi Linear checkpoint in the HF cache. Single-GPU,
all-GPU — `gpu_expert_layers=1000` puts every MoE expert on the card, no CPU
offload. rho=1/32, prefill 4096 tokens, 10 profiled decode steps.

## Verdict
Profiling result, no pass/fail: the ~82 ms 8k decode step is dominated by the
MoE GEMM, not attention. This is the baseline that the V5 milestones (fused
attention M1, MoE fast-path M2, batch projection M3) target — and the reason
cap removal is nearly free at batch=1 (PREREG33): attention is a small fraction
of the step.
