---
name: v5m3-batch-projection
description: Projects, from measured component costs, that the VestigeKV end-to-end decode speedup ratio rises with batch size (MoE amortizes, attention does not). Run to characterize the batch-regime payoff.
---

# V5-M3 projection: end-to-end speedup ratio rises with batch

**PREREG**: none | **Output**: out/v5m3_projection.json | **Phase**: D

## What it measures
Times the two cost components on the card — the MoE block at batch B in
{1,2,4,8,16} (amortizes with batch, shared weights) and per-request NoPE
attention full vs VestigeKV-sparse at 32k — then projects the per-step ratio
`step_full/step_vk = (26*moe_ms[B] + n_mla*B*attn) ...`. Shows the ratio grows
with batch because attention (what VestigeKV cuts) is per-request while MoE is
shared. Full continuous batching is future work.

## Frozen decision rule (if pre-registered)
No PREREG and no pass/fail bar — this is a projection from measured components.
The claim is directional: the projected `ratio` per batch is monotone
increasing. Pro6000 absolute ms are explicitly for-reference only (recorded in
the JSON `note`).

## Reproduce
```bash

python mexp/v5m3_batch_projection.py
```
Requires: the mini-sglang package importable (see project README "Setup"; loaded
via `mexp/_bootstrap.py`), Kimi Linear checkpoint in the HF cache. Single-GPU,
`gpu_expert_layers=24`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
context S=32768.

## Verdict
Projection: the end-to-end speedup ratio rises with batch (per-batch `ratio`
printed and written to `rows`), because MoE GEMM amortizes across the batch
while attention stays per-request — so as batch grows the step ratio approaches
the pure-attention ratio (~24.5x at 32k, cited in PREREG33). Absolute Pro6000
milliseconds are for reference only; the load-bearing claim is the trend.
