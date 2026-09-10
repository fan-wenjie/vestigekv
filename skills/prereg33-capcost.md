---
name: prereg33-capcost
description: Measures the wall-clock cost of removing the topj/512 tier-2 fetch cap (capped topj=16 vs uncapped), all-GPU, at 8k and 32k. The last experiment; run to settle whether the cap buys batch=1 speed.
---

# PREREG33: cost of removing the topj/512 fetch cap (LAST experiment)

**PREREG**: PREREG33.md | **Output**: out/capcost.json | **Phase**: D

## What it measures
Two arms on the same needle context, all-GPU, both fully open for ratio
fairness: `capped` (topj=16, bounded fetch) vs `uncapped` (topj=1e9, fetch =
full fire-set), at 8k and 32k. Reports median decode-step ms each arm,
`degradation = ms_uncapped/ms_capped - 1`, and re-checks recovery on both.
Companion pre-check `mexp/v5_capcheck.py` measures whether the topj=16 cap even
binds (fire-union rows/query vs the 512 union cap).

## Frozen decision rule (if pre-registered)
No bar (PREREG33 is a characterization, not a gate): report degradation and the
fetch-volume ratio at each length, with recovery re-checked identical on both
arms (the cap trims only sub-top-16 marginal fires). Frozen expectation (from
run 080): uncapped fetch blows up ~10-13x on worst steps, modest step effect at
8k, growing toward the fetch ratio at long context.

## Reproduce
```bash

python mexp/v5_capcheck.py      # pre-check: does the topj=16 cap bind?
python mexp/v5_capcost.py       # the two-arm wall-clock measurement -> out/capcost.json
```
Requires: the mini-sglang package importable (see project README "Setup"; loaded
via `mexp/_bootstrap.py`), Kimi Linear checkpoint in the HF cache. Single-GPU;
`v5_capcost.py` uses `gpu_expert_layers=24` with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, contexts (8k,T=4096) and
(32k,T=16384), 4 trials/arm, rho=1/128; `v5_capcheck.py` is all-GPU
(`gpu_expert_layers=1000`), 8k. Note: the CU129/flashinfer upgrade (PREREG33)
found no SM120 prebuilt `sgl_kernel` on this Blackwell card, so the gather+bmm
fast path is used here; H100/H200 would use `fused_experts_impl`.

## Verdict
Cap removal is NEARLY FREE at batch=1, <=32k: 8k 40.5->40.1 ms (-1%, no effect),
32k 40.3->41.8 ms (+4%); recovery 4/4 both arms both lengths. This CORRECTS the
earlier "cap is wall-clock-load-bearing" claim — at batch=1 short/mid context on
Pro6000, attention is a small fraction of the ~40 ms step (MoE GEMM dominates),
so even the uncapped worst-case fetch (fire max 3983 rows) is swamped. Keep the
cap as the theoretical bounded-fetch / no-degeneration guarantee and for the
long-context + large-batch regime where attention dominates (unmeasured — no
wall-clock claim there).
