---
name: prereg31-decode512k
description: 512k dual-GPU decode wall-clock, stock vs vestige — the first real wall-clock realization of the read reduction, plus the 512k tier-2 trigger rate.
---

# PREREG31 512k decode wall clock (dual-GPU)

**PREREG**: PREREG31.md | **Output**: out/decode512k.json (parity setup out/v4m5_parity.json) | **Phase**: C

## What it measures
Dual-machine pipeline (local layers 0-19 + remote 5090 layers 20-26, gloo/CPU transport). One 512k natural-language (fineweb-edu) compressed prefill with the interceptor enabled, tier-2 built, then the full serving state snapshotted to disk on both machines; decode benchmarks reload the snapshot. Two arms on the same snapshot — A stock (interceptor disabled, full 512k-row view) vs B vestige (attended ~1/32 + index scan + fetch) — 64 timed decode steps each after 8 warmup: median/p90 step time, tok/s, ratio A/B, plus the device-side tier-2 trigger rate at 512k (AMENDMENT). Cross-machine parity is set up first by v4m5_parity.py (v4m5_run.sh) — median|dNLL| <= 1.02e-2, |mean diff| <= 1.82e-3.

## Frozen decision rule (if pre-registered)
PREREG31 measures TIME ONLY (no quality claim — arm A's cache was built under compressed prefill). Expectation stated up front: B faster, of order the byte account (2-4x at this scale); if B is NOT faster at 512k, that is a red flag to investigate publicly, not to bury. Trigger rate reported with no bar (first measurement at 512k).

## Reproduce
```bash

bash mexp/v4m5_run.sh     # cross-machine parity gate first -> out/v4m5_parity.json
bash mexp/v4m6_run.sh     # 512k decode wall clock       -> out/decode512k.json
```
Both launchers push the vestigekv-dev branch to the remote 5090, start the remote worker (minisgl.kimi.native.dist) in a tmux session over ssh with the NCCL env tuning, then run the local rank-0 script; the transport falls back to gloo (torch NCCL rendezvous hangs on this TCP link — see mexp/README.md and memory). Requires: the mini-sglang package importable (see project README "Setup"), Kimi Linear checkpoint in the HF cache, and the second box reachable.

## Verdict
4.21x speedup — stock 0.472 s/step -> vestige 0.112 s/step (8.92 vs 2.12 tok/s), matching the 4.0x byte account; the first real wall-clock realization of the read reduction. tier-2 trigger rate at 512k: 99.8% (503/504 query-calls fired) — this CONTRADICTS the length-independence assumption from the 8k rho-sweep (0.4%); the 4.21x holds despite ~full triggering because fetches are topj=16-capped narrow rows. Length-independence of the trigger economy must be scoped to <=32k. Results in out/decode512k.json.
