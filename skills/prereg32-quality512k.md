---
name: prereg32-quality512k
description: 512k end-to-end recovery — passcode needle at 5 positions through the dual pipeline; does recovery hold under the 99.8%-trigger regime.
---

# PREREG32 512k quality (dual-GPU recovery)

**PREREG**: PREREG32.md | **Output**: out/quality512k.json | **Phase**: C

## What it measures
Whether RECOVERY holds at 512k under the 99.8%-trigger regime PREREG31 found. A passcode needle is planted at 5 positions (10/25/50/75/90% of a 512k fineweb prefix), the context is prefilled compressed through the dual pipeline (local 0-19 + remote 20-26, gloo), tier-2 built (rho=1/32, r=64, topj=16), and the question decoded through the vestige stack. Reports per-position answer NLL, trigger rate during answer decode, and the fetch-volume diagnostic (rows/call, rows_max, fetch fraction of the archive).

## Frozen decision rule (if pre-registered)
Frozen before data (PREREG32): intact = answer NLL < 1.0; PASS iff >= 4/5 positions intact. No uncompressed 512k baseline (the OOM the method exists to avoid); the needle format was pre-validated at 8k (M2/M4). Outcomes: PASS => the 99.8%-trigger regime still recovers, mandatory-tier-2 story holds at 512k; FAIL => hard scope limit, long-context claims cap at 65k.

## Reproduce
```bash

bash mexp/v4m7_run.sh     # -> out/quality512k.json
```
The launcher pushes vestigekv-dev to the remote 5090, starts the remote worker (minisgl.kimi.native.dist, tmux over ssh) with expandable_segments and gpu-expert-layers=3, then runs v4m7_quality512k.py locally over the gloo transport. Requires: the mini-sglang package importable (see project README "Setup"), Kimi Linear checkpoint in the HF cache, and the second box reachable — see mexp/README.md.

## Verdict
PASS 5/5 — answer NLL 0.011/0.013/0.008/0.006/0.005 at 10/25/50/75/90%; recovery holds at 512k under the 99.8% trigger regime. The fetch-volume diagnostic resolves PREREG31's open question: 99.8% TRIGGER but only 0.024% FETCH — rows_per_call ~119-132 (local)/~102-116 (remote), rows_max <= 296, ~4000x below naive MLA's 508k. Triggering is a cheap existence test; fetch stays topj-capped and sparse — no degeneration to naive MLA. Results in out/quality512k.json.
