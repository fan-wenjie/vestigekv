---
name: v4m4-stress
description: 30-min serving stress — 8 async workers under mixed load plus periodic needle probes; checks fairness, leaks, and quality under load.
---

# V4-M4 stress (fairness / leaks / quality under load)

**PREREG**: none | **Output**: out/v4m4_stress.json | **Phase**: B

## What it measures
30-minute serving stress. The harness owns the server lifecycle (native server, port 30081, max-reqs 8): 8 async workers issue mixed ~512/~4k-token prompts (temp 0.7, vestige=True), and a prober plants a greedy passcode needle roughly every 2 min. Reports TTFT/total p50/p99, decode tok/s, error counts, slot drain, and probe hit rate.

## Frozen decision rule (if pre-registered)
Frozen bar (PORT.md V4-M4): PASS iff zero request errors AND server alive at end AND zero server-side errors AND all 8 slots free after drain AND (no probes OR probe hits >= 90% of probes). Otherwise FAIL.

## Reproduce
```bash

python mexp/v4m4_stress.py
```
Requires: the mini-sglang package importable (see project README "Setup"), Kimi Linear checkpoint in the HF cache, and aiohttp. Runs ~30 min. Phase C (dual-machine) also needs the second box reachable with the gloo transport — see mexp/README.md.

## Verdict
PASS — no request or server errors over 30 min, all slots freed after drain, and needle probe hit-rate >= 90% under concurrent load; no fairness or leak regression. Summary + rows in out/v4m4_stress.json.
