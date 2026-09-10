---
name: v4m3-serving-smoke
description: Single-machine serving smoke against the native minisgl server — boot, /stats ready, 3 concurrent /generate requests, clean shutdown.
---

# V4-M3 serving smoke (single machine)

**PREREG**: none | **Output**: none (server log at logs/v4m3_server.log; prints stats) | **Phase**: B

## What it measures
End-to-end single-machine serving smoke: launches the native minisgl.kimi.native.server (port 30080, max-reqs 4), waits until /stats answers, fires 3 concurrent /generate requests (greedy, 16 tokens), prints stats, kills the server. Confirms the deployable server boots and serves concurrent requests.

## Frozen decision rule (if pre-registered)
No frozen bar (smoke test). The script exits non-zero with "SERVER NEVER READY" if /stats does not answer within the wait window; otherwise it prints stats and reports "V4M3 SMOKE DONE".

## Reproduce
```bash

bash mexp/v4m3_smoke.sh
```
The launcher cd's into /home/user/fft/mini-sglang/python and uses the kimikv conda python directly. Requires: the mini-sglang package importable (see project README "Setup"), Kimi Linear checkpoint in the HF cache. Phase C (dual-machine) also needs the second box reachable with the gloo transport — see mexp/README.md.

## Verdict
PASS — the native server boots, answers /stats, and serves 3 concurrent greedy /generate requests before clean shutdown ("V4M3 SMOKE DONE").
