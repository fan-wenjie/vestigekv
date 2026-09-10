---
name: prereg24-lse-compatible-gate
description: Tests whether a confidence signal derivable from a flash-kernel's logsumsexp (conf = s_max - LSE) can replace the PREREG18 entropy gate, which needs attention weights that flash kernels do not expose; run to unblock the gate for production deployment.
---

# PREREG24 — LSE-compatible gate (conf = smax - LSE)

**PREREG**: PREREG24.md | **Output**: out/lse_trigger.json | **Phase**: A

## What it measures
Flash-class kernels expose per-query logsumexp but not attention weights, so the entropy gate cannot be computed in production. This tests the substitute conf = s_max - LSE(kept) (= log p_max <= 0; gate fires when conf is LOW), under the exact PREREG18 protocol: calibrate the threshold for hard-recall 0.97 on doc006 with the auto-disable rule (cal fire > 60% -> off), then evaluate the cascade (gate -> conformal index scan) on doc007 + OOD.

## Frozen decision rule (if pre-registered)
LSE gate within 5pp of the entropy gate's rate at recall within 0.03 (entropy reference: test gate 0.334 / cascade recall 0.905) validates the swap and resolves ENGINEERING.md item 8 to "use conf". Fails -> deployment guidance stays "run gateless".

## Reproduce
```bash

python lse_trigger.py
```
Requires: the mini-sglang package importable (see project README "Setup"), Kimi Linear checkpoint in the HF cache. Reads pre-extracted latents from out/main2 (basis docs 000-005; CAL doc006, TEST doc007) and OOD out/code_kimi, out/logs_kimi produced by extract.py; the per-layer table prints to stdout (no model forward runs here).

## Verdict
PASS, numerically indistinguishable from the entropy gate. Test cascade gate 0.334 (entropy: 0.334, 0pp gap), recall 0.913 (entropy: 0.905); gate-alone 0.205/0.931 vs entropy's 0.214/0.931. The OOD weakness is identical in shape (layers 23/26). The LSE-derived confidence is a drop-in substitute, so ENGINEERING.md item 8 resolves to: use conf = smax - LSE from the kernel's existing outputs. (A separate lever-1 tt-r192 check recorded recovery 1.00 but fetch/q ~20 per layer on needle workloads, not the predicted 1-4 — that 1-4 figure is perplexity-workload-only.)
