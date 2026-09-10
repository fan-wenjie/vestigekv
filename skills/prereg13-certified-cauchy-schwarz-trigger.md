---
name: prereg13-certified-cauchy-schwarz-trigger
description: Measures the trigger rate and fetch cost of a provably-sound Cauchy-Schwarz upper-bound trigger (zero missed references by construction); run to find out whether soundness is affordable in this cache geometry.
---

# PREREG13 — Certified trigger: rate and fetch cost

**PREREG**: PREREG13.md | **Output**: out/cert_trigger.json | **Phase**: A

## What it measures
A sound two-tier trigger: index stores exact sidecar r_u, rank-64 sketch V^T c_u, and residual norm rho_u; upper bound UB_u = exact index part + ||(I-VV^T) q_c'|| * rho_u. Tier 2 is scanned only when max_u UB_u > (tier-1 max logit) - tau, tau=3.0. Soundness guarantees zero missed references, so the experiment measures only the cost: how often it triggers and how many rows it fetches.

## Frozen decision rule (if pre-registered)
Median trigger rate <= 15% AND median fetches-per-triggered-query <= 32 makes the certified design VIABLE. Trigger rate >= 50% REJECTS it. Soundness check must be exactly 100% hard-set coverage or the run is an implementation bug.

## Reproduce
```bash

python cert_trigger.py
```
Requires: the mini-sglang package importable (see project README "Setup"), Kimi Linear checkpoint in the HF cache. Reads pre-extracted latents from out/main2 (basis docs 000-005; held-out 006-007; OOD out/code_kimi, out/logs_kimi) produced by extract.py; results print to stdout (no model forward runs here).

## Verdict
REJECTED: sound but uselessly loose. Trigger rate 0.995 (held and OOD medians), median fetches ~2900-3300 of ~3968 archived rows, so the ">= 50% -> REJECTED" clause fires. Soundness confirmed exactly (hard-set coverage 1.0000 in every layer). The 512-dim content residual makes the Cauchy-Schwarz term dominate every bound. This independently demonstrates why ArkVale abandoned soundness (its r_mean/r_center radii are deliberately non-enclosing): in this cache geometry sound bounds trigger ~100%. The design space is sound-but-useless vs useful-but-unguaranteed; no known trigger here is both sound and cheap.
