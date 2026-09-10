---
name: prereg15-arkvale-page-digests
description: Compares ArkVale-style page sphere digests head-to-head on NoPE (Kimi) vs RoPE (DSV2), measuring page-recall of the true argmax per layer; run to test whether digest-based tier-2 retrieval works better under NoPE.
---

# PREREG15 — ArkVale-style page digests: NoPE vs RoPE

**PREREG**: PREREG15.md | **Output**: out/page_8192.json | **Phase**: A

## What it measures
Pages of 32 consecutive tokens (128 pages over T=4096), a sphere digest per ArkVale Eq.1-2 (AABB center; radii r_max / r_mean), page importance q.c + r|q|; per (query, head) it ranks pages and records recall@k of the page containing the true argmax (k in {8,16,40}). Same protocol on Kimi (NoPE, unrotated) and DSV2 (RoPE, as cached), to compare digest quality across positional schemes.

## Frozen decision rule (if pre-registered)
P2 is the deciding rule: digest recall@8 higher on Kimi than DSV2 by >= 0.15 (median over layers, r_mean variant). P2 holding = "ArkVale's trigger works better under NoPE, by this much"; P2 failing = the derivation is wrong and the claim is retracted.

## Reproduce
```bash

python pagedig.py
```
Requires: the mini-sglang package importable (see project README "Setup"), Kimi Linear checkpoint in the HF cache. Reads pre-extracted latents from out/main2 (Kimi NoPE) and out/dsv2 (DSV2 RoPE, as cached) produced by extract.py; the per-layer table and the P1/P2 summary print to stdout (no model forward runs here).

## Verdict
P2 FAILS: median rec@8 delta +0.055 < +0.15, so the median claim "ArkVale's trigger is better under NoPE" is RETRACTED. P1 radius ratio 1.33x, just below the predicted 1.4-2.0 band. The real finding is layer-uniformity (post-hoc observation): RoPE digests are fine in shallow layers and catastrophic in deep ones — worst-layer rec@8 0.024 (r_mean) / 0.014 (r_max, sound) — while NoPE stays uniformly dependable, worst-layer 0.672 (r_mean) and 0.97+ everywhere for the sound r_max radius. Since end-to-end retrieval fails if the needed page is missed at any layer, the worst-layer gap (0.672 vs 0.024, ~28x) is the deployment-relevant statistic — recorded as the hypothesis PREREG15 generates, not a licensed conclusion.
