---
name: prereg09-learned-static-selector
description: Tests whether a tiny trained per-token scorer beats the hand-crafted sidecar statistic as a static NoPE eviction selector; run when evaluating learned vs. hand-crafted salience.
---

# PREREG9 — Learned static selector (safe form of Recommendation ①)

**PREREG**: PREREG9.md | **Output**: out/learnedsel_8192.json | **Phase**: A

## What it measures
Whether a tiny standalone per-layer MLP scorer (576->32->1, hinge/pairwise-ranking trained on offline attention mass) can replace the hand-crafted sidecar deviation-from-lowpass statistic as the eviction selector, without touching model weights.

## Frozen decision rule (if pre-registered)
Offline gate (must pass before any GPU e2e): sink-excluded recall@(T/128) on held-out fineweb must beat the sigma baseline by >= +0.05 absolute AND not lose on OOD (code+logs) by more than 0.05; fail either -> rejected offline, no e2e. E2e bar (only if offline gate passes): 24 trials, rho 1/128 (+1/32 guard), learned-selector eviction must exceed the best baseline (sel_k16/dig_r64) by >= 0.09 (2 trials) at 1/128 with no regression at 1/32.

## Reproduce
```bash

# 1. Train the scorer offline (CPU) -> out/selector.pt, out/selector_offline.json
python selector_train.py
# 2. End-to-end eviction with the trained selector (needs out/selector.pt)
python harness/e2e.py --arch kimi --seq-len 8192 --n-docs 0 --needle-trials 24 \
  --gpu-expert-layers 18 --seed 11 --rhos 32,128 \
  --ops learned_sel,sel_k16,dig_r64,recent --out out/learnedsel_8192.json
```
Requires: the mini-sglang package importable (see project README "Setup"), Kimi Linear checkpoint in the HF cache. Step 1 also needs the saved activations under out/main2/ (and out/code_kimi, out/logs_kimi for OOD).

## Verdict
Rejected — the tenth offline/e2e inversion recorded in the project. The offline gate passed 24/24 (+0.406 held-out recall win), but end-to-end the learned selector collapsed to 0.17 @1/32 and 0.00 @1/128 versus sel_k16/dig_r64 at 0.83-0.88/0.67 — a 0.71 e2e loss, the largest proxy/e2e inversion in the project. Hypothesised mechanism: the net regresses toward the training distribution's notion of importance while the needle is a training-less outlier, whereas the hand statistic is an outlier detector by construction. Recommendation ① (safe form) is closed as refuted; the hand-crafted sidecar deviation statistic remains champion.
