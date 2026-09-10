---
name: prereg07-readout-rank-selector-align
description: Probes whether the selector tracks attention mass, whether the QK read-out is low-rank (feature-axis compressible), and whether K-covering the bulk works; also the sink-effect re-scoping. Run to locate what the Kimi/DSV2 gap actually is.
---

# Read-out rank, selector<->attention alignment, K-cover

**PREREG**: PREREG7.md | **Output**: out/align.json, out/qkrank.json, out/proj_8192.json, out/e5_kimi.json | **Phase**: A

## What it measures
After the NoPE premise failed 3 mechanistic tests (PREREG6), this locates the gap: (1) Spearman between the selector score and attention mass per layer (`align.py`); (2) effective rank of the QK read-out map, and whether DSV2's collapse is depth-localised (`qkrank.py`, `qeffdim.py`, `wspectrum.py`, `dissect.py`); (3) feature-axis projection `proj_r`; (4) K-covering the bulk in full 576-d vs sidecar-only selection (`kcover`, `kcover_dig`); (5) sink effect and expected-attention estimators (`sink.py`, `expattn.py`, `estimator.py`).

## Frozen decision rule (if pre-registered)
H-qk-rank: if the read-out rank does not track the depth-localised DSV2 collapse, H-qk-rank is refuted for the collapse. E5 acceptance (user criterion, both halves required): a method must WORK on NoPE and FAIL on RoPE. Kimi arm (24 trials): `kcover_dig` intact >= `sel_k16` - 0.04 at every rho in {1/8, 1/32, 1/128}. DSV2 matched-7 arm (12 trials): the same op must land <= 0.30 intact at rho=1/32 (the 0.17 leakage floor plus noise); if it exceeds 0.30 the method works under RoPE and is REJECTED as out of scope regardless of Kimi performance.

## Reproduce
```bash

# selector <-> attention-mass alignment, and QK read-out rank (out/main2, out/dsv2)
python align.py       # -> out/align.json
python qkrank.py      # -> out/qkrank.json
python qeffdim.py
python wspectrum.py
python dissect.py <hf-snapshot-dir-of-Kimi-Linear> out/dissect_kimi.json
# feature-axis projection arm (shared with PREREG6)
python harness/e2e.py --arch kimi --seq-len 8192 --n-docs 0 --needle-trials 12 \
  --gpu-expert-layers 18 --seed 11 --rhos 8,16,32,64 \
  --ops proj_r,sel_k16,recent --out out/proj_8192.json
# E5: K-cover the bulk vs sidecar-only selection
python harness/e2e.py --arch kimi --seq-len 8192 --n-docs 0 --needle-trials 24 \
  --gpu-expert-layers 18 --seed 11 --rhos 8,32,128 \
  --ops kcover_dig,kcover,kcenter,kmeans_m,sel_k16,recent --out out/e5_kimi.json
# sink effect + expected-attention estimators (out/main2, out/dsv2)
python sink.py        # -> out/sink.json
python expattn.py
python estimator.py
```
Requires: the mini-sglang package importable (see project README "Setup"), Kimi Linear and DeepSeek-V2-Lite checkpoints in the HF cache. (`dissect.py` reads the checkpoint's config.json + safetensors directly; pass the HF snapshot dir.)

## Verdict
The QK read-out is full-rank 512, so feature-axis (per-token dimension) compression scores `proj_r` = 0.00 everywhere — the eighth-shape merge refutation, and it does not track DSV2's depth collapse (H-qk-rank refuted for the collapse). `kcover` is rejected (approximate bulk covering fails; another merge refutation), while the sidecar-only selector holds. The sink effect re-scopes the earlier deviation statistic as substantially sink-detection. Net: the gap is attributable to something that is not positional encoding.
