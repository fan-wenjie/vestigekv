---
name: prereg06-mechanism-cluster-derot-granite
description: Three mechanistic tests of whether NoPE-exclusivity is positional - key-space clustering, de-rotation of Kimi's branch, and the Granite second-NoPE-model union. Run to decide if the Kimi/DSV2 gap is caused by positional encoding.
---

# Mechanism: is NoPE-exclusivity positional? (cluster / de-rot / Granite)

**PREREG**: PREREG6.md | **Output**: out/cluster_8192.json, out/proj_8192.json, out/digestwidth_8192.json, out/granite_union.json, out/derot.json | **Phase**: A

## What it measures
Whether the Kimi-holds / DSV2-fails gap is caused by positional encoding, via three arms: (1) key-space clustering merge at matched storage (`cluster_n*`, `kcenter`, `kmeans_m`, plus static `clustererr.py`) — RoPE's linear-closure argument says merge should work under NoPE; (2) de-rotation — rotate Kimi's NoPE branch and see if it closes (`derot.py`, `poscorr.py`, `sharedcomp.py`); (3) Granite union — does a second NoPE-trained model (Granite-4.0-H) show the same top-1 selection union (`granite_union.py`, `e2e_granite.py`).

## Frozen decision rule (if pre-registered)
Clustering: support if matched-cost cluster survival is not below eviction while keeping all tokens; negation if clustering is significantly below eviction (softmax nonlinearity eats the linear-closure gain). Granite union: top-1 union < 10% -> mechanism stands; > 20% -> mechanism WITHDRAWN; 10-20% indeterminate. `proj_r` (pure linear projection): if 0.00 everywhere it is another refutation of the merge shape. Hook fire-count gate: exactly 1 fire per projection per layer.

## Reproduce
```bash

# key-space clustering merge (the pre-registered core experiment)
python harness/e2e.py --arch kimi --seq-len 8192 --n-docs 0 --needle-trials 12 \
  --gpu-expert-layers 18 --seed 11 --rhos 8,16,32,64 \
  --ops cluster_n1024,cluster_n512,cluster_n256,cluster_n128,sel_k16,recent \
  --out out/cluster_8192.json
python clustererr.py          # static key-space merge error, Kimi vs DSV2
# pure linear projection arm
python harness/e2e.py --arch kimi --seq-len 8192 --n-docs 0 --needle-trials 12 \
  --gpu-expert-layers 18 --seed 11 --rhos 8,16,32,64 \
  --ops proj_r,sel_k16,recent --out out/proj_8192.json
# digest width (direction-A cap)
python harness/e2e.py --arch kimi --seq-len 8192 --n-docs 0 --needle-trials 24 \
  --gpu-expert-layers 18 --seed 11 --rhos 8,32,128 \
  --ops dig_r4,dig_r8,dig_r16,dig_r32,dig_r64,sel_k16,recent --out out/digestwidth_8192.json
# de-rotation (hardcoded to out/main2 / out/dsv2)
python derot.py               # -> out/derot.json
python poscorr.py
python sharedcomp.py
# Granite union (second NoPE model)
python granite_union.py       # -> out/granite_union.json
python e2e_granite.py --seq-len 8192 --needle-trials 12 \
  --ops sel_k4,sel_k16,sel_k64,sel_coupled,recent --seed 11 --out out/granite_final.json
```
Requires: the mini-sglang package importable (see project README "Setup"), Kimi Linear (and Granite-4.0-H, DeepSeek-V2-Lite) checkpoints in the HF cache.

## Verdict
The central NoPE-exclusivity premise FAILED all three mechanistic tests. Key-space cluster merge refuted 16/16 cells — matched-storage eviction wins, and RoPE inflates cluster spread ~1.8x rather than blocking merge. De-rotation retracted as a general law: rotating Kimi's NoPE branch is harmless (does not close the gap), so the gap is not the rotation's kinematics. Granite (a second NoPE model) exceeds the 20% top-1 union in 3/4 layers, refuting the union claim. The Kimi/DSV2 gap is real but not caused by positional encoding.
