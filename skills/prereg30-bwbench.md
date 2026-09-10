---
name: prereg30-bwbench
description: Raw-tensor bandwidth microbench — stock full-row MLA read vs VestigeKV kept+index+fetch, up to 524k rows; is the ~4x byte account realizable.
---

# PREREG30 bandwidth microbenchmark

**PREREG**: PREREG30.md | **Output**: out/bwbench.json | **Phase**: C

## What it measures
Pure-tensor, CUDA-event-timed memory-system microbench (single local GPU, no model). Synthetic bf16 latent pools at T in {8k, 32k, 131k, 524k}. Arm A: stock pattern — full-row read + score ([T,576] matvec against an absorbed query), the flash-MLA read shape. Arm B: VestigeKV pattern — kept-row (rho*T) gather+score, a (64+r)-dim index GEMV over the archive rows, then a topj=16 fetch gather+score. Reports ms/step/layer, achieved GB/s, and ratio A/B per T.

## Frozen decision rule (if pre-registered)
Frozen writing rule (PREREG30): if ratio >= 3.0 at T >= 131k, the paper's reads paragraph gains one sentence with the measured ratio; if ratio < 3.0 at 131k, the limitation stands unchanged and the microbench goes to the engineering doc only with the measured number — no softening of eq:reads, no cherry-picking a friendlier T. Kernel-launch and fused-kernel overheads are explicitly out of scope (roadmap).

## Reproduce
```bash

python mexp/v4_bwbench.py
```
Single-GPU (no second box, no distributed transport needed for this microbench). Requires: the mini-sglang package importable (see project README "Setup"), Kimi Linear checkpoint in the HF cache. Phase C (dual-machine) also needs the second box reachable with the gloo transport — see mexp/README.md.

## Verdict
The byte account is realized at raw-tensor level: the A/B ratio reaches the ~4x band at large T (>= 131k rows), confirming the kept+index+fetch pattern moves ~4x fewer bytes than a stock full-row read. Per-T ms/GB-s/ratio in out/bwbench.json.
