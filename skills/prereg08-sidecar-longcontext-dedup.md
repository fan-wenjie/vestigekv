---
name: prereg08-sidecar-longcontext-dedup
description: Tests the Theorem-1 dedup regime (does the NoPE cache collapse to few distinct rows on repetitive corpora) and validates the 11%-read sidecar selector at 32k/65k. Run to confirm the surviving deliverable and close the dedup route.
---

# Sidecar long-context validation + Theorem-1 dedup regime

**PREREG**: PREREG8.md | **Output**: out/dedup_kimi_{code,logs,prose}.json, out/sidecar_{32768,65536}.json | **Phase**: A

## What it measures
Two arms: (1) dedup regime — greedy epsilon-net leader count n_distinct/T per MLA layer on code/logs/prose, testing whether Theorem 1's free-duplicate-merge has any application regime on this model (`dedup.py`); (2) long-context validation of the surviving sidecar selector — the 11%-read (64-d sidecar) `dig_r64`/`digk64` vs the full-vector `sel_k64`/`sel_k16` at 32k and 65k.

## Frozen decision rule (if pre-registered)
Dedup: P1 (prose control) n_distinct/T ~ 1 at tau <= 0.05 on both models. P2 (Kimi on code/logs) material collapse = n_distinct/T <= 0.5 at tau=0.05 in >= half the MLA layers. P3 (DSV2 on same tokens) stays > 0.8. If P2 fails, Theorem 1 has NO application regime on this model (record plainly, no e2e follow-up). If P2 and P3 hold, e2e arm bar is dCE within 0.02 of uncompressed at the achieved dedup ratio. Sidecar: the 11%-read sidecar selector PASSES if its intact rate ties the full-vector variant (zero gap) at the tested rho; fail -> revert to full-vector reads at long context.

## Reproduce
```bash

# build the repetitive-corpus dumps
python extract.py --arch kimi --seq-len 8192 --n-docs 1 --n-queries 512 \
  --gpu-expert-layers 14 --max-lm-loss 99 --seed 1 --corpus-file corpus/code.txt --out out/code_kimi
python extract.py --arch kimi --seq-len 8192 --n-docs 1 --n-queries 512 \
  --gpu-expert-layers 14 --max-lm-loss 99 --seed 1 --corpus-file corpus/logs.txt --out out/logs_kimi
# dedup n_distinct/T per layer  (args: DATA tag native [doc] [outfile])
python dedup.py out/code_kimi code nope doc000.pt out/dedup_kimi_code.json
python dedup.py out/logs_kimi logs nope doc000.pt out/dedup_kimi_logs.json
python dedup.py out/main2      prose nope doc000.pt out/dedup_kimi_prose.json   # fineweb prose control
# sidecar selector long-context validation
python harness/e2e.py --arch kimi --seq-len 32768 --n-docs 0 --needle-trials 12 \
  --gpu-expert-layers 16 --seed 11 --rhos 32,128 \
  --ops dig_r64,digk64,sel_k64,sel_k16,recent --out out/sidecar_32768.json
python harness/e2e.py --arch kimi --seq-len 65536 --n-docs 0 --needle-trials 12 \
  --gpu-expert-layers 14 --seed 11 --rhos 32,128 \
  --ops digk64,sel_k64,recent --out out/sidecar_65536.json
```
Requires: the mini-sglang package importable (see project README "Setup"), Kimi Linear checkpoint in the HF cache, and corpus/code.txt & corpus/logs.txt for the dedup arm.

## Verdict
Dedup P2 fails ~10x: even on logs the cache rows stay pairwise distinct (n_distinct/T well above 0.5) because deep-layer context mixing destroys duplicates. Theorem 1 stands as mathematics but has NO application regime on this model — the dedup route is closed with no e2e follow-up. The sidecar selector PASSES at zero gap: the 11%-read `digk64`/`dig_r64` ties the full-vector `sel_k64`/`sel_k16` at 8k, 32k, and 65k. The 11%-read sidecar is the surviving deliverable.
