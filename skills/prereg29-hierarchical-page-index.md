---
name: prereg29-hierarchical-page-index
description: Tests the hierarchical page index (P=64 interval boxes over the 129-dim index row, admissible page bound to prune whole pages before the flat scan). Run to check whether admissible page pruning actually saves scan.
---

# PREREG29: hierarchical page index for the tier-2 scan

**PREREG**: PREREG29.md | **Output**: out/page_{8192,smoke}.json | **Phase**: E

## What it measures
Archived rows grouped in pages of P=64; per page an interval box over the
129-dim index row; per query a page-bound (interval-arithmetic upper bound of
idxs + zp*cert) prunes pages whose bound <= max1, and only surviving pages are
scanned with the flat scorer. Op `ttpage` vs `twotier`. The bound is admissible
(score linear in the row), so the fired set is identical to flat by
construction — the open question is whether it prunes anything.

## Frozen decision rule (if pre-registered)
Correctness: fired-set equality vs flat >= 0.999, and e2e `ttpage` intact ==
`twotier` at 1/128, 8k AND 32k. Economy (the real question): realized scan ratio
= 2/P + s (s = page survival). PASS iff mean ratio <= 0.60; REFUTED iff mean
ratio > 0.85 (bounds too loose to prune) — one strike.

## Reproduce
```bash

python harness/e2e.py --arch kimi --seq-len 8192 --n-docs 0 --needle-trials 3  \
  --gpu-expert-layers 18 --seed 11 --rhos 128 --ops ttpage,recent --out out/page_smoke.json
python harness/e2e.py --arch kimi --seq-len 8192 --n-docs 0 --needle-trials 12 \
  --gpu-expert-layers 18 --seed 11 --rhos 128 --ops ttpage,twotier,recent --out out/page_8192.json
```
Requires: Kimi Linear checkpoint in the HF cache. HF-harness (`e2e.py`) — does
NOT need the mini-sglang package; single-GPU.

## Verdict
REFUTED at the smoke — sound and vacuous. Page survival 0.998-1.000 on every
layer; realized scan_ratio 1.029-1.031, WORSE than flat (the page-bound pass is
pure overhead). Admissibility held exactly (zero violations — the bound never
missed a fired row). The frozen null verbatim: interval boxes over 256 dims sum
per-dim maxima over 64 rows, which in high dimension exceeds every actual row's
score; the per-page max certificate scale compounds it. One-strike clause closes
route (2). The 12-trial leg-1 rerun reproduced it identically. ICBINB Class D,
third exhibit (sibling to the sound C-S trigger). Paper: the Quest-admissibility
dividend stays true but its appendix should note this box looseness at P=64.
