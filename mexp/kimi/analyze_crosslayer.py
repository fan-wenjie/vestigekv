#!/usr/bin/env python3
"""Can an earlier MLA layer's outcome set a later one's certificate width?

    python mexp/kimi/analyze_crosslayer.py [--dir results/kimi/stepattr]

Asked because MLA layers run SEQUENTIALLY inside one decode step, so layer L
can read layer L-1's result from the same step rather than from the previous
one. Every layer but the first would then be adapting with no staleness at
all, which is strictly better than the stale-by-one shape the architecture
otherwise forces (max1g is computed before the archive is scanned).

Three questions, and the third is the one that decides it.

  1. Does the signal work per layer? AUC of leak_cert against the step dump's
     own label, n_beat_max1 > 0 -- whether an archived row truly beat the kept
     maximum there. A signal with AUC 0.80 pooled could be 0.80 in one layer
     and chance in the rest.

  2. Is the LABEL correlated across layers within a step? If layer L-1 needing
     the archive says nothing about layer L needing it, no cross-layer scheme
     can work whatever the signal is. This is the precondition and it is a
     property of the model, not of the method.

  3. Does layer L-1's SIGNAL predict layer L's label, and how does that compare
     to layer L's own signal? Cross-layer is only worth its complexity if it
     approaches the same-layer ceiling. Prior evidence is against: sigma was
     measured not to transfer across layers on the GLM port.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import statistics


def auc(pos, neg):
    """Mann-Whitney rank AUC; 0.5 is chance, no binning."""
    if not pos or not neg:
        return None
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    i, ranks = 0, {}
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        r = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    rsum = sum(ranks[k] for k, (_, lab) in enumerate(allv) if lab == 1)
    n1, n0 = len(pos), len(neg)
    return (rsum - n1 * (n1 + 1) / 2) / (n1 * n0)


def phi(a, b):
    """Correlation of two 0/1 sequences; 0 means the layers say nothing about
    each other, which is the precondition failing."""
    n = len(a)
    if n < 2:
        return None
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va == 0 or vb == 0:
        return None
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    return cov / (va * vb) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/kimi/stepattr")
    a = ap.parse_args()

    for path in sorted(glob.glob(os.path.join(a.dir, "stepattr_*_tp0.jsonl"))):
        tag = os.path.basename(path).split("stepattr_", 1)[1].rsplit("_tp", 1)[0]
        rows = []
        for line in open(path):
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
        by = {}
        for r in rows:
            if r.get("leak_cert_p50") is None:
                continue
            by[(r["step"], r["slot"], r["lid"])] = (
                r["leak_cert_p50"], 1 if r.get("n_beat_max1", 0) > 0 else 0)
        if not by:
            print(f"\n=== {tag}: no record carries leak_cert (older server)")
            continue
        lids = sorted({k[2] for k in by})
        print(f"\n=== {tag}   {len(by)} (step, lane, layer) records, "
              f"layers {lids}")

        print("  per layer:  need-rate   AUC of leak_cert")
        for lid in lids:
            v = [by[k] for k in by if k[2] == lid]
            pos = [x[0] for x in v if x[1]]
            neg = [x[0] for x in v if not x[1]]
            u = auc(pos, neg)
            print(f"    lid {lid:<3} {len(pos)}/{len(v):<6} "
                  f"({100 * len(pos) / len(v):>5.1f}%)   "
                  f"{'n/a (one class)' if u is None else f'{u:.3f}'}")

        # 2. label agreement between consecutive layers, same step and lane
        print("\n  label correlation between consecutive layers (same step):")
        for p, c in zip(lids, lids[1:]):
            keys = [(s, sl) for (s, sl, l) in by if l == p
                    and (s, sl, c) in by]
            if len(keys) < 8:
                continue
            lp = [by[(s, sl, p)][1] for s, sl in keys]
            lc = [by[(s, sl, c)][1] for s, sl in keys]
            r = phi(lp, lc)
            print(f"    lid {p} -> {c}   n={len(keys):<6} "
                  f"phi={'n/a' if r is None else f'{r:+.3f}'}")

        # 3. previous layer's signal against this layer's label, next to this
        #    layer's own signal on the same rows -- the ceiling
        print("\n  predicting layer L's label:   from L-1's signal   from L's own")
        for p, c in zip(lids, lids[1:]):
            keys = [(s, sl) for (s, sl, l) in by if l == p
                    and (s, sl, c) in by]
            if len(keys) < 16:
                continue
            lab = [by[(s, sl, c)][1] for s, sl in keys]
            prev = [by[(s, sl, p)][0] for s, sl in keys]
            own = [by[(s, sl, c)][0] for s, sl in keys]
            a1 = auc([x for x, y in zip(prev, lab) if y],
                     [x for x, y in zip(prev, lab) if not y])
            a2 = auc([x for x, y in zip(own, lab) if y],
                     [x for x, y in zip(own, lab) if not y])
            f = lambda u: "n/a" if u is None else f"{u:.3f}"
            print(f"    lid {p} -> {c}   n={len(keys):<6} "
                  f"{f(a1):>16} {f(a2):>14}")
    print("\n  phi near 0 kills the idea whatever the signal: the layers would "
          "not be describing the same event. AUC from L-1 near 0.5, or far "
          "below L's own, says the same for the signal specifically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
