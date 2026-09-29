#!/usr/bin/env python3
"""Is the certified leakage a usable signal, and what would acting on it cost?

    python mexp/kimi/analyze_leak.py [--dir results/kimi/stepattr] [--eps 0.01]

Three questions in the order that makes the later ones worth asking. Each is
answered from the step dump's own records, which carry both the quantity a
runtime rule could see and the truth it is supposed to bound.

  1. TIGHTNESS. leak_cert is the log-sum-exp of the CERTIFIED scores of the
     rows the scan did not fire, against the kept log-sum-exp; leak_true is
     the same thing computed from the exact scores. The bound is sound by
     construction, so leak_cert >= leak_true always. What matters is by how
     much: a bound loose by six orders of magnitude is the ln(N/eps) failure
     again -- provably safe, and vacuous at 99.996% fallback.

  2. SEPARATION. A signal is only a signal if its distribution differs between
     the regimes it is meant to tell apart. The label here is not a guess: the
     step dump records n_beat_max1, the number of archived rows that TRULY beat
     the kept maximum on that step, so "this step needed the archive" is
     measured rather than assumed. hard_rate failed exactly this test (0.047
     against 0.031) and it failed it after being implemented, not before.

  3. PRICE. If the rule is "fall back when leak_cert > eps", then over the
     recorded steps: what fraction falls back, and what true leakage survives
     on the steps it does not? A rule that never fires costs nothing and buys
     nothing; one that always fires is dense attention with extra steps.

Reports per workload separately, because the whole point is the contrast
between a copy and ordinary generation.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics


def load(d):
    by = {}
    for p in sorted(glob.glob(os.path.join(d, "stepattr_*.jsonl"))):
        tag = os.path.basename(p).split("stepattr_", 1)[1].rsplit("_tp", 1)[0]
        rows = []
        for line in open(p):
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
        by.setdefault(tag, []).extend(rows)
    return by


def q(v, p):
    if not v:
        return float("nan")
    s = sorted(v)
    return s[min(len(s) - 1, int(p * len(s)))]


def tightness(rows):
    pairs = [(r["leak_cert_p50"], r["leak_true_p50"]) for r in rows
             if r.get("leak_cert_p50") is not None and r.get("leak_true_p50") is not None]
    pairs = [(c, t) for c, t in pairs if t > 0 and c > 0]
    if not pairs:
        return None
    ratios = [c / t for c, t in pairs]
    viol = sum(1 for c, t in pairs if c < t * (1 - 1e-6))
    return {
        "n": len(pairs),
        "ratio_p50": q(ratios, 0.5), "ratio_p90": q(ratios, 0.9),
        "ratio_max": max(ratios),
        "log10_p50": math.log10(q(ratios, 0.5)),
        "unsound": viol,
        "cert_p50": q([c for c, _ in pairs], 0.5),
        "true_p50": q([t for _, t in pairs], 0.5),
    }


def separation(rows, key="leak_cert_p50"):
    """Distribution of the signal, split by whether the step needed the archive."""
    need = [r[key] for r in rows if r.get(key) is not None and r.get("n_beat_max1", 0) > 0]
    idle = [r[key] for r in rows if r.get(key) is not None and r.get("n_beat_max1", 0) == 0]
    if not need or not idle:
        return None
    # AUC by the rank form of Mann-Whitney, which needs no binning
    allv = sorted([(v, 1) for v in need] + [(v, 0) for v in idle])
    ranks, i = {}, 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        r = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks.setdefault(k, r)
        i = j + 1
    rsum = sum(ranks[k] for k, (_, lab) in enumerate(allv) if lab == 1)
    n1, n0 = len(need), len(idle)
    auc = (rsum - n1 * (n1 + 1) / 2) / (n1 * n0)
    return {"n_need": n1, "n_idle": n0, "auc": auc,
            "need_p50": q(need, 0.5), "idle_p50": q(idle, 0.5),
            "need_p10": q(need, 0.1), "idle_p90": q(idle, 0.9)}


def price(rows, eps):
    have = [r for r in rows if r.get("leak_cert_p50") is not None]
    if not have:
        return None
    fires = [r for r in have if r["leak_cert_p50"] > eps]
    rest = [r for r in have if r["leak_cert_p50"] <= eps]
    return {
        "fallback_frac": len(fires) / len(have),
        "true_leak_p50_kept": q([r["leak_true_p50"] for r in rest], 0.5) if rest else float("nan"),
        "true_leak_max_kept": max((r["leak_true_max"] for r in rest), default=float("nan")),
        "missed": sum(1 for r in rest if r.get("n_beat_max1", 0) > 0),
        "n_rest": len(rest),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/kimi/stepattr")
    ap.add_argument("--eps", type=float, default=0.01)
    a = ap.parse_args()
    by = load(a.dir)
    if not by:
        raise SystemExit(f"ABORT: no stepattr_*.jsonl under {a.dir}")

    for tag, rows in sorted(by.items()):
        print(f"\n=== {tag}   {len(rows)} step-layer records")
        t = tightness(rows)
        if t is None:
            print("  tightness: no record carries both leakages "
                  "(an older server wrote these; rerun on the current tree)")
            continue
        print(f"  tightness   cert/true  p50 {t['ratio_p50']:.3g}  p90 {t['ratio_p90']:.3g}"
              f"  max {t['ratio_max']:.3g}   (10^{t['log10_p50']:.1f} at the median)")
        print(f"              cert_p50 {t['cert_p50']:.3g}   true_p50 {t['true_p50']:.3g}"
              f"   soundness violations {t['unsound']}/{t['n']}")
        s = separation(rows)
        if s:
            print(f"  separation  AUC {s['auc']:.3f}   "
                  f"needed-archive p50 {s['need_p50']:.3g} (n={s['n_need']})   "
                  f"idle p50 {s['idle_p50']:.3g} (n={s['n_idle']})")
            print(f"              overlap: needed p10 {s['need_p10']:.3g} "
                  f"vs idle p90 {s['idle_p90']:.3g}")
        else:
            print("  separation  one class is empty; no contrast on this workload")
        p = price(rows, a.eps)
        if p:
            print(f"  price @eps={a.eps}   fallback {100 * p['fallback_frac']:.1f}% of steps;"
                  f" on the rest true leak p50 {p['true_leak_p50_kept']:.3g},"
                  f" max {p['true_leak_max_kept']:.3g}")
            print(f"              steps kept that DID need the archive: "
                  f"{p['missed']}/{p['n_rest']}")
    print("\nAUC 0.5 is no signal at all; hard_rate, the detector already in the "
          "tree, sat there. A ratio far above 1 at the median is the ln(N/eps) "
          "failure in a new costume: sound and unusable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
