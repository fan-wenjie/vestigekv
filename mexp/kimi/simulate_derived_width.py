#!/usr/bin/env python3
"""What the derived-width arm WOULD choose, replayed on recorded step dumps.

    python mexp/kimi/simulate_derived_width.py [--dir results/kimi/stepattr]

The controller consumes only the fired-row count and the archive length, and
the step dump records both, so its decisions can be replayed exactly without a
GPU. Written to predict the delta sweep BEFORE it lands: a prediction that
arrives after the numbers is worth much less, and this one is cheap.

It reports, per layer, the derived target rho = 1 - delta/(2*hbar*T) and
whether that target is REACHABLE at the calibration sample size. The second is
the interesting column: an order statistic over n points cannot express a
target past n/(n+1), and an answer-level spec over a few hundred steps demands
far tighter than that, so the expectation is that the binding layers pin at the
sample maximum and only the idle ones move.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os


def wilson_upper(a: float, n: float, z: float) -> float:
    if n <= 0:
        return 1.0
    c = (a + 0.5 * z * z + z * math.sqrt(a * (n - a) / n + 0.25 * z * z)) / (n + z * z)
    return min(1.0, max(0.0, c))


def normal_quantile(p: float) -> float:
    # Acklam, same form the controller uses; adequate for a report
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/kimi/stepattr")
    ap.add_argument("--deltas", default="0.5,0.05,0.001")
    ap.add_argument("--n-cal", type=int, default=64,
                    help="calibration sample size; caps rho at n/(n+1)")
    args = ap.parse_args()
    deltas = [float(x) for x in args.deltas.split(",")]
    cap = args.n_cal / (args.n_cal + 1.0)

    for path in sorted(glob.glob(os.path.join(args.dir, "stepattr_*_tp0.jsonl"))):
        tag = os.path.basename(path).split("stepattr_", 1)[1].rsplit("_tp", 1)[0]
        seq = collections.defaultdict(list)
        for line in open(path):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("n_arch") is None or r.get("n_fired") is None:
                continue
            seq[(r["slot"], r["lid"])].append((r["step"], r["n_fired"]))
        if not seq:
            continue
        print(f"\n=== {tag}   cap = n/(n+1) = {cap:.4f}")
        print("   lid    steps   fire-rate   " +
              "   ".join(f"rho(d={d:g})  reach" for d in deltas))
        per_lid = collections.defaultdict(list)
        for (slot, lid), rows in seq.items():
            per_lid[lid].append(sorted(rows))
        for lid in sorted(per_lid):
            runs = per_lid[lid]
            cells = []
            for d in deltas:
                z = normal_quantile(1.0 - 0.5 * d)
                rhos, reach = [], 0
                for rows in runs:
                    a = n = 0
                    for _, nf in rows:
                        a += 1 if nf > 0 else 0
                        n += 1
                    h = wilson_upper(a, n, z)
                    rho = 1.0 - 0.5 * d / max(h * n, 1e-9)
                    rho = min(1.0, max(0.0, rho))
                    rhos.append(rho)
                    reach += 1 if rho <= cap else 0
                m = sum(rhos) / len(rhos)
                cells.append(f"{m:.5f}  {100*reach/len(rhos):3.0f}%")
            tot = sum(len(r) for r in runs)
            fr = sum(1 for r in runs for _, nf in r if nf > 0) / max(tot, 1)
            print(f"   {lid:<5}  {tot:>6}   {100*fr:>7.1f}%   " + "   ".join(cells))
    print("\n  reach% is the fraction of (lane, layer) runs whose derived target "
          "is\n  expressible at this sample size. Anything below it pins at the "
          "sample\n  maximum, which is the build's own widest certificate -- no "
          "narrowing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
