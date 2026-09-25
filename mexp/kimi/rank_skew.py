"""Separate real compute from TP synchronization in decode profiles.

    python mexp/kimi/rank_skew.py results/kimi/profile/<job>/ctx256k [--steps 200]

Takes a trace prefix (the tool appends -TP-0 / -TP-1 .trace.json.gz) and reports,
per rank and per step: GPU time in collectives (ncclDevKernel_*) and GPU time in
everything else. A collective's duration absorbs the wait for the slower rank, so
it is not work: with VestigeKV the two ranks fire and fetch different row sets and
the imbalance lands there. `compute` (non-collective kernel time) is what a code
change actually costs, and `max compute + transfer floor` is what the step cannot
go below; the per-rank spread of compute is the skew the collectives absorb.

Several prefixes can be passed: each is one row, so two trees or two contexts are
compared with the collective time held apart from the compute.
"""

import argparse
import collections
import gzip
import json
import re


def rank_totals(path, steps):
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    events = data["traceEvents"] if isinstance(data, dict) else data
    nccl = compute = 0.0
    per = collections.Counter()
    for e in events:
        if e.get("cat") == "kernel" and e.get("ph") == "X":
            name = re.split(r"[(<]", e["name"], 1)[0].strip()
            if name.startswith("ncclDevKernel"):
                nccl += float(e["dur"])
            else:
                compute += float(e["dur"])
                per[name] += float(e["dur"])
    span = None
    ks = [e for e in events if e.get("cat") == "kernel"]
    if ks:
        span = (max(e["ts"] + e["dur"] for e in ks) - min(e["ts"] for e in ks)) / steps
    return nccl / steps, compute / steps, span, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefixes", nargs="+", help="trace prefixes, e.g. results/kimi/profile/<job>/ctx256k")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--ranks", default="0,1")
    ap.add_argument("--top", type=int, default=0, help="also print the N largest compute kernels of rank 0")
    args = ap.parse_args()
    ranks = [int(r) for r in args.ranks.split(",")]
    print(f"{'trace':>44s}" + "".join(f"{'TP' + str(r) + ' nccl':>12s}{'TP' + str(r) + ' compute':>14s}" for r in ranks)
          + f"{'skew':>8s}{'step':>9s}")
    rows = {}
    for p in args.prefixes:
        cells, comps, span = "", [], None
        for r in ranks:
            nccl, comp, sp, per = rank_totals(f"{p}-TP-{r}.trace.json.gz", args.steps)
            cells += f"{nccl:12.1f}{comp:14.1f}"
            comps.append(comp)
            span = sp if span is None else span
            if r == 0:
                rows[p] = per
        skew = max(comps) - min(comps)
        print(f"{p.split('/')[-2] + '/' + p.split('/')[-1]:>44s}" + cells + f"{skew:8.1f}{span:9.1f}")
    if args.top and len(args.prefixes) == 2:
        a, b = (rows[p] for p in args.prefixes)
        names = set(a) | set(b)
        diff = sorted(((b[n] - a[n]) / args.steps, n) for n in names)
        print(f"\nrank-0 compute kernels, {args.prefixes[1].split('/')[-2]} minus {args.prefixes[0].split('/')[-2]} (us/step):")
        for d, n in diff[-args.top:][::-1] + [x for x in diff[: args.top] if x[0] < -0.05][::-1]:
            print(f"{d:10.1f}  {n[:80]}")


if __name__ == "__main__":
    main()
