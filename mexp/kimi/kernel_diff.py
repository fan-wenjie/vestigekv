"""Per-kernel GPU time per decode step, diffed between two torch-profiler traces.

    python mexp/kimi/kernel_diff.py results/kimi/profile/<a>/ctx128k-TP-0.trace.json.gz \
                                     results/kimi/profile/<b>/ctx128k-TP-0.trace.json.gz [--steps 200] [--top 30]

Sums the duration of every GPU kernel event (chrome-trace cat "kernel") by kernel
name, divides by the number of profiled steps, and prints both trees side by side
sorted by the difference (b - a), plus the totals: the per-step GPU time each tree
spends and where the difference sits. Kernel names are cut at the first '(' or '<'.
"""

import argparse
import collections
import gzip
import json
import re


def load(path):
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    events = data["traceEvents"] if isinstance(data, dict) else data
    per = collections.Counter()
    calls = collections.Counter()
    for e in events:
        if e.get("cat") == "kernel" and e.get("ph") == "X":
            name = re.split(r"[(<]", e["name"], 1)[0].strip()
            per[name] += float(e["dur"])
            calls[name] += 1
    return per, calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--label-a", default="a")
    ap.add_argument("--label-b", default="b")
    args = ap.parse_args()
    a, ca = load(args.a)
    b, cb = load(args.b)
    names = set(a) | set(b)
    rows = sorted(((b[n] - a[n]) / args.steps, n) for n in names)
    ta, tb = sum(a.values()) / args.steps, sum(b.values()) / args.steps
    print(f"GPU kernel time per step (us): {args.label_a}={ta:.1f}  {args.label_b}={tb:.1f}  diff={tb - ta:+.1f}  ({len(names)} kernels)")
    print(f"{'diff us/step':>13s} {args.label_a:>10s} {args.label_b:>10s} {'calls/step':>12s}  kernel")
    shown = rows[-args.top:][::-1] + [r for r in rows[: args.top] if r[0] < 0][::-1]
    seen = set()
    for d, n in shown:
        if n in seen:
            continue
        seen.add(n)
        print(f"{d:13.1f} {a[n] / args.steps:10.1f} {b[n] / args.steps:10.1f} {ca[n] / args.steps:5.1f}/{cb[n] / args.steps:<5.1f}  {n[:90]}")
    only_a = [n for n in names if n not in b]
    only_b = [n for n in names if n not in a]
    if only_a:
        print(f"only in {args.label_a}: " + ", ".join(sorted(only_a))[:600])
    if only_b:
        print(f"only in {args.label_b}: " + ", ".join(sorted(only_b))[:600])


if __name__ == "__main__":
    main()
