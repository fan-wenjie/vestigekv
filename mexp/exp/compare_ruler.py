"""Two-arm RULER table from mexp/exp/run_ruler.py result files.

    python mexp/glm53/compare_ruler.py [--a baseline] [--b vestigekv] [--n 10]

Prints one row per task with the a / b score per length, the per-length means and
the number of cells where b differs from a; the last column is the mean over
lengths. Scores are the lm-eval RULER metric (string-match / exact-match, 0-1).
"""

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def load(arm, n, out):
    paths = sorted(p for p in os.listdir(out) if p.startswith(f"results_{arm}_n{n}_"))
    if not paths:
        raise SystemExit(f"no results_{arm}_n{n}_*.json under {out}")
    r = json.load(open(os.path.join(out, paths[-1])))
    return r, paths[-1]


def cell(row, L):
    return row.get(f"{L},none", row.get(str(L)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="baseline")
    ap.add_argument("--b", default="vestigekv")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "kimi", "ruler"))
    args = ap.parse_args()
    ra, fa = load(args.a, args.n, args.out)
    rb, fb = load(args.b, args.n, args.out)
    lengths = ra["lengths"]
    tasks = [t for t in ra["tasks"] if t in ra["results"] and t in rb["results"]]
    print(f"a={args.a} ({fa}, wall {ra['wall_s'] / 60:.0f} min)  b={args.b} ({fb}, wall {rb['wall_s'] / 60:.0f} min)  n={args.n}/cell")
    head = f"{'task':18s}" + "".join(f"{L // 1024:>4d}k a/b   " for L in lengths) + "  mean a/b"
    print(head)
    sums_a = {L: 0.0 for L in lengths}
    sums_b = {L: 0.0 for L in lengths}
    worse = better = 0
    for t in tasks:
        line = f"{t:18s}"
        ma = mb = 0.0
        for L in lengths:
            a = cell(ra["results"][t], L)
            b = cell(rb["results"][t], L)
            sums_a[L] += a
            sums_b[L] += b
            ma += a / len(lengths)
            mb += b / len(lengths)
            worse += b < a - 1e-9
            better += b > a + 1e-9
            mark = " " if abs(a - b) < 1e-9 else ("-" if b < a else "+")
            line += f" {a:.2f}/{b:.2f}{mark}  "
        print(line + f"  {ma:.3f}/{mb:.3f}")
    line = f"{'mean':18s}"
    for L in lengths:
        line += f" {sums_a[L] / len(tasks):.2f}/{sums_b[L] / len(tasks):.2f}   "
    ga = sum(sums_a.values()) / len(tasks) / len(lengths)
    gb = sum(sums_b.values()) / len(tasks) / len(lengths)
    print(line + f"  {ga:.3f}/{gb:.3f}")
    print(f"cells: {len(tasks) * len(lengths)}  b<a: {worse}  b>a: {better}  (n={args.n}: one sample = {1 / args.n:.2f})")


if __name__ == "__main__":
    main()
