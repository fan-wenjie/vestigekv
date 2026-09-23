"""Paired two-arm RULER comparison with an error bar, from run_ruler.py results.

    python mexp/glm53/ruler_stats.py --a baseline --b vestigekv --n 100

compare_ruler.py prints the table and counts cells; what a merge decision needs
is whether the difference survives its own noise. Each cell is n independent
Bernoulli draws of the same task at the same length, and the two arms see the
SAME prompts (same seed, same generators), so the comparison is paired per cell
and the cell difference's variance is what matters, not each arm's.

Two numbers are reported. The macro mean over cells with its paired standard
error is the headline. The sign test over cells is the distribution-free
backstop: it uses only which arm won each cell, so a single task with a large
swing cannot carry it.

Saturated cells (both arms exactly 1.0) are counted and reported separately:
they are real evidence of no regression and no evidence of improvement, and
averaging them in drags every effect toward zero by however many of them the
task list happens to contain.
"""

import argparse
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def load(arm, n, out, tag):
    pre = f"results_{arm}_n{n}_"
    paths = sorted(
        p for p in os.listdir(out) if p.startswith(pre) and (not tag or tag in p)
    )
    if not paths:
        raise SystemExit(f"no {pre}*{tag}*.json under {out}")
    return json.load(open(os.path.join(out, paths[-1]))), paths[-1]


def cells(res):
    # lm-eval keys a cell "<length>,none" and carries a "<length>_stderr,none"
    # sibling that is the string "N/A" for these tasks.
    out = {}
    for task, row in res["results"].items():
        for k, v in row.items():
            head = str(k).split(",")[0]
            if not head.isdigit() or not isinstance(v, (int, float)):
                continue
            out[(task, int(head))] = float(v)
    return out


def binom_sign_p(wins, losses):
    """Two-sided exact sign test on the cells where the arms differ."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2**n
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="baseline")
    ap.add_argument("--b", default="vestigekv")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "glm53", "ruler"))
    args = ap.parse_args()

    ra, pa = load(args.a, args.n, args.out, args.tag)
    rb, pb = load(args.b, args.n, args.out, args.tag)
    ca, cb = cells(ra), cells(rb)
    keys = sorted(set(ca) & set(cb))
    if not keys:
        raise SystemExit("the two runs share no (task, length) cell")
    print(f"a={args.a} ({pa})\nb={args.b} ({pb})\nn={args.n}/cell, {len(keys)} cells\n")

    sat = [k for k in keys if ca[k] == 1.0 and cb[k] == 1.0]
    live = [k for k in keys if k not in set(sat)]
    diffs = [cb[k] - ca[k] for k in keys]
    live_d = [cb[k] - ca[k] for k in live]

    def stat(d, label):
        if not d:
            print(f"{label}: no cells")
            return
        m = sum(d) / len(d)
        var = sum((x - m) ** 2 for x in d) / max(1, len(d) - 1)
        se = math.sqrt(var / len(d))
        w = sum(1 for x in d if x > 0)
        l = sum(1 for x in d if x < 0)
        z = m / se if se > 0 else float("inf")
        print(
            f"{label}: cells={len(d)} mean_diff={m:+.4f} se={se:.4f} "
            f"({z:+.1f} se) b>a={w} b<a={l} tie={len(d) - w - l} "
            f"sign p={binom_sign_p(w, l):.4f}"
        )

    stat(diffs, "all cells      ")
    stat(live_d, "non-saturated  ")
    print(f"saturated (1.00/1.00 both arms): {len(sat)} cells -- no regression, no signal")

    print(f"\n{'task':<18}{'a':>8}{'b':>8}{'diff':>8}{'cells':>7}")
    tasks = sorted({t for t, _ in keys})
    for t in tasks:
        ks = [k for k in keys if k[0] == t]
        ma = sum(ca[k] for k in ks) / len(ks)
        mb = sum(cb[k] for k in ks) / len(ks)
        print(f"{t:<18}{ma:>8.3f}{mb:>8.3f}{mb - ma:>+8.3f}{len(ks):>7}")
    ma = sum(ca[k] for k in keys) / len(keys)
    mb = sum(cb[k] for k in keys) / len(keys)
    print(f"{'MEAN':<18}{ma:>8.3f}{mb:>8.3f}{mb - ma:>+8.3f}{len(keys):>7}")


if __name__ == "__main__":
    main()
