"""Is the workload detectable from a per-step statistic the scan already has?

The one knob the design still exposes is RECALL_TARGET, and its right value
plausibly differs by traffic: a needle lookup wants recall, continuous prose
does not (MRCR scored 0.510-0.517 under every configuration tried, so it is
insensitive). That only helps if the traffic can be told apart ONLINE, from
quantities already computed.

It can, but only as an AGGREGATE. Every earlier attempt to gate a single step
on one of these signals failed (leak_cert AUC 0.155 inverted on the branch,
qperp_rel 0.514). Classifying a STREAM is a different problem: a request runs
thousands of layer-steps, so averaging beats the per-step noise down by sqrt(K)
and a weak per-step signal becomes a usable one.

Mechanism, which matters because it is what makes the signal plausible on a
third workload: in a needle haystack the documents are mutually unrelated, so
every step's attention is peaked; in continuous prose the neighbouring tokens
are all somewhat relevant, so there are always several competitors
(n_beat_max1 median 5 against 1). It is a SHIFT of the whole distribution, not
a few spikes -- entropy's every percentile moves by 0.5-1.3 -- which is why the
window MEAN works and a lower-tail statistic buys nothing (measured: identical
AUC to three decimals).

CPU only.
"""

import argparse
import bisect
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIELDS = ("coverage", "entropy", "qperp_rel", "n_beat_max1", "hard_rate",
          "leak_cert_p50", "leak_true_p50", "top1_rank_p50")


def load(job, min_arch=4096):
    out = []
    pat = os.path.join(ROOT, "results/kimi/stepattr", f"stepattr_{job}_tp*.jsonl")
    for p in sorted(glob.glob(pat)):
        with open(p) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("n_arch", 0) < min_arch:
                    continue
                rec = {"r": int(d["n_fired"]) / int(d["n_arch"])}
                for k in FIELDS:
                    v = d.get(k)
                    rec[k] = float(v) if (v is not None and v == v) else None
                out.append(rec)
    if not out:
        raise SystemExit(f"no dumps for {job}")
    return out


def auc(a_sorted, b):
    """P(b > a); 0.5 is no separation, and a value below 0.5 means b is lower."""
    n = len(a_sorted)
    s = 0.0
    for v in b:
        lo = bisect.bisect_left(a_sorted, v)
        s += lo + 0.5 * (bisect.bisect_right(a_sorted, v) - lo)
    return s / (n * len(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="rcf-base", help="dump for workload A (prose/copy)")
    ap.add_argument("--b", default="mk131-dump", help="dump for workload B (needle)")
    ap.add_argument("--windows", default="1,10,50,200")
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    print(f"  A={args.a}: {len(A)} layer-steps    B={args.b}: {len(B)} layer-steps\n")

    print("per-step separability")
    print(f"  {'signal':<16} {'A p50':>10} {'B p50':>10} {'AUC':>7}")
    rows = []
    for nm in ("r",) + FIELDS:
        a = sorted(x[nm] for x in A if x[nm] is not None)
        b = [x[nm] for x in B if x[nm] is not None]
        if len(a) < 100 or len(b) < 100:
            continue
        u = auc(a, b)
        med = lambda v: sorted(v)[len(v) // 2]
        rows.append((abs(u - 0.5), nm, med(a), med(b), u))
    for _, nm, ma, mb, u in sorted(rows, reverse=True):
        print(f"  {nm:<16} {ma:>10.4f} {mb:>10.4f} {u:>7.3f}")

    best = sorted(rows, reverse=True)[0][1]
    print(f"\nwindow-mean separability (strongest per-step signal: {best})")
    print(f"  {'K':>5} {'AUC':>7} {'windows A/B':>14}")
    for K in (int(x) for x in args.windows.split(",")):
        for nm in (best,):
            a = [x[nm] for x in A if x[nm] is not None]
            b = [x[nm] for x in B if x[nm] is not None]
            wa = sorted(sum(a[i:i + K]) / K for i in range(0, len(a) - K + 1, K))
            wb = [sum(b[i:i + K]) / K for i in range(0, len(b) - K + 1, K)]
            print(f"  {K:>5} {auc(wa, wb):>7.3f} {f'{len(wa)}/{len(wb)}':>14}")

    print("\n  Caveats that bound every number above:")
    print("  - two workloads only. This is a line fitted to two points; a third")
    print("    workload's position is unknown, so the AUC is BETWEEN THESE TWO,")
    print("    not 'for arbitrary text'.")
    print("  - separating the classes does not give tau. That needs a tau sweep on")
    print("    EACH class, not one, because tau's optimum is not shown monotone in")
    print("    this signal.")
    print("  - this closes on concentration, not on quality. The conformal")
    print("    guarantee is per step and multi-key failure is a conjunction across")
    print("    steps, so no per-step signal observes 'will the answer be wrong'.")


if __name__ == "__main__":
    main()
