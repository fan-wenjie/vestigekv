"""How far over the cap does an overflowing layer-step sit?

The overflow fallback is load-bearing -- deleting it costs niah_multikey_3
0.82 -> 0.18 at 65k (fbablate-on/off, n=50). But `NO_OVERFLOW_FALLBACK`
truncates on EVERY overflow, including steps that fired many times W rows.
If the overflow mass is concentrated just above W, a policy that truncates
only the marginal steps and still falls back on the rest would take most of
the fallback traffic at a fraction of the quality cost.

That is a property of the data, not of a design: it is the distribution of
r = n_fired / W over overflowing layer-steps. Reported here, per layer, from
the step-attribution dumps. CPU only -- analysis must never touch a serving
GPU.
"""

import argparse
import glob
import json
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pct(xs, q):
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, help="VK_JOB whose dumps to read")
    ap.add_argument("--w", type=int, default=4096, help="recall capacity W")
    ap.add_argument("--min-arch", type=int, default=4096,
                    help="ignore steps whose archive is below this (warmup)")
    ap.add_argument("--slack", action="store_true",
                    help="also report per-layer certificate slack")
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(
        ROOT, "results/kimi/stepattr", f"stepattr_{a.job}_tp*.jsonl")))
    if not paths:
        raise SystemExit(f"no dumps for job {a.job}")

    per_layer = defaultdict(list)   # lid -> [n_fired] over live steps
    arch = defaultdict(list)
    for p in paths:
        with open(p) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("n_arch", 0) < a.min_arch:
                    continue
                per_layer[d["lid"]].append(int(d["n_fired"]))
                arch[d["lid"]].append(int(d["n_arch"]))

    W = a.w
    print(f"job={a.job}  W={W}  dumps={len(paths)}  min_arch={a.min_arch}")
    print(f"{'lid':>4} {'steps':>7} {'ovf%':>6} {'fired p50':>10} {'p90':>7} "
          f"{'arch p50':>9} | over-cap r=fired/W: {'p10':>5} {'p50':>6} {'p90':>7} {'max':>8}"
          f" | {'r<=1.25':>8} {'r<=2':>6}")
    tot_steps = tot_ovf = 0
    marg125 = marg2 = 0
    for lid in sorted(per_layer):
        fs = per_layer[lid]
        ov = [f for f in fs if f > W]
        rs = [f / W for f in ov]
        m125 = sum(1 for r in rs if r <= 1.25)
        m2 = sum(1 for r in rs if r <= 2.0)
        tot_steps += len(fs); tot_ovf += len(ov); marg125 += m125; marg2 += m2
        print(f"{lid:>4} {len(fs):>7} {100*len(ov)/max(1,len(fs)):>5.1f}% "
              f"{pct(fs,0.5):>10} {pct(fs,0.9):>7} {pct(arch[lid],0.5):>9} | "
              f"{pct(rs,0.1):>5.2f} {pct(rs,0.5):>6.2f} {pct(rs,0.9):>7.2f} "
              f"{(max(rs) if rs else float('nan')):>8.2f} | "
              f"{100*m125/max(1,len(ov)):>7.1f}% {100*m2/max(1,len(ov)):>5.1f}%")

    print(f"\nall layers: {tot_steps} live layer-steps, {tot_ovf} overflowing "
          f"({100*tot_ovf/max(1,tot_steps):.1f}%)")
    if tot_ovf:
        print(f"  of the overflows, {100*marg125/tot_ovf:.1f}% are within 1.25x W "
              f"and {100*marg2/tot_ovf:.1f}% within 2x W")
        print(f"  a marginal-only truncation at 1.25x would remove "
              f"{100*marg125/max(1,tot_steps):.1f} of the {100*tot_ovf/max(1,tot_steps):.1f} "
              f"points of fallback")
    if a.slack:
        slack(a.job)




def slack(job):
    """Per-layer certificate slack: is a heavily-overflowing layer vacuous?

    `leak_cert` is the bound's own term, `leak_true` what the exact residual
    inner product turned out to be. Their ratio is how much of the admission
    is the bound being loose rather than the rows mattering -- a layer that
    fires 14x W rows with cert/true in the tens can absorb a margin almost
    for free, one with the ratio near 1 cannot.
    """
    paths = sorted(glob.glob(os.path.join(
        ROOT, "results/kimi/stepattr", f"stepattr_{job}_tp*.jsonl")))
    agg = defaultdict(lambda: defaultdict(list))
    for p in paths:
        with open(p) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("n_arch", 0) < 4096:
                    continue
                L = agg[d["lid"]]
                ct, tr = d.get("leak_cert_p50"), d.get("leak_true_p50")
                if ct and tr and tr > 0:
                    L["ratio"].append(ct / tr)
                for k in ("coverage", "top1_attended", "zp", "qperp_rel",
                          "top1_rank_p50", "entropy"):
                    v = d.get(k)
                    if v is not None and v == v:
                        L[k].append(float(v))
    print(f"\n{'lid':>4} {'cert/true p50':>13} {'p90':>7} | {'coverage p50':>12} "
          f"{'cov p10':>8} | {'top1_att p50':>12} {'zp p50':>7} {'qperp p50':>9} "
          f"{'entropy p50':>11}")
    for lid in sorted(agg):
        L = agg[lid]
        print(f"{lid:>4} {pct(L['ratio'],0.5):>13.2f} {pct(L['ratio'],0.9):>7.2f} | "
              f"{pct(L['coverage'],0.5):>12.4f} {pct(L['coverage'],0.1):>8.4f} | "
              f"{pct(L['top1_attended'],0.5):>12.4f} {pct(L['zp'],0.5):>7.2f} "
              f"{pct(L['qperp_rel'],0.5):>9.4f} {pct(L['entropy'],0.5):>11.4f}")


if __name__ == "__main__":
    main()
