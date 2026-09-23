#!/usr/bin/env python3
"""Is a runtime haystack detector worth building, and should it notch or fall back?

    notch_study.py <results/kimi>

Three questions, all answerable from calibration dumps already on disk, none
of them changing any behaviour:

  separability  peakiness (largest above-cutoff bin against the median one)
                over every (request, layer) of both corpora. A detector needs
                a margin between the two clouds, not a difference of means.
  notch effect  how much of tier-1's kept set survives notching the standing
                component -- the paper's number, here per layer rather than
                for layer 19 alone.
  does it help  the one the paper does not measure. The dump carries the
                request's own absorbed decode queries, so the rows those
                queries really score highest are computable: an ORACLE top-m.
                Overlap(sigma top-m, oracle) says how good the selector is;
                overlap(notched sigma, oracle) says whether notching moves it
                toward the queries or merely moves it. Deploying a detector is
                worth it only if the second is larger than the first, and only
                on the corpus that trips it.
"""
import glob, os, re, sys, statistics
import torch

sys.path.insert(0, os.path.expanduser("~/vestigekv/mexp/kimi"))
from sidecar_spectrum_offline import load  # noqa: E402

KAPPA, BLOCK, RHO, NOTCH_BINS = 16, 4096, 0.03, 6
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def sigma_blocks(side, notch=0):
    """[T'] sigma over whole blocks; notch = how many above-cutoff bins to keep."""
    out = []
    for b in range(side.shape[0] // BLOCK):
        seg = side[b * BLOCK:(b + 1) * BLOCK].to(DEV)
        f = torch.fft.rfft(seg, dim=0)
        keep = f.clone(); keep[KAPPA:] = 0
        if notch:
            e = torch.fft.rfft(seg - seg.mean(0, keepdim=True), dim=0).abs().pow(2).sum(1)[KAPPA:]
            for bin_ in (torch.topk(e, notch).indices + KAPPA).tolist():
                keep[bin_] = f[bin_]
        out.append((seg - torch.fft.irfft(keep, n=BLOCK, dim=0)).pow(2).sum(1).sqrt())
    return torch.cat(out)


def peakiness(side):
    e = []
    for b in range(side.shape[0] // BLOCK):
        seg = side[b * BLOCK:(b + 1) * BLOCK].to(DEV)
        e.append(torch.fft.rfft(seg - seg.mean(0, keepdim=True), dim=0).abs().pow(2).sum(1))
    e = torch.stack(e).mean(0)[KAPPA:]
    return float(e.max() / e.median()), int(e.argmax()) + KAPPA


def study(path):
    blob, side = load(path)
    T = (side.shape[0] // BLOCK) * BLOCK
    if T == 0:
        return None
    peak, bin_ = peakiness(side)
    s0, s1 = sigma_blocks(side, 0), sigma_blocks(side, NOTCH_BINS)
    m = max(1, int(RHO * T))
    # oracle: the rows this request's own absorbed decode queries score highest
    rows = blob["rows"][:T].to(DEV).float()
    q = blob["qcal"].to(DEV).float().reshape(-1, rows.shape[1])   # [n*H, 576]
    best = torch.full((T,), -1e30, device=DEV)
    for i in range(0, q.shape[0], 32):
        best = torch.maximum(best, (q[i:i + 32] @ rows.T).amax(0))
    oracle = set(torch.topk(best, m).indices.tolist())
    top0 = set(torch.topk(s0, m).indices.tolist())
    top1 = set(torch.topk(s1, m).indices.tolist())
    del rows, q, best
    return dict(lid=int(blob["lid"]), seq=int(blob["seq_len"]), peak=peak, bin=bin_,
                survive=len(top0 & top1) / m,
                hit0=len(top0 & oracle) / m, hit1=len(top1 & oracle) / m)


def main():
    res = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/vestigekv/results/kimi")
    conds = {"RULER haystack": ["sidecardump", "sidecardump5"],
             "real prose": ["sidecardump_prose", "sidecardump_prose5"]}
    allrows = {}
    for name, dirs in conds.items():
        rows = []
        for d in dirs:
            for p in sorted(glob.glob(os.path.join(res, d, "cal_tp0_*.pt"))):
                r = study(p)
                if r: rows.append(r)
        allrows[name] = rows
        print(f"\n== {name}: {len(rows)} (request, layer) snapshots, "
              f"{len({r['seq'] for r in rows})} requests x {len({r['lid'] for r in rows})} layers")
        print(f"{'layer':>6}{'n':>4}{'peakiness p50':>15}{'min':>9}{'max':>9}"
              f"{'peak bin':>10}{'kept survives':>15}{'sigma@oracle':>14}{'notched@oracle':>16}")
        for lid in sorted({r["lid"] for r in rows}):
            g = [r for r in rows if r["lid"] == lid]
            f = lambda k: [x[k] for x in g]  # noqa: E731
            print(f"{lid:>6}{len(g):>4}{statistics.median(f('peak')):>15.0f}"
                  f"{min(f('peak')):>9.0f}{max(f('peak')):>9.0f}"
                  f"{'/'.join(str(b) for b in sorted({x['bin'] for x in g})):>10}"
                  f"{statistics.fmean(f('survive')):>14.1%}"
                  f"{statistics.fmean(f('hit0')):>13.1%}"
                  f"{statistics.fmean(f('hit1')):>15.1%}")
    r_all = [r["peak"] for r in allrows["RULER haystack"]]
    p_all = [r["peak"] for r in allrows["real prose"]]
    print(f"\nseparability: RULER peakiness min {min(r_all):.0f}, prose max {max(p_all):.0f}"
          f"  -> {'separable, margin x%.1f' % (min(r_all)/max(p_all)) if min(r_all) > max(p_all) else 'OVERLAP: no single threshold'}")
    for name in conds:
        g = allrows[name]
        d = statistics.fmean(x["hit1"] - x["hit0"] for x in g)
        print(f"notch moves sigma toward the queries on {name}: "
              f"{statistics.fmean(x['hit0'] for x in g):.1%} -> "
              f"{statistics.fmean(x['hit1'] for x in g):.1%}  ({d:+.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
