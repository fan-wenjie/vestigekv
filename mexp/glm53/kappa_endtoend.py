"""The optimal kappa for the CONSTANT-rho scheme, judged on the rows actually attended.

Earlier kappa grids scored the tier-1 kept set alone. That is the wrong criterion on
this arm: tier 1 alone scores 0.000 on multikey and tier 2 does the retrieving. kept
has two jobs -- be attended, AND set the tier-2 threshold, which is the q0.75 quantile
of the kept GROUPS' indexer scores. A kappa that picks high-scoring rows raises that
quantile and suppresses the firing that does the work, so the two jobs pull opposite
ways and only the end-to-end row set can rank kappa.

So for each kappa: kept = top-32 groups by sigma_kappa, tau = q0.75 of their scores,
fired = groups scoring above tau (capped at the budget-derived fence), and the figure
of merit is the real attention mass over kept + fired + sinks + unclosed tail, with
|fired| as the paired cost.

Runs on the RE-COLLECTED dumps (cd3_*), where qidx is distinct per step.
"""

import argparse, glob, sys, collections
import torch

sys.path.insert(0, "mexp/glm53")
from dsa_selection_stats import score, sigma_at


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", default="cd3_64k")
    ap.add_argument("--kappas", default="1,2,4,8,16,32")
    ap.add_argument("--kept-groups", type=int, default=32)
    ap.add_argument("--branch-qs", default="0.50,0.60,0.70,0.75,0.80,0.85,0.90")
    ap.add_argument("--fence-groups", type=int, default=223)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--max-q", type=int, default=4)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--sinks", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(8)
    kappas = [int(x) for x in a.kappas.split(",")]
    bqs = [float(x) for x in a.branch_qs.split(",")]

    files = []
    for d in a.dirs.split(","):
        files += sorted(glob.glob(f"results/glm53/{d}/*.pt"))
    files = files[: a.limit] if a.limit else files
    acc = collections.defaultdict(list)
    nq = 0
    for fp in files:
        d = torch.load(fp, map_location="cpu", weights_only=False)
        gk = d["index_k"]["deq"].float()
        pool = int(d["index_k"]["pool_size"])
        G = gk.shape[0]
        rows = d["rows"].float()
        Tn = rows.shape[0]
        sc = float(d["scale"])
        nb = G // a.block
        if nb == 0:
            continue
        C = nb * a.block
        qs, gates, qc = d["qidx"], d["qgate"], d["qcal"].float()
        if not isinstance(qs, (list, tuple)):
            continue
        n = min(a.max_q, len(qs), qc.shape[0])
        sig = {k: sigma_at(gk, a.block, k)[:C] for k in kappas}
        tail = torch.arange(C * pool, Tn)          # un-closed, attended unconditionally
        for i in range(n):
            s = score(qs[i], gates[i], gk)
            full = torch.softmax(qc[i] @ rows.T * sc, dim=-1)   # [H, Tn]
            nq += 1
            for k in kappas:
              kept = sig[k].topk(min(a.kept_groups, C)).indices
              for bq in bqs:
                tau = float(s[kept].quantile(bq))
                cand = (s[:C] > tau)
                cand[kept] = False
                nfire = int(cand.sum())
                fired = cand.nonzero().flatten()
                key = f"{k}|{bq}"
                if nfire > a.fence_groups:
                    fired = s[:C].topk(a.fence_groups).indices
                    nfire = a.fence_groups
                    acc[f"fenced@{key}"].append(1.0)
                else:
                    acc[f"fenced@{key}"].append(0.0)
                g = torch.cat([kept, fired])
                r = (g[:, None] * pool + torch.arange(pool)[None, :]).reshape(-1)
                r = r[r < Tn]
                r = torch.cat([torch.arange(min(a.sinks, Tn)), r, tail]).unique()
                m = full[:, r].sum(-1)
                acc[f"mass@{key}"].append(float(m.mean()))
                acc[f"worst@{key}"].append(float(m.min()))
                acc[f"fire@{key}"].append(float(nfire))
                acc[f"rows@{key}"].append(float(r.numel()))
    print(f"{len(files)} dumps from {a.dirs}, {nq} honest queries, kept "
          f"{a.kept_groups} groups, fence {a.fence_groups} groups")
    print("\n== the (cost, quality) curve per kappa: fired groups vs attention mass\n")
    print(f"{'kappa':>6s} " + "".join(f"{'q'+str(b):>18s}" for b in bqs))
    for k in kappas:
        cells = ""
        for b in bqs:
            key = f"{k}|{b}"
            n_ = len(acc[f"mass@{key}"])
            cells += f"{sum(acc[f'fire@{key}'])/n_:7.1f}/{sum(acc[f'mass@{key}'])/n_:.4f}"
            cells = cells + " " * (18 * (bqs.index(b) + 1) + 6 - len(cells) - 0)
        print(f"{k:6d} {cells}")
    print("\n   each cell is  fired_groups / attention_mass\n")
    print("== cost-matched read: interpolate each kappa to a common fired count")
    import bisect
    for target in (96.0, 110.0, 130.0):
        print(f"\n   at {target:.0f} fired groups:")
        for k in kappas:
            xs = [(sum(acc[f'fire@{k}|{b}'])/len(acc[f'fire@{k}|{b}']),
                   sum(acc[f'mass@{k}|{b}'])/len(acc[f'mass@{k}|{b}']),
                   sum(acc[f'fenced@{k}|{b}'])/len(acc[f'fenced@{k}|{b}'])) for b in bqs]
            xs.sort()
            if not (xs[0][0] <= target <= xs[-1][0]):
                print(f"      kappa {k:2d}: outside the swept range "
                      f"[{xs[0][0]:.0f}, {xs[-1][0]:.0f}]")
                continue
            i = bisect.bisect_left([x[0] for x in xs], target)
            lo, hi = xs[max(i-1,0)], xs[min(i,len(xs)-1)]
            t = 0.0 if hi[0]==lo[0] else (target-lo[0])/(hi[0]-lo[0])
            print(f"      kappa {k:2d}: mass {lo[1]+t*(hi[1]-lo[1]):.4f}   "
                  f"fenced {lo[2]+t*(hi[2]-lo[2]):.3f}")


if __name__ == "__main__":
    main()
