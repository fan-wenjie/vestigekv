"""Can the hot set's own score quantile stand in for DSA's 2048nd-best score?

The top-k kernel's input is the already-computed logits array, so no pre-filter can
cost less than the one pass a radix select already makes. What CAN be removed is the
PASSES: a radix select needs several because it does not know the cut. The branch rule
already computes a quantile of the kept groups' scores for free, so if that quantile
predicts the true 2048nd-best score closely, the select collapses to one pass --
count above the estimate, and correct only if the count is off.

What matters is not the threshold's relative error but the COUNT it admits: a
threshold slightly too low admits far too many groups on a heavy-tailed score.
"""

import argparse, glob, sys
import torch

sys.path.insert(0, "mexp/glm53")
from dsa_selection_stats import score, sigma_at


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", default="cd_w3,cd_w2,cd_g2")
    ap.add_argument("--hot-groups", type=int, default=32)
    ap.add_argument("--topk-rows", type=int, default=2048)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--kappa", type=int, default=16)
    ap.add_argument("--max-q", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(8)
    files = []
    for d in a.dirs.split(","):
        files += sorted(glob.glob(f"results/glm53/{d}/*.pt"))
    ests = ("hot q0.50", "hot q0.75", "hot q0.90", "hot max", "hot min")
    acc = {e: [] for e in ests}
    nq = 0
    for fp in files:
        d = torch.load(fp, map_location="cpu", weights_only=False)
        gk = d["index_k"]["deq"].float()
        pool = int(d["index_k"]["pool_size"])
        G = gk.shape[0]
        topg = max(1, a.topk_rows // pool)
        if G <= topg:
            continue
        nb = G // a.block
        if nb == 0:
            continue
        C = nb * a.block
        hot = sigma_at(gk, a.block, a.kappa)[:C].topk(min(a.hot_groups, C)).indices
        qs, gates = d["qidx"], d["qgate"]
        if not isinstance(qs, (list, tuple)):
            qs, gates = [qs], [gates]
        for q, g in list(zip(qs, gates))[: a.max_q]:
            s = score(q, g, gk)
            tau = float(s.topk(topg).values.min())
            hs = s[hot]
            cand = {"hot q0.50": float(hs.quantile(0.50)),
                    "hot q0.75": float(hs.quantile(0.75)),
                    "hot q0.90": float(hs.quantile(0.90)),
                    "hot max": float(hs.max()),
                    "hot min": float(hs.min())}
            nq += 1
            for k, v in cand.items():
                n_above = int((s > v).sum())
                acc[k].append((v - tau, n_above / topg))
    print(f"{nq} queries, DSA cut = {a.topk_rows//4} groups of {len(ests)} estimators, "
          f"hot set = {a.hot_groups} groups by sigma kappa={a.kappa}\n")
    print(f"{'estimator':>12s} {'median (est-tau)':>17s} {'median count/512':>17s} "
          f"{'p10 count':>10s} {'p90 count':>10s} {'frac within 2x':>15s}")
    for k in ests:
        d_ = torch.tensor([x[0] for x in acc[k]])
        r = torch.tensor([x[1] for x in acc[k]])
        print(f"{k:>12s} {float(d_.median()):17.3f} {float(r.median()):17.3f} "
              f"{float(r.quantile(0.1)):10.3f} {float(r.quantile(0.9)):10.3f} "
              f"{float(((r>0.5)&(r<2.0)).float().mean()):15.3f}")
    print("\n  count/512 = how many groups clear the estimate, relative to the 512 wanted.")
    print("  A one-pass select needs this near 1 and tightly spread; >>1 means the pass")
    print("  compacts far too many and a second pass is needed anyway.")


if __name__ == "__main__":
    main()
