"""On a fenced lane: TRUNCATE our own fired set, or fall back to DSA's 2048?

Current implementation: `fenced = ng > WF` zeroes flen and the lane attends DSA's
index_topk selection -- 2048 rows. Proposal: keep our own fired groups but truncate to
the WF highest-scoring, which is 223*4 = 892 rows and needs NO DSA top-k at all.

That second property is the prize. The fenced lane is the only consumer of DSA's top-k
output on this arm, so if truncation replaces it the 85.9 us (36k) / 157.4 us (124k)
top-k kernel has no consumer and can be skipped -- while the logits kernel stays,
because the branch rule thresholds its scores.

Compared only on the steps where the two differ, i.e. the fenced ones, by the real
softmax mass over the rows each design attends.
"""

import argparse, glob, sys, collections
import torch

sys.path.insert(0, "mexp/glm53")
from dsa_selection_stats import score, sigma_at


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", default="cd3_64k")
    ap.add_argument("--kappa", type=int, default=4)
    ap.add_argument("--kept-groups", type=int, default=32)
    ap.add_argument("--branch-q", type=float, default=0.75)
    ap.add_argument("--fences", default="223,320,512")
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--max-q", type=int, default=4)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--sinks", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(8)
    fences = [int(x) for x in a.fences.split(",")]
    files = sorted(sum([glob.glob(f"results/glm53/{d}/*.pt") for d in a.dirs.split(",")], []))[: a.limit]
    acc = collections.defaultdict(list)
    nq = nf = 0
    for fp in files:
        d = torch.load(fp, map_location="cpu", weights_only=False)
        gk = d["index_k"]["deq"].float()
        pool = int(d["index_k"]["pool_size"])
        G = gk.shape[0]
        topg = 2048 // pool
        if G <= topg:
            continue
        rows = d["rows"].float(); Tn = rows.shape[0]; sc = float(d["scale"])
        nb = G // a.block
        if nb == 0:
            continue
        C = nb * a.block
        sig = sigma_at(gk, a.block, a.kappa)[:C]
        kept = sig.topk(min(a.kept_groups, C)).indices
        tail = torch.arange(C * pool, Tn)
        qs, gates, qc = d["qidx"], d["qgate"], d["qcal"].float()
        if not isinstance(qs, (list, tuple)):
            continue
        for i in range(min(a.max_q, len(qs), qc.shape[0])):
            s = score(qs[i], gates[i], gk)
            tau = float(s[kept].quantile(a.branch_q))
            cand = (s[:C] > tau); cand[kept] = False
            ng = int(cand.sum())
            nq += 1
            full = None
            for WF in fences:
                if ng <= WF:
                    acc[f"rate@{WF}"].append(0.0)
                    continue
                acc[f"rate@{WF}"].append(1.0)
                if full is None:
                    full = torch.softmax(qc[i] @ rows.T * sc, dim=-1)
                    nf += 1

                def mass(groups):
                    r = (groups[:, None] * pool + torch.arange(pool)[None, :]).reshape(-1)
                    r = r[r < Tn]
                    r = torch.cat([torch.arange(min(a.sinks, Tn)), r, tail]).unique()
                    m = full[:, r].sum(-1)
                    return float(m.mean()), float(m.min()), r.numel()

                trunc = torch.cat([kept, s[:C].topk(WF).indices])
                fb = s.topk(topg).indices                  # DSA's own 2048 rows
                for nm, g in (("trunc", trunc), ("fallback", fb)):
                    mm, mw, nr = mass(g)
                    acc[f"{nm}_mass@{WF}"].append(mm)
                    acc[f"{nm}_worst@{WF}"].append(mw)
                    acc[f"{nm}_rows@{WF}"].append(float(nr))
    print(f"{len(files)} dumps, {nq} honest queries, kappa={a.kappa}, kept "
          f"{a.kept_groups} groups, branch q{a.branch_q}\n")
    print(f"{'WF':>5s} {'fence rate':>11s} {'design':>10s} {'mass':>8s} {'worst head':>11s} "
          f"{'rows read':>10s}")
    for WF in fences:
        rate = sum(acc[f"rate@{WF}"]) / max(len(acc[f"rate@{WF}"]), 1)
        n_ = len(acc[f"trunc_mass@{WF}"])
        if n_ == 0:
            print(f"{WF:5d} {rate:11.3f}   (never fenced)"); continue
        for nm in ("trunc", "fallback"):
            print(f"{WF if nm=='trunc' else '':>5} {rate if nm=='trunc' else '':>11} "
                  f"{nm:>10s} {sum(acc[f'{nm}_mass@{WF}'])/n_:8.4f} "
                  f"{sum(acc[f'{nm}_worst@{WF}'])/n_:11.4f} "
                  f"{sum(acc[f'{nm}_rows@{WF}'])/n_:10.1f}")
    print("\n   compared ONLY on fenced steps, where the two designs differ")


if __name__ == "__main__":
    main()
