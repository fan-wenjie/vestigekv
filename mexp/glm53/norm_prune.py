"""Can a norm bound skip index-k rows without reading them?

DSA's indexer score is s_g = sum_h w_h * relu(<q_h, k_g>), and Cauchy-Schwarz
gives a bound that needs only the key's NORM, not the key:

    s_g <= sum_h max(w_h, 0) * ||q_h|| * ||k_g||  =  C(q) * ||k_g||

C(q) is one scalar per query (the negative-weight terms only subtract, so dropping
them is sound). So if C(q)*||k_g|| < tau, group g cannot enter the top-2048 and its
132 bytes never have to be read. Norms are 4 bytes a group against 132, so the
screen itself is 3% of the scan.

Three thresholds, because which tau is available decides everything:

  tau_oracle  the true 2048th-best score. Not available online -- it is what you
              are trying to find -- so it is the CEILING on what any norm-bound
              scheme could ever prune. If this prunes nothing, the design closes.
  tau_hot     q0.75 of the hot set's group scores, which is the threshold the
              served tier 2 actually uses. Available for free.
  tau_hotmax  the hot set's best score, the most aggressive sound-ish variant.

Also reported: the bound's LOOSENESS, median C*||k||/s over groups that are in the
budget. That number is why the answer comes out the way it does -- relu discards
about half the dot products and the surviving cosines are small, so the bound sits
far above the score it is bounding.
"""

import argparse
import glob
import sys

import torch

sys.path.insert(0, "mexp/glm53")
from dsa_selection_stats import score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", default="cd_w3,cd_w2,cd_g2")
    ap.add_argument("--topk-rows", type=int, default=2048)
    ap.add_argument("--hot-groups", type=int, default=32)
    ap.add_argument("--block", type=int, default=256, help="sigma window in groups")
    ap.add_argument("--max-q", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(8)
    from dsa_selection_stats import sigma_at

    files = []
    for d in a.dirs.split(","):
        files += sorted(glob.glob(f"results/glm53/{d}/*.pt"))
    if a.limit:
        files = files[: a.limit]
    if not files:
        sys.exit("no dumps found")

    acc = {k: 0.0 for k in ("pr_or", "pr_hot", "pr_hotmax", "loose", "lost_or",
                            "lost_hot", "lost_hotmax", "normprec", "pagesave")}
    nq = 0
    Gs = set()
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        gk = d["index_k"]["deq"].float()
        pool = int(d["index_k"]["pool_size"])
        G = gk.shape[0]
        topg = max(1, a.topk_rows // pool)
        if G <= topg:
            continue
        Gs.add(G)
        nrm = gk.norm(dim=-1)
        nb = G // a.block
        C = nb * a.block
        sig = sigma_at(gk, a.block, 1)[:C]
        hot = sig.topk(min(a.hot_groups, C)).indices
        qs, gates = d["qidx"], d["qgate"]
        if not isinstance(qs, (list, tuple)):
            qs, gates = [qs], [gates]
        for q, g in list(zip(qs, gates))[: a.max_q]:
            s = score(q, g, gk)
            top = s.topk(topg).indices
            tau_or = float(s[top].min())
            cq = float((g.float().clamp(min=0) * q.float().norm(dim=-1)).sum())
            bound = cq * nrm
            hs = s[hot]
            tau_hot = float(hs.quantile(0.75))
            tau_hotmax = float(hs.max())
            inb = torch.zeros(G, dtype=torch.bool)
            inb[top] = True
            nq += 1
            for name, tau in (("or", tau_or), ("hot", tau_hot), ("hotmax", tau_hotmax)):
                skip = bound < tau
                acc[f"pr_{name}"] += float(skip.float().mean())
                # a skipped group that WAS in the budget is a recall loss; a sound
                # bound can only lose one if tau is above the true cut
                acc[f"lost_{name}"] += float((skip & inb).sum()) / topg
            acc["loose"] += float((bound[top] / s[top].clamp(min=1e-6)).median())
            acc["normprec"] += float(inb[nrm.topk(topg).indices].float().mean())
            # pages the skip actually frees: DSA's index pool is 64 slots a page,
            # so a scattered skip frees nothing
            pg = (G + 63) // 64
            sk = (bound < tau_or).float()
            per_page = sk[: pg * 64 - (pg * 64 - G) if pg * 64 >= G else G]
            padded = torch.zeros(pg * 64)
            padded[:G] = sk
            acc["pagesave"] += float((padded.reshape(pg, 64).min(1).values).mean())

    n = max(nq, 1)
    print(f"{len(files)} dumps, {nq} queries, G {sorted(Gs)}, budget {a.topk_rows} rows "
          f"= {a.topk_rows//pool} groups, hot set {a.hot_groups} groups\n")
    print("  sound norm bound C(q)*||k|| vs threshold")
    print(f"    tau = true 2048th best (CEILING, not available online)")
    print(f"      groups skippable          {acc['pr_or']/n:7.4f}")
    print(f"      budget groups lost        {acc['lost_or']/n:7.4f}")
    print(f"      whole 64-slot pages freed {acc['pagesave']/n:7.4f}")
    print(f"    tau = q0.75 of hot set (what tier 2 serves)")
    print(f"      groups skippable          {acc['pr_hot']/n:7.4f}")
    print(f"      budget groups lost        {acc['lost_hot']/n:7.4f}")
    print(f"    tau = hot set max")
    print(f"      groups skippable          {acc['pr_hotmax']/n:7.4f}")
    print(f"      budget groups lost        {acc['lost_hotmax']/n:7.4f}")
    print(f"\n  bound looseness, median C*||k||/s over in-budget groups "
          f"{acc['loose']/n:8.2f}x")
    print(f"  norm as a SELECTOR: precision of top-512 by norm {acc['normprec']/n:7.4f}"
          f"  (random {topg/max(Gs):.4f})")


if __name__ == "__main__":
    main()
