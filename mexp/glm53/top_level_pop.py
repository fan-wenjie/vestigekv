"""How often does the OPEN-ENDED top coarse level alone overpopulate the buffer?

The cap drops whole levels, which is score-ordered. But level NC-1 is open above:
everything beyond lo + 2*(hi-lo) clamps into it, where lo/hi span the KEPT groups'
logits. So if that level alone holds more than WF fires, no cap can bound it and the
compaction's `pos < WG` clamp takes over -- which keeps the LOWEST group ids, i.e. the
earliest positions, discarding the newest. Position-order truncation is the rule
measured to retain 18.7% of what beats max1 against 80.3% by score, and the top level
is the one place where "same level" does NOT imply "similar score".

So: per query, how many fires land in the top level, and how often is that > WF.
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
    ap.add_argument("--wf", type=int, default=512)
    ap.add_argument("--nc", type=int, default=16)
    ap.add_argument("--mult", type=float, default=2.0)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--max-q", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(8)
    files = sorted(sum([glob.glob(f"results/glm53/{d}/*.pt") for d in a.dirs.split(",")], []))
    if a.limit:
        files = files[: a.limit]
    acc = collections.defaultdict(list)
    per_layer = collections.defaultdict(list)
    for fp in files:
        d = torch.load(fp, map_location="cpu", weights_only=False)
        gk = d["index_k"]["deq"].float()
        pool = int(d["index_k"]["pool_size"])
        G = gk.shape[0]
        if G <= 2048 // pool:
            continue
        nb = G // a.block
        if nb == 0:
            continue
        C = nb * a.block
        kept = sigma_at(gk, a.block, a.kappa)[:C].topk(min(a.kept_groups, C)).indices
        qs, gates = d["qidx"], d["qgate"]
        if not isinstance(qs, (list, tuple)):
            continue
        for i in range(min(a.max_q, len(qs))):
            s = score(qs[i], gates[i], gk)
            ks = s[kept]
            lo, hi = float(ks.min()), float(ks.max())
            width = max(hi - lo, 1e-6)
            thr = float(ks.quantile(a.branch_q))
            cand = (s[:C] > thr)
            cand[kept] = False
            fired = cand.nonzero().flatten()
            if fired.numel() == 0:
                continue
            lvl = (((s[fired] - lo) / (a.mult * width) * a.nc).to(torch.int32)
                   ).clamp(0, a.nc - 1)
            top = int((lvl == a.nc - 1).sum())
            acc["nfire"].append(float(fired.numel()))
            acc["top"].append(float(top))
            acc["top_frac"].append(top / fired.numel())
            acc["overflows"].append(1.0 if fired.numel() > a.wf else 0.0)
            acc["degenerate"].append(1.0 if top > a.wf else 0.0)
            per_layer[int(d["lid"])].append(1.0 if top > a.wf else 0.0)
            # if it IS degenerate, how much score spread is inside the top level?
            if top > a.wf:
                ts = s[fired][lvl == a.nc - 1]
                acc["top_spread_widths"].append(
                    float((ts.max() - ts.min()) / width))
    n = len(acc["nfire"])
    print(f"{len(files)} dumps, {n} honest queries, kappa={a.kappa}, kept {a.kept_groups}, "
          f"q{a.branch_q}, WF={a.wf}, NC={a.nc}\n")
    print(f"   fires per query, mean           {sum(acc['nfire'])/n:8.1f}")
    print(f"   of which in the TOP level       {sum(acc['top'])/n:8.1f}"
          f"   ({100*sum(acc['top_frac'])/n:.1f}% of fires)")
    print(f"   steps that overflow WF          {100*sum(acc['overflows'])/n:8.1f}%")
    print(f"   steps where the TOP LEVEL alone")
    print(f"   overflows WF  (-> position order) {100*sum(acc['degenerate'])/n:6.2f}%")
    if acc["top_spread_widths"]:
        v = sorted(acc["top_spread_widths"])
        print(f"\n   on those steps, the score spread INSIDE the top level, in units of")
        print(f"   the kept range: median {v[len(v)//2]:.2f}, max {v[-1]:.2f}")
    print(f"\n   per-layer share of degenerate steps")
    for lid in sorted(per_layer):
        v = per_layer[lid]
        print(f"      lid {lid:2d}  {100*sum(v)/len(v):6.2f}%")


if __name__ == "__main__":
    main()
