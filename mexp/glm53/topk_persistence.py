"""How long does DSA's top-2048 stay valid as the decode advances?

Re-run on the RE-COLLECTED dumps (cd3_*), where qidx is distinct per step. The first
run of this used cd_* and read r(k)=1.0000 at every lag -- it was comparing one query
with itself seven times.

The dumps hold 8 queries at CONSECUTIVE positions (qpos span 7) for one layer and
request, which is the wrong data for a static pool but exactly the right data for
persistence: how much of step t+k's top-512 is already inside step t's.

If r(k) decays slowly then the selection can be refreshed every K steps instead of
every step, and the 87.1 us/step top-k becomes 87.1/K. That is a RELATIVE win over
DSA, which re-selects every step -- unlike making the kernel itself faster, which
speeds the baseline up identically.

Also reported: the same curve for a pool of 2x and 4x the budget, since a refreshed
pool can be made larger than 512 to buy more steps of validity, and the union over
the 8 steps, which is the smallest pool that would have covered all of them.
"""

import glob, sys, collections
import torch

sys.path.insert(0, "mexp/glm53")
from dsa_selection_stats import score


def main():
    torch.set_num_threads(8)
    topg = 512
    rec = collections.defaultdict(list)       # (mult, k) -> recalls
    uni = []
    per_layer = collections.defaultdict(list)
    for dn in ("cd3_64k", "cd3_16k"):
        for fp in sorted(glob.glob(f"results/glm53/{dn}/*.pt")):
            d = torch.load(fp, map_location="cpu", weights_only=False)
            gk = d["index_k"]["deq"].float()
            G = gk.shape[0]
            if G <= topg:
                continue
            qs, gates = d["qidx"], d["qgate"]
            if not isinstance(qs, (list, tuple)):
                continue
            if len(qs) < 2:
                continue
            S = [score(q, g, gk) for q, g in zip(qs, gates)]
            tops = [set(s.topk(topg).indices.tolist()) for s in S]
            for mult in (1, 2, 4):
                M = min(mult * topg, G)
                pool0 = set(S[0].topk(M).indices.tolist())
                for k in range(1, len(tops)):
                    r = len(tops[k] & pool0) / topg
                    rec[(mult, k)].append(r)
                    if mult == 1:
                        per_layer[int(d["lid"])].append((k, r))
            u = set()
            for t in tops: u |= t
            uni.append(len(u) / topg)

    print("== recall of step t+k's top-512 inside step t's top-(mult*512)")
    print(f"{'k':>3s} " + "".join(f"{'pool '+str(m)+'x':>11s}" for m in (1, 2, 4)) + f"{'n':>7s}")
    for k in range(1, 8):
        cells = ""
        n = 0
        for m in (1, 2, 4):
            v = rec[(m, k)]
            cells += f"{(sum(v)/len(v) if v else float('nan')):11.4f}"
            n = len(v)
        print(f"{k:3d} {cells}{n:7d}")
    print(f"\n   union of all 8 steps' top-512, as a multiple of 512: "
          f"{sum(uni)/len(uni):.3f}  (1.0 = the 8 steps picked the SAME 512 groups)")
    print("\n== per-layer recall at k=7, pool 1x")
    print(f"{'lid':>4s} {'r(7)':>8s}")
    for lid in sorted(per_layer):
        v = [r for k, r in per_layer[lid] if k == 7]
        if v: print(f"{lid:4d} {sum(v)/len(v):8.4f}")


if __name__ == "__main__":
    main()
