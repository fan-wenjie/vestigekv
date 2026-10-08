"""Does DSA's selection transfer ACROSS LAYERS, for the same query?

The one untested member of the faster-approximate-top-k family. DSA runs its top-k
once per full-attention layer -- 11 times a step, 87.1 us at 36k and 157.4 at 124k --
and if layer L's top-512 predicted layer L' s, the selection could be computed on a
subset of layers and shared. Unlike making the kernel faster, that is a RELATIVE win:
DSA computes all eleven.

Known already: sigma does not transfer across layers. That is a different statistic --
sigma is query-free and derived from the keys, while this asks about the query-dependent
selection itself.

Matched by request AND by decode step: the dumps key on (slot, seq_len) per layer and
hold 8 queries at consecutive qpos, so query i of layer A and query i of layer B are
the same step only when their qpos agree, which is asserted rather than assumed.
"""

import glob, sys, collections, itertools
import torch

sys.path.insert(0, "mexp/glm53")
from dsa_selection_stats import score


def main():
    torch.set_num_threads(8)
    topg = 512
    # group dumps by (slot, seq_len) so layers of the SAME request line up
    by_req = collections.defaultdict(dict)
    for dn in ("cd3_64k", "cd3_16k"):
        for fp in sorted(glob.glob(f"results/glm53/{dn}/*.pt")):
            d = torch.load(fp, map_location="cpu", weights_only=False)
            gk = d["index_k"]["deq"].float()
            if gk.shape[0] <= topg:
                continue
            qs, gates, qp = d["qidx"], d["qgate"], d.get("qpos")
            if not isinstance(qs, (list, tuple)) or qp is None:
                continue
            tops = []
            for q, g in zip(qs, gates):
                tops.append(set(score(q, g, gk).topk(topg).indices.tolist()))
            by_req[(dn, int(d["slot"]), int(d["seq_len"]))][int(d["lid"])] = (
                tops, [int(x) for x in qp.flatten()], gk.shape[0])
    pair = collections.defaultdict(list)
    same_layer_ctrl = []
    nreq = 0
    for key, layers in by_req.items():
        lids = sorted(layers)
        if len(lids) < 2:
            continue
        nreq += 1
        for a, b in itertools.combinations(lids, 2):
            ta, qa, Ga = layers[a]
            tb, qb, Gb = layers[b]
            common = set(qa) & set(qb)
            for p in common:                       # same decode step, asserted
                ia, ib = qa.index(p), qb.index(p)
                r = len(ta[ia] & tb[ib]) / topg
                pair[(b - a)].append(r)
        # control: within ONE layer, two DIFFERENT steps (the persistence number)
        ta, qa, _ = layers[lids[0]]
        if len(ta) > 1:
            same_layer_ctrl.append(len(ta[0] & ta[1]) / topg)
    print(f"{nreq} requests with >=2 layers dumped, {sum(len(v) for v in pair.values())} "
          f"layer pairs at a matched decode step\n")
    print(f"{'layer gap':>10s} {'overlap of top-512':>20s} {'n':>7s}")
    for gap in sorted(pair):
        v = pair[gap]
        if len(v) < 20:
            continue
        print(f"{gap:10d} {sum(v)/len(v):20.4f} {len(v):7d}")
    allv = [x for v in pair.values() for x in v]
    print(f"\n   all gaps pooled           {sum(allv)/len(allv):.4f}")
    print(f"   SAME layer, next step     {sum(same_layer_ctrl)/len(same_layer_ctrl):.4f}"
          f"   <- the persistence control")
    print(f"   random expectation        {512/15350:.4f}   (G ~ 15350 at 61k)")
    print("\n   For sharing to pay, cross-layer overlap must beat the same-layer")
    print("   next-step number: that is the alternative use of one selection.")


if __name__ == "__main__":
    main()
