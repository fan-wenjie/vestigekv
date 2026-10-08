"""How large is the UNION over heads of what each head actually needs?

The mass is concentrated per head (n_eff ~ 3-30 of 2048) but DSA's selection is one
shared set for the layer, so the budget is set by the union of 32 peaked-but-different
distributions. MLA shares the latent row across heads, so reading a row serves every
head: the COST is the union, not the sum. Therefore the quantity that decides whether
2048 is tight or slack is

    |union_h  top-k_h|   where top-k_h is the smallest set giving head h mass >= q

If that union is far below 2048, an oracle selector could serve the same quality on
fewer rows and there is headroom for a better ranking. If it is ~2048, DSA's budget is
already tight and any approximation must pay mass.

Also: the mass each head gets from DSA's 2048 relative to FULL attention over the whole
archive -- DSA is itself an approximation and the comparison needs its reference.
"""

import glob, sys, collections
import torch

sys.path.insert(0, "mexp/glm53")
from dsa_selection_stats import score


def main():
    torch.set_num_threads(8)
    TOPR, qs_ = 2048, (0.90, 0.99, 0.999)
    agg = collections.defaultdict(list)
    nd = 0
    for dn in ("cd_w3", "cd_w2", "cd_g2"):
        for fp in sorted(glob.glob(f"results/glm53/{dn}/*.pt")):
            d = torch.load(fp, map_location="cpu", weights_only=False)
            gk = d["index_k"]["deq"].float()
            pool = int(d["index_k"]["pool_size"])
            G = gk.shape[0]
            topg = max(1, TOPR // pool)
            if G <= topg:
                continue
            rows = d["rows"].float()
            Tn = rows.shape[0]
            qc = d["qcal"].float()[-1]
            sc = float(d["scale"])
            nd += 1
            # FULL attention over the whole archive, per head
            full = torch.softmax(qc @ rows.T * sc, dim=-1)        # [H, Tn]
            # DSA's selection
            s = score(d["qidx"][-1], d["qgate"][-1], gk)
            gsel = s.topk(topg).indices
            r = (gsel[:, None] * pool + torch.arange(pool)[None, :]).reshape(-1)
            r = r[r < Tn]
            agg["dsa mass of full"].append(float(full[:, r].sum(-1).mean()))
            agg["dsa mass worst head"].append(float(full[:, r].sum(-1).min()))
            # per-head minimal set for mass q, measured on the FULL distribution,
            # then the union over heads
            ps, idx = full.sort(dim=-1, descending=True)
            cum = ps.cumsum(-1)
            for q in qs_:
                kh = (cum < q).sum(-1) + 1                        # [H] rows per head
                u = set()
                for h in range(full.shape[0]):
                    u |= set(idx[h, : int(kh[h])].tolist())
                agg[f"per-head k @{q}"].append(float(kh.float().mean()))
                agg[f"union @{q}"].append(float(len(u)))
                agg[f"union/2048 @{q}"].append(len(u) / TOPR)
    print(f"{nd} dumps, one honest query each\n")
    print(f"  mass DSA's 2048 rows capture of FULL attention:  "
          f"mean head {sum(agg['dsa mass of full'])/nd:.4f}   "
          f"worst head {sum(agg['dsa mass worst head'])/nd:.4f}")
    print(f"\n{'target q':>9s} {'rows/head':>11s} {'UNION rows':>12s} {'union/2048':>12s}")
    for q in qs_:
        print(f"{q:9.3f} {sum(agg[f'per-head k @{q}'])/nd:11.1f} "
              f"{sum(agg[f'union @{q}'])/nd:12.1f} {sum(agg[f'union/2048 @{q}'])/nd:12.3f}")
    print("\n  union/2048 < 1 means an oracle could serve that quality on fewer rows")
    print("  than DSA reads; > 1 means DSA's budget is already short of it.")


if __name__ == "__main__":
    main()
