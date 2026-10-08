"""Is an approximate top-2048 feasible? The criterion is dropped softmax MASS.

Exact identity, no assumptions: for S subset of T with dropped mass
m = sum_{t in T\\S} p_t, where p = softmax over T,

    o_S - o_T = m/(1-m) * (vbar_S - vbar_D)

so the output error is linear in m and the set size never appears. Feasibility is
therefore a question about the concentration of p over DSA's 2048 rows:

    n_eff = 1 / sum_t p_t^2      (effective support)

n_eff << 2048 -> a low-recall approximation is sound; n_eff ~ 2048 (uniform) -> any
dropped row costs 1/2048 of the mass and recall must be ~0.99.

Per-head, and reported at the WORST head: DSA's top-2048 is shared across the layer's
32 heads but each head has its own softmax, so a dropped row hurts whichever head
wanted it, and the layer's output error is bounded by the max over heads.

qidx/qgate are aliased in these dumps (every entry holds the last step's write), so
the only correct (qcal, qidx) pair is index -1 -- one honest query per dump, 88 total.
"""

import glob, sys, collections
import torch

sys.path.insert(0, "mexp/glm53")
from dsa_selection_stats import score


def main():
    torch.set_num_threads(8)
    TOPR = 2048
    Ms = (64, 128, 256, 512, 1024)
    agg = collections.defaultdict(list)
    cover = collections.defaultdict(list)
    per_layer = collections.defaultdict(list)
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
            qc = d["qcal"].float()[-1]            # [H, 512] -- last step, matches qidx[-1]
            qi, qg = d["qidx"][-1], d["qgate"][-1]
            sc = float(d["scale"])
            s = score(qi, qg, gk)
            gsel = s.topk(topg).indices                     # groups DSA keeps
            # expand to rows, clipped to the sequence
            r = (gsel[:, None] * pool + torch.arange(pool)[None, :]).reshape(-1)
            r = r[r < Tn]
            sord = s[gsel].argsort(descending=True)         # row order by indexer score
            rsorted = (gsel[sord][:, None] * pool + torch.arange(pool)[None, :]).reshape(-1)
            rsorted = rsorted[rsorted < Tn]
            nd += 1
            K = rows[r]                                     # [|T|, 512]
            logit = qc @ K.T * sc                           # [H, |T|]
            p = torch.softmax(logit, dim=-1)
            neff = 1.0 / (p * p).sum(-1)                    # [H]
            agg["n_eff mean"].append(float(neff.mean()))
            agg["n_eff worst(min)"].append(float(neff.min()))
            agg["|T|"].append(float(r.numel()))
            ps, _ = p.sort(dim=-1, descending=True)
            cum = ps.cumsum(-1)
            for q in (0.50, 0.90, 0.99, 0.999):
                k = (cum < q).sum(-1) + 1
                cover[q].append((float(k.float().mean()), float(k.max())))
            # mass dropped if only the top-M rows BY INDEXER SCORE are kept
            pos = {int(x): i for i, x in enumerate(r.tolist())}
            order = torch.tensor([pos[int(x)] for x in rsorted.tolist() if int(x) in pos])
            v = rows[r]                                     # value proxy = latent row
            oT = p @ v
            for M in Ms:
                keep = order[:M]
                mass = p[:, keep].sum(-1)                   # kept mass
                mdrop = 1.0 - mass
                oS = (p[:, keep] @ v[keep]) / mass[:, None]
                relerr = (oS - oT).norm(dim=-1) / oT.norm(dim=-1)
                agg[f"m@{M}"].append((float(mdrop.mean()), float(mdrop.max())))
                agg[f"err@{M}"].append((float(relerr.mean()), float(relerr.max())))
                if M == 256:
                    per_layer[int(d["lid"])].append(float(mdrop.max()))

    print(f"{nd} dumps, one honest query each, DSA budget {TOPR} rows\n")
    print(f"  |T| actually kept           {sum(agg['|T|'])/nd:8.1f} rows")
    print(f"  effective support n_eff     mean over heads {sum(agg['n_eff mean'])/nd:8.1f}"
          f"   worst head {sum(agg['n_eff worst(min)'])/nd:8.1f}   of {TOPR}")
    print("\n== rows needed to reach a mass quantile (within DSA's 2048)")
    print(f"{'quantile':>10s} {'mean head':>11s} {'worst head':>11s}")
    for q in (0.50, 0.90, 0.99, 0.999):
        a = sum(x[0] for x in cover[q]) / nd
        b = sum(x[1] for x in cover[q]) / nd
        print(f"{q:10.3f} {a:11.1f} {b:11.1f}")
    print("\n== keep only the top-M rows BY INDEXER SCORE: dropped mass and output error")
    print(f"{'M':>6s} {'recall':>8s} {'m mean':>9s} {'m worst':>9s} "
          f"{'relerr mean':>12s} {'relerr worst':>13s}")
    for M in Ms:
        mm = agg[f"m@{M}"]; ee = agg[f"err@{M}"]
        print(f"{M:6d} {M/2048:8.3f} {sum(x[0] for x in mm)/nd:9.4f} "
              f"{sum(x[1] for x in mm)/nd:9.4f} {sum(x[0] for x in ee)/nd:12.4f} "
              f"{sum(x[1] for x in ee)/nd:13.4f}")
    print("\n== worst-head dropped mass at M=256, per layer")
    print(f"{'lid':>4s} {'m worst':>9s}")
    for lid in sorted(per_layer):
        print(f"{lid:4d} {sum(per_layer[lid])/len(per_layer[lid]):9.4f}")


if __name__ == "__main__":
    main()
