"""Which (kappa, m) a CONSTANT kept set should use, measured at the served size.

defaults.py picked kappa at m = rho*T groups with rho = 1/32, and the kappa sweep in
dsa_selection_stats.py ranks a frequency-aggregated set at m = the whole 512-group
budget. Constant kept moved the used size to 32 groups, and a selector has to be
judged at the size it is used at -- so kappa and m sweep together here.

Two criteria, because the indexer score cannot carry a mass argument. DSA's score

    s_g = sum_h w_h * relu(<q_h, k_g>)

takes its w from a plain linear projection with no activation, so 53% of the gates
and 99.6% of the scores are NEGATIVE: "share of in-budget score mass" is ill-posed
on it (the budget sum is negative, a subset can exceed it, random comes out below
zero). What is well defined on a signed score is the SET: precision against DSA's
own top-512. For a mass criterion the dumps carry the real thing -- qcal and the
latent rows -- so attention mass is computed from the actual softmax:

    a_r = mean_h softmax_r(qcal_h @ rows^T / sqrt(d))_r      (non-negative, sums to 1)

and a candidate set of m groups is scored by the mass of the 4m rows it covers.
Random is the closed form (4m/T for mass, 512/G for precision), never a draw.
"""

import argparse
import glob
import sys

import torch

sys.path.insert(0, "mexp/glm53")
from dsa_selection_stats import score, sigma_at


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", default="cd_w3,cd_w2,cd_g2")
    ap.add_argument("--kappas", default="2,4,8,16,32")
    ap.add_argument("--ms", default="32,64,128,256", help="candidate set size in GROUPS")
    ap.add_argument("--topk-rows", type=int, default=2048)
    ap.add_argument("--block", type=int, default=256, help="sigma window in groups (close_block/pool)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-q", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(8)
    kappas = [int(x) for x in a.kappas.split(",")]
    ms = [int(x) for x in a.ms.split(",")]

    files = []
    for d in a.dirs.split(","):
        files += sorted(glob.glob(f"results/glm53/{d}/*.pt"))
    if a.limit:
        files = files[: a.limit]
    if not files:
        sys.exit("no dumps found")

    acc = {}
    nq_tot, nd = 0, 0
    seqs, layers = set(), set()
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        gk = d["index_k"]["deq"].float()
        pool = int(d["index_k"]["pool_size"])
        G = gk.shape[0]
        topg = max(1, a.topk_rows // pool)
        if G <= topg:
            continue
        rows = d["rows"].float()
        T = rows.shape[0]
        qc = d["qcal"].float()  # [nq, H, kv_lora_rank]
        qs, gates = d["qidx"], d["qgate"]
        if not isinstance(qs, (list, tuple)):
            qs, gates = [qs], [gates]
        n = min(a.max_q, len(qs), qc.shape[0])
        if n == 0:
            continue
        nd += 1
        seqs.add(T)
        layers.add(int(d["lid"]))

        # EVERY selector ranks the CLOSED prefix only. sigma exists for closed
        # blocks alone and the un-closed tail is attended unconditionally, so a
        # baseline allowed to reach into the tail is not ranking salience at all:
        # it is picking the most recent rows, which carry attention mass by
        # recency. That artifact read as a 1.3x win for a centred row norm.
        nb = G // a.block
        if nb == 0:
            continue
        C = nb * a.block
        sig = {f"sigma k={k}": sigma_at(gk, a.block, k)[:C] for k in kappas}
        # kappa=1 keeps DC alone, so sigma k=1 IS the block-centred row norm --
        # the limit where the projection does nothing a mean and a norm could not.
        b = gk[:C].reshape(nb, a.block, -1)
        sig["centred row norm"] = (b - b.mean(1, keepdim=True)).reshape(C, -1).norm(dim=-1)
        sig["raw row norm"] = gk[:C].norm(dim=-1)
        sets = {nm: {m: v.topk(m).indices for m in ms} for nm, v in sig.items()}
        scale = rows.shape[1] ** -0.5

        for i in range(n):
            s = score(qs[i], gates[i], gk)
            dsa = s.topk(topg).indices
            inb = torch.zeros(G, dtype=torch.bool)
            inb[dsa] = True
            # real attention mass per row, averaged over heads
            att = torch.softmax(qc[i] @ rows.T * scale, dim=-1).mean(0)  # [T]
            # group mass: rows covered by each group, pool-aligned, clipped to T
            gm = torch.zeros(G)
            lim = min(G * pool, T)
            gm[: lim // pool] = att[: (lim // pool) * pool].reshape(-1, pool).sum(1)
            nq_tot += 1
            for m in ms:
                cands = [(nm, c[m]) for nm, c in sets.items()]
                cands.append(("oracle (att mass)", gm[:C].topk(m).indices))
                cands.append(("oracle (dsa score)", s[:C].topk(m).indices))
                for nm, idx in cands:
                    e = acc.setdefault((nm, m), [0.0, 0.0, 0])
                    e[0] += float(gm[idx].sum())
                    e[1] += float(inb[idx].float().mean())
                    e[2] += 1
                e = acc.setdefault(("random (closed form)", m), [0.0, 0.0, 0])
                e[0] += float(gm[:C].sum()) * m / C
                e[1] += float(inb[:C].float().mean())
                e[2] += 1

    print(f"{nd} dumps from {a.dirs}, {nq_tot} queries, layers {sorted(layers)}, "
          f"seq {sorted(seqs)}, DSA budget {a.topk_rows} rows = {a.topk_rows//4} groups")
    names = [f"sigma k={k}" for k in kappas] + [
        "centred row norm", "raw row norm",
        "oracle (att mass)", "oracle (dsa score)", "random (closed form)"]
    for metric, col in (("captured attention mass (real softmax, mean over heads)", 0),
                        ("precision vs DSA top-512", 1)):
        print(f"\n== {metric}")
        print(f"{'selector':>22s} " + "".join(f"{'m='+str(m):>10s}" for m in ms))
        for nm in names:
            row = "".join(f"{acc[(nm,m)][col]/acc[(nm,m)][2]:10.4f}" if (nm, m) in acc
                          else f"{'-':>10s}" for m in ms)
            print(f"{nm:>22s} {row}")


if __name__ == "__main__":
    main()
