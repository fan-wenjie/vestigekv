"""Mass-vs-bytes curve for the GLM branch rule, offline (see branch_recall.py).

branch_recall.py showed the literal rule -- fire above the best KEPT group --
recalls ~2 groups a step: the recent window sits in the kept set and the
indexer scores it above nearly every archive group. This sweeps the threshold
instead of asserting one, so the design is picked off a curve:

  thr=q<p>      p-quantile of the kept groups' index scores (1.0 = the max)
  thr=nr-max    max over the kept groups OUTSIDE sinks+recent (tier 1's own picks)
  budget        no threshold: top-cap archive groups every step (f = 1)

Every row: oracle mass of kept U fired (and DSA's own top-cap for reference),
mean fired groups, capped fraction, bytes relative to DSA (33 B/token index
read, 1 KB per attended row, DSA attends 2048 rows).
"""
import argparse, glob, sys
import torch
sys.path.insert(0, "/home/user/vestigekv-wt/engine-glm521/python")
from sglang.srt.layers.attention.vestigekv import defaults as D  # noqa: E402
from branch_recall import recall, expand, dsa_score, ROW_B, IDX_B  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", default="cd_g2")
    ap.add_argument("--cap", type=int, default=512)
    a = ap.parse_args()
    torch.set_num_threads(8)
    qs = [1.0, 0.99, 0.95, 0.9, 0.75, 0.5, 0.25]
    labels = [f"q{p}" for p in qs] + ["nr-max", "budget"]
    for dname in a.dirs.split(","):
        files = sorted(glob.glob(f"results/glm53/{dname}/*.pt"))
        mass = {l: 0.0 for l in labels}; fired = {l: 0.0 for l in labels}; capd = {l: 0 for l in labels}
        byt = {l: 0.0 for l in labels}; bdsa = 0.0; mdsa = 0.0; mkept = 0.0; nq = 0; argmax_recent = 0
        for f in files:
            d = torch.load(f, map_location="cpu", weights_only=False)
            ik = d["index_k"]; pool = int(ik["pool_size"]); gk = ik["deq"]; G = gk.shape[0]
            rows = d["rows"].float(); T = rows.shape[0]
            kept = d["kept"]; kept = kept[kept < G * pool]
            recent = torch.cat([torch.arange(D.SINKS), torch.arange(max(0, T - D.RECENT_WINDOW), T)])
            km = torch.zeros(G, dtype=torch.bool); km[(kept // pool).unique()] = True
            rm = torch.zeros(G, dtype=torch.bool); rm[(recent[recent < G * pool] // pool).unique()] = True
            nr = km & ~rm
            for i in range(d["qcal"].shape[0]):
                q = d["qcal"][i].float(); oracle = (rows @ q.T).amax(1)
                s = dsa_score(d["qidx"][i], d["qgate"][i], gk)
                ks = s[km]
                argmax_recent += bool(rm[torch.nonzero(km).flatten()[ks.argmax()]])
                mkept += recall(oracle, kept)[1]
                dsa_sel = expand(s.topk(min(a.cap, G)).indices, pool, T)
                mdsa += recall(oracle, dsa_sel)[1]; bdsa += IDX_B * T + 2048 * ROW_B
                arch = s.clone(); arch[km] = -float("inf")
                for l in labels:
                    if l == "budget":
                        thr = -float("inf")
                    elif l == "nr-max":
                        thr = s[nr].max() if nr.any() else -float("inf")
                    else:
                        thr = torch.quantile(ks, float(l[1:]))
                    cand = arch > thr; n = int(cand.sum())
                    if n:
                        sel = arch.topk(min(a.cap, n)).indices
                    else:
                        sel = torch.zeros(0, dtype=torch.int64)
                    att = torch.cat([kept, expand(sel, pool, T)]).unique()
                    mass[l] += recall(oracle, att)[1]; fired[l] += len(sel); capd[l] += n > a.cap
                    byt[l] += IDX_B * T + att.numel() * ROW_B
                nq += 1
        print(f"\n== {dname}: {nq} queries, T={T}, cap={a.cap}   kept-only mass={mkept/nq:.3f}   "
              f"DSA top-{a.cap} mass={mdsa/nq:.3f} bytes=1.000   argmax-kept-in-recent={argmax_recent/nq:.2f}")
        print(f"{'thr':<8} {'mass':>6} {'fired':>7} {'capped':>7} {'bytes/DSA':>10}")
        for l in labels:
            print(f"{l:<8} {mass[l]/nq:>6.3f} {fired[l]/nq:>7.1f} {capd[l]/nq:>7.3f} {byt[l]/bdsa:>10.3f}")


if __name__ == "__main__":
    main()
