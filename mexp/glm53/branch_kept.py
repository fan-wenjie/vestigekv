"""Kept-set sweep for the GLM branch rule (follows branch_curve.py).

branch_curve.py: the fetch side works (q0.9 of the kept scores matches DSA's
mass at 4-7x fewer fetched rows) and the KEPT set is the byte term. In the
dumps that set is sinks + 512 sigma picks + the whole un-closed tail (~1.7k
rows at CLOSE_BLOCK=4096), so this rebuilds it from the dump's sigma record as
  sinks + last W rows + top-K positions by tier-1 sigma (closed prefix)
and sweeps W x K x threshold quantile -> (oracle mass, fired groups, bytes/DSA).
The tail W is what a smaller CLOSE_BLOCK buys; K is rho.
"""
import argparse, glob, sys
import torch
sys.path.insert(0, "/home/user/vestigekv-wt/engine-glm521/python")
from sglang.srt.layers.attention.vestigekv import defaults as D  # noqa: E402
from branch_recall import recall, expand, dsa_score, ROW_B, IDX_B  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", default="cd_g2,cd_w2,cd_w3")
    ap.add_argument("--cap", type=int, default=512)
    a = ap.parse_args()
    torch.set_num_threads(8)
    Ws = [256, 512, 1024]; Ks = [0, 256, 512, 1024, 2048]; qs = [0.9, 0.75]
    cells = [(W, K, q) for W in Ws for K in Ks for q in qs]
    for dname in a.dirs.split(","):
        files = sorted(glob.glob(f"results/glm53/{dname}/*.pt"))
        mass = {c: 0.0 for c in cells}; fired = {c: 0.0 for c in cells}; byt = {c: 0.0 for c in cells}
        bdsa = 0.0; mdsa = 0.0; nq = 0
        for f in files:
            d = torch.load(f, map_location="cpu", weights_only=False)
            ik = d["index_k"]; pool = int(ik["pool_size"]); gk = ik["deq"]; G = gk.shape[0]
            rows = d["rows"].float(); T = rows.shape[0]; sig = d["sigma"]; C = sig.shape[0]
            order = sig.argsort(descending=True)
            for i in range(d["qcal"].shape[0]):
                q = d["qcal"][i].float(); oracle = (rows @ q.T).amax(1)
                s = dsa_score(d["qidx"][i], d["qgate"][i], gk)
                mdsa += recall(oracle, expand(s.topk(min(a.cap, G)).indices, pool, T))[1]
                bdsa += IDX_B * T + 2048 * ROW_B
                for W in Ws:
                    tail = torch.arange(max(0, T - W), T)
                    for K in Ks:
                        picks = order[:K]; picks = picks[picks < T - W]
                        kept = torch.cat([torch.arange(D.SINKS), picks, tail]).unique()
                        km = torch.zeros(G, dtype=torch.bool); km[(kept[kept < G * pool] // pool).unique()] = True
                        ks = s[km]; arch = s.clone(); arch[km] = -float("inf")
                        for qq in qs:
                            thr = torch.quantile(ks, qq)
                            cand = arch > thr; n = int(cand.sum())
                            sel = arch.topk(min(a.cap, n)).indices if n else torch.zeros(0, dtype=torch.int64)
                            att = torch.cat([kept, expand(sel, pool, T)]).unique()
                            c = (W, K, qq)
                            mass[c] += recall(oracle, att)[1]; fired[c] += len(sel)
                            byt[c] += IDX_B * T + att.numel() * ROW_B
                nq += 1
        print(f"\n== {dname}: {nq} queries, T={T}   DSA top-{a.cap} mass={mdsa/nq:.3f} bytes=1.000")
        print(f"{'W':>5} {'K':>5} {'q':>5} {'mass':>6} {'fired':>6} {'bytes/DSA':>10}")
        for c in cells:
            print(f"{c[0]:>5} {c[1]:>5} {c[2]:>5} {mass[c]/nq:>6.3f} {fired[c]/nq:>6.1f} {byt[c]/bdsa:>10.3f}")


if __name__ == "__main__":
    main()
