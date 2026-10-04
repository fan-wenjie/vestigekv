"""If a certificate cannot cut the indexer read, can a cheap PREFILTER?

    python mexp/glm53/indexer_sketch_prefilter.py [--dirs cd_g2,cd_w2]

The steelman of "fit one vector and take the norm": store a fitted PROJECTION
instead of a norm. Same 4 B per group, but a projection carries direction and a
norm does not, and direction is what the certificate was missing (the norm
bound's slack is 63-127 against margins of 2.6-5.9, and only 5-22 of that slack
is the discarded negative-gate heads -- the rest is Cauchy-Schwarz being blind
to angle, cos ~ 0.09 in 128 dims).

Generalise to m coefficients on the key set's own principal directions, score
the indexer's REAL function on the m-term reconstruction (relu and the head
gates included), rank, keep the top N groups, and read full keys only for
those. Soundness is given up, so what matters is recall of the true top-512.

A fitted vector ADDED to the keys cannot do this at all: inside relu it
contributes <q_h, w>, which is constant across groups, so it adds no
per-group information -- and relu(u + a) != relu(u) + a, so it is not even a
free reparametrisation; it would change the selection. Softmax shift
invariance does not rescue it either: a constant shift moves the bound and the
threshold equally, so the test U < tau is unchanged.
"""
import argparse, glob
import torch


def score(qidx, gate, gkeys):
    return (torch.relu(qidx.float() @ gkeys.T.float()) * gate.float()[:, None]).sum(0)


def basis(K, m):
    mu = K.mean(0)
    X = K - mu
    _, _, Vh = torch.linalg.svd(X, full_matrices=False)
    V = Vh[:m]
    return mu, V, X @ V.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", default="cd_g2,cd_w2")
    ap.add_argument("--ms", default="1,4,8,16,32")
    ap.add_argument("--ns", default="512,1024,2048")
    ap.add_argument("--cap", type=int, default=512)
    a = ap.parse_args()
    torch.set_num_threads(8)
    MS = [int(x) for x in a.ms.split(",")]
    NS = [int(x) for x in a.ns.split(",")]
    agg, Gs = {}, []
    for dname in a.dirs.split(","):
        for f in sorted(glob.glob(f"results/glm53/{dname}/*.pt")):
            d = torch.load(f, map_location="cpu", weights_only=False)
            K = d["index_k"]["deq"].float()
            G = K.shape[0]
            Gs.append(G)
            for m in MS:
                mu, V, c = basis(K, m)
                Kh = mu[None, :] + c @ V
                for i in range(len(d["qidx"])):
                    q, g = d["qidx"][i].float(), d["qgate"][i].float()
                    top = set(score(q, g, K).topk(min(a.cap, G)).indices.tolist())
                    sh = score(q, g, Kh)
                    for N in NS:
                        pre = set(sh.topk(min(N, G)).indices.tolist())
                        agg.setdefault((m, N), []).append(len(top & pre) / max(len(top), 1))
    Gm = sum(Gs) / len(Gs)
    print(f"recall of the true top-{a.cap} groups, mean G={Gm:.0f}")
    print(f"{'m':>3}{'B/group':>9}{'B/token':>9}" + "".join(f"{'N='+str(N):>10}" for N in NS))
    for m in MS:
        row = f"{m:>3}{4*m:>9}{m:>9d}"
        for N in NS:
            v = agg[(m, N)]
            row += f"{sum(v)/len(v):>10.3f}"
        print(row)
    print(f"\nfixed-term read per token (DSA reads 33 B and stores 33 B):")
    for m in MS:
        for N in NS:
            r = sum(agg[(m, N)]) / len(agg[(m, N)])
            if r > 0.95:
                print(f"  m={m:<3} N={N:<5} recall {r:.3f}  read {m + 33*N/Gm:5.1f} B/token"
                      f"  store {33 + m:.0f} B/token")


if __name__ == "__main__":
    main()
