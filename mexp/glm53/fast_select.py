"""Two first-principles routes to an approximate top-2048, measured against DSA's own.

ROUTE A -- static candidate pool.  sigma is frozen at close and query-free, so a
pool S_M chosen by sigma costs nothing per step and shrinks BOTH the logits GEMM and
the top-k input. The objection is that a static pool must cover the UNION of every
future query's top-512, not one query's. So the upper bound for ANY static scheme is
the frequency-ranked pool (an oracle that already knows which groups future queries
will want); sigma is the achievable one. If the oracle static pool is also poor, the
limit is staticness and no better query-free ranking can fix it.

ROUTE B -- rank-r query truncation.  All 32 heads are linear images of one c_q and the
q second moment has effective rank 3-6, so q_h ~ sum_j a_hj v_j and
<q_h,k_g> ~ sum_j a_hj <v_j,k_g> with <v_j,k_g> PRECOMPUTABLE at close (NoPE: k_g is
content only and does not move). relu is still applied exactly; the approximation is
only the truncation of q. Per-step bytes drop from 132 per group to 2r.

The basis V_r is fitted on HELD-OUT dumps and applied to the eval dumps, never fitted
on the queries it is scored against.
"""

import argparse, glob, sys, collections
import torch

sys.path.insert(0, "mexp/glm53")
from dsa_selection_stats import score, sigma_at


def qmat(d):
    qs = d["qidx"]
    return [qs] if not isinstance(qs, (list, tuple)) else list(qs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="cd_w2,cd_g2")
    ap.add_argument("--eval", default="cd_w3")
    ap.add_argument("--topk-rows", type=int, default=2048)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--kappa", type=int, default=16)
    ap.add_argument("--fracs", default="0.05,0.10,0.25,0.50")
    ap.add_argument("--ranks", default="2,4,8,16,32,64")
    ap.add_argument("--max-q", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(8)
    fracs = [float(x) for x in a.fracs.split(",")]
    ranks = [int(x) for x in a.ranks.split(",")]

    # --- fit a per-layer query basis on the held-out dirs
    mom = collections.defaultdict(lambda: torch.zeros(128, 128, dtype=torch.double))
    nq_fit = collections.Counter()
    for dn in a.fit.split(","):
        for fp in sorted(glob.glob(f"results/glm53/{dn}/*.pt")):
            d = torch.load(fp, map_location="cpu", weights_only=False)
            Q = torch.cat([q.double() for q in qmat(d)], 0)
            mom[int(d["lid"])] += Q.T @ Q
            nq_fit[int(d["lid"])] += Q.shape[0]
    basis = {}
    for lid, M in mom.items():
        ev, V = torch.linalg.eigh(M)
        basis[lid] = V.flip(1)          # columns = descending eigenvectors
    print(f"query basis fitted on {a.fit}: {len(basis)} layers, "
          f"{sum(nq_fit.values())} query-head rows\n")

    recA = {f: [] for f in fracs}
    recA_oracle = {f: [] for f in fracs}
    recB = {r: [] for r in ranks}
    nq = 0
    for dn in a.eval.split(","):
        for fp in sorted(glob.glob(f"results/glm53/{dn}/*.pt")):
            d = torch.load(fp, map_location="cpu", weights_only=False)
            lid = int(d["lid"])
            if lid not in basis:
                continue
            gk = d["index_k"]["deq"].float()
            pool = int(d["index_k"]["pool_size"])
            G = gk.shape[0]
            topg = max(1, a.topk_rows // pool)
            if G <= topg:
                continue
            nb = G // a.block
            if nb == 0:
                continue
            C = nb * a.block
            sig = sigma_at(gk, a.block, a.kappa)[:C]
            qs, gates = qmat(d), d["qgate"]
            if not isinstance(gates, (list, tuple)):
                gates = [gates]
            pairs = list(zip(qs, gates))[: a.max_q]
            # true selections, and the frequency oracle over THIS dump's queries
            truth, freq = [], torch.zeros(G)
            for q, g in pairs:
                s = score(q, g, gk)
                t = s.topk(topg).indices
                truth.append((s, set(t.tolist())))
                freq[t] += 1
            V = basis[lid].float()
            P = gk @ V                      # [G, 128] precomputed projections
            for (s, want), (q, g) in zip(truth, pairs):
                nq += 1
                for f in fracs:
                    M = max(1, int(f * G))
                    got = set(sig.topk(min(M, C)).indices.tolist())
                    recA[f].append(len(want & got) / len(want))
                    got2 = set(freq.topk(min(M, G)).indices.tolist())
                    recA_oracle[f].append(len(want & got2) / len(want))
                for r in ranks:
                    # q truncated to V_r, relu applied exactly on the truncation
                    A = (q.float() @ V[:, :r])            # [32, r]
                    approx = (torch.relu(A @ P[:, :r].T) * g.float()[:, None]).sum(0)
                    got3 = set(approx.topk(topg).indices.tolist())
                    recB[r].append(len(want & got3) / len(want))

    print(f"eval on {a.eval}: {nq} queries, DSA cut = {a.topk_rows//4} groups\n")
    print("== ROUTE A: static candidate pool, recall of THIS query's top-512")
    print(f"{'M/G':>8s} {'sigma (achievable)':>20s} {'frequency (static oracle)':>27s}")
    for f in fracs:
        print(f"{f:8.2f} {sum(recA[f])/len(recA[f]):20.4f} "
              f"{sum(recA_oracle[f])/len(recA_oracle[f]):27.4f}")
    print("\n== ROUTE B: rank-r query truncation, recall of the true top-512")
    print(f"{'r':>4s} {'recall':>9s} {'bytes/group':>12s} {'vs 132 B':>9s}")
    for r in ranks:
        print(f"{r:4d} {sum(recB[r])/len(recB[r]):9.4f} {2*r:12d} {132/(2*r):8.1f}x")


if __name__ == "__main__":
    main()
