"""Can the indexer CERTIFY that a group cannot enter its own top-2048?

    python mexp/glm53/indexer_prune.py [--dirs cd_g2,cd_w2,cd_w3] [--cap 512]

The indexer's score is s_g = sum_h gate_h * relu(<q_h, k_g>), one 128-dim NoPE
key per pooled group of 4 rows (GLM passes rope_head_dim=qk_rope_head_dim=0 to
the indexer, so the whole index key space is NoPE). Write the key set as its
own spectrum,

    k_g = mu + sum_{i<m} c_{g,i} v_i + e_g,

project the query onto the basis ONCE per step (m*H dots, not per group), and
the per-group work is m multiply-adds on stored coefficients plus one residual
norm. relu is monotone, so bounding the dot bounds the score term -- but 44% of the
head gates are NEGATIVE (measured: gate in [-0.0068, 0.0040], 14 of 32 heads
below zero on every step of this layer), so a gated sum of upper bounds is NOT
an upper bound. A head with gate_h < 0 needs the dot's LOWER bound, because
the largest value of gate_h*relu(u) is at the smallest relu(u):

    base_{g,h} = <q_h,mu> + sum_i c_{g,i}<q_h,v_i>,   slack = |q_h| eps_g
    U(g) = sum_{gate_h>=0} gate_h relu(base + slack)
         + sum_{gate_h<0}  gate_h relu(base - slack)

U(g) < tau  =>  g is certainly outside the top-`cap` groups, for any tau that
is a LOWER bound on the cap-th best score. This is a certificate, not a
predictor: it is never wrong, only vacuous, so the only question is its rate.

Two taus are reported because they fail differently:
  oracle  the true cap-th best score (upper bound on what any threshold buys)
  kept    the cap-th best among the already-scored candidates (kept groups +
          recent window), which is what a kernel could actually use

And two granularities, because only the coarse one can pay: a per-GROUP bound
still reads every group's coefficients, and the indexer scan is latency-bound
on GLM (0.29 ms/step against an ~8 us bandwidth floor), so removing arithmetic
removes no time. A per-BLOCK bound over `--blocks` consecutive groups reads one
record per block and discards all of them, which is the only shape that cuts
the 33 B/token index read itself.
"""
import argparse, glob, sys
import torch

sys.path.insert(0, "/home/user/vestigekv-wt/engine-glm521/python")
from sglang.srt.layers.attention.vestigekv import defaults as D  # noqa: E402


def score(qidx, gate, gkeys):
    return (torch.relu(qidx.float() @ gkeys.T.float()) * gate.float()[:, None]).sum(0)


def basis(gkeys, m):
    mu = gkeys.mean(0)
    X = gkeys - mu
    # economy SVD on [G,128]; V's columns are the principal key directions
    _, S, Vh = torch.linalg.svd(X, full_matrices=False)
    V = Vh[:m]                      # [m,128]
    c = X @ V.T                     # [G,m]
    e = X - c @ V                   # [G,128]
    eps = e.norm(dim=1)             # [G]
    energy = float((S[:m] ** 2).sum() / (S ** 2).sum())
    return mu, V, c, eps, energy


def bound(qidx, gate, mu, V, c, eps):
    """U(g), the per-group certificate. m*H dots once, then m MACs per group."""
    q = qidx.float(); g = gate.float()
    qmu = q @ mu                    # [H]
    qV = q @ V.T                    # [H,m]
    # e_g lies in the complement of span(V), so <q,e> = <P_perp q, e>: scale the
    # residual by the query's ORTHOGONAL part, not by |q|
    qn = (q - qV @ V).norm(dim=1)   # [H]
    base = c @ qV.T + qmu[None, :]  # [G,H]
    slack = eps[:, None] * qn[None, :]
    hi = torch.relu(base + slack)
    lo = torch.relu(base - slack)
    # a negative gate takes the dot's LOWER bound, or the product grows
    pick = torch.where(g[None, :] >= 0, hi, lo)
    return (pick * g[None, :]).sum(1)


def block_bound(qidx, gate, mu, V, c, eps, B):
    """One record per block of B groups. Each term is maximised independently
    over the block, which is sound and loose; the looseness is what B buys."""
    q = qidx.float(); g = gate.float()
    G, m = c.shape
    nb = (G + B - 1) // B
    pad = nb * B - G
    cp = torch.cat([c, torch.full((pad, m), -float("inf"))]) if pad else c
    ep = torch.cat([eps, torch.full((pad,), -float("inf"))]) if pad else eps
    qmu = q @ mu; qV = q @ V.T; qn = (q - qV @ V).norm(dim=1)
    cb = cp.view(nb, B, m)
    cmax = cb.amax(1); cmin = cb.amin(1)                     # [nb,m]
    pos = torch.clamp(qV, min=0); neg = torch.clamp(qV, max=0)
    # block MAX of the dot (for gate >= 0) and block MIN of it (for gate < 0),
    # each from the coefficient extreme that the sign of <q_h,v_i> selects
    bmax = cmax @ pos.T + cmin @ neg.T + qmu[None, :]        # [nb,H]
    bmin = cmin @ pos.T + cmax @ neg.T + qmu[None, :]
    emax = ep.view(nb, B).amax(1)                            # [nb]
    slack = emax[:, None] * qn[None, :]
    hi = torch.relu(bmax + slack)
    lo = torch.relu(bmin - slack)
    pick = torch.where(g[None, :] >= 0, hi, lo)
    return (pick * g[None, :]).sum(1), nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", default="cd_g2,cd_w2,cd_w3")
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--ms", default="4,8,16,32")
    ap.add_argument("--blocks", default="1,16,64,256")
    a = ap.parse_args()
    torch.set_num_threads(8)
    ms = [int(x) for x in a.ms.split(",")]
    bs = [int(x) for x in a.blocks.split(",")]
    agg = {}
    energy = {}
    nq = 0
    for dname in a.dirs.split(","):
        for f in sorted(glob.glob(f"results/glm53/{dname}/*.pt")):
            d = torch.load(f, map_location="cpu", weights_only=False)
            gkeys = d["index_k"]["deq"].float()
            G = gkeys.shape[0]
            pool = int(d["index_k"]["pool_size"])
            T = int(d["seq_len"])
            kept = d["kept"]; kept = kept[kept < G * pool]
            recent = torch.cat([torch.arange(D.SINKS),
                                torch.arange(max(0, T - D.RECENT_WINDOW), T)])
            recent = recent[recent < G * pool]
            cand = torch.zeros(G, dtype=torch.bool)
            cand[(torch.cat([kept, recent]) // pool).unique()] = True
            for m in ms:
                mu, V, c, eps, en = basis(gkeys, m)
                energy.setdefault(m, []).append(en)
                for i in range(len(d["qidx"])):
                    s = score(d["qidx"][i], d["qgate"][i], gkeys)
                    k = min(a.cap, G)
                    tau_or = s.topk(k).values[-1]
                    sc = s.clone(); sc[~cand] = -float("inf")
                    kk = min(k, int(cand.sum()))
                    tau_kept = sc.topk(kk).values[-1] if kk > 0 else s.min() - 1
                    topmask = torch.zeros(G, dtype=torch.bool)
                    topmask[s.topk(k).indices] = True
                    for B in bs:
                        if B == 1:
                            U = bound(d["qidx"][i], d["qgate"][i], mu, V, c, eps)
                            nb = G
                            keep_top = topmask
                        else:
                            U, nb = block_bound(d["qidx"][i], d["qgate"][i], mu, V, c, eps, B)
                            pad = nb * B - G
                            tm = torch.cat([topmask, torch.zeros(pad, dtype=torch.bool)]) if pad else topmask
                            keep_top = tm.view(nb, B).any(1)
                        for tname, tau in (("oracle", tau_or), ("kept", tau_kept)):
                            elim = U < tau
                            # soundness: a unit holding a true top-k group is never eliminated
                            unsound = int((elim & keep_top).sum())
                            key = (m, B, tname)
                            r = agg.setdefault(key, [0, 0, 0])
                            r[0] += int(elim.sum()); r[1] += nb; r[2] += unsound
                    nq += 1
    print(f"queries x layers: {nq}")
    print("\nspectral energy captured by the top-m NoPE key directions")
    for m in ms:
        v = torch.tensor(energy[m])
        print(f"  m={m:<3} mean {v.mean():.4f}   min {v.min():.4f}   max {v.max():.4f}")
    print(f"\nfraction of units CERTIFIED out of the top-{a.cap} groups")
    print(f"  {'m':>3} {'block':>6} {'tau':>7} {'eliminated':>11} {'unsound':>8}")
    for (m, B, tname), (el, tot, bad) in sorted(agg.items()):
        print(f"  {m:>3} {B:>6} {tname:>7} {el/max(tot,1):>10.3f}  {bad:>8}")


if __name__ == "__main__":
    main()
