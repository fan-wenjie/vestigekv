"""Can a NORM-only certificate discard groups outside the indexer's top-512?

    python mexp/glm53/indexer_prune_norm.py [--dirs cd_g2,cd_w2,cd_w3]

The appeal is the byte account: one norm is 4 B per group against 132 B for the
group's pooled index key, and the bound is MONOTONE in the norm, so the test
collapses to a single scalar threshold and the surviving groups are read as
whole contiguous 132 B records (no positional blocking, so the scatter that
closes the block-bound variant does not apply here).

    s_g = sum_h gate_h relu(<q_h,k_g>)
    relu(u) <= |u| <= |q_h||k_g|, and a negative-gate head's best case is 0, so
    U(g) = ( sum_{gate_h > 0} gate_h |q_h| ) |k_g| = C_q |k_g|

It certifies nothing, for two reasons that are both structural:

1. The indexer LayerNorms its keys (`IndexerKPool.k_norm`, applied at every
   key path), so the norms are near-constant BY CONSTRUCTION: max/min 1.31,
   p1..p99 inside 9%. The norm carries almost no information to begin with.
2. **tau512 is NEGATIVE** in 11 of 12 layer-dumps (-12.4 to +0.4; only 0.1% to
   15% of groups score above zero), because 14 to 17 of the 32 head gates are
   negative. Any bound that drops the negative-gate heads to relu's floor of 0
   is non-negative, so it can never fall below a negative threshold. This kills
   the whole bound FAMILY, not this instance of it: certifying against a
   negative tau needs a strictly positive lower bound on the negative-gate
   heads' dots, which is directional information the norm does not carry.

That is also why the spectral variant (indexer_prune.py) needed m ~ 96 of 128:
m is how much directional information it takes to prove those dots positive.
"""
import argparse, glob, sys
import torch

sys.path.insert(0, "/home/user/vestigekv-wt/engine-glm521/python")
from sglang.srt.layers.attention.vestigekv import defaults as D  # noqa: E402


def score(qidx, gate, gkeys):
    return (torch.relu(qidx.float() @ gkeys.T.float()) * gate.float()[:, None]).sum(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", default="cd_g2,cd_w2,cd_w3")
    ap.add_argument("--cap", type=int, default=512)
    a = ap.parse_args()
    torch.set_num_threads(8)
    print(f"{'dir':<7}{'lid':>4}{'G':>6}{'|k| max/min':>12}{'tau512':>9}"
          f"{'frac s>0':>9}{'neg gates':>10}{'elim':>7}")
    elim_all = []
    for dname in a.dirs.split(","):
        for f in sorted(glob.glob(f"results/glm53/{dname}/*.pt")):
            d = torch.load(f, map_location="cpu", weights_only=False)
            K = d["index_k"]["deq"].float()
            G = K.shape[0]
            pool = int(d["index_k"]["pool_size"])
            kn = K.norm(dim=1)
            T = int(d["seq_len"])
            kept = d["kept"]; kept = kept[kept < G * pool]
            rec = torch.cat([torch.arange(D.SINKS),
                             torch.arange(max(0, T - D.RECENT_WINDOW), T)])
            rec = rec[rec < G * pool]
            cand = torch.zeros(G, dtype=torch.bool)
            cand[(torch.cat([kept, rec]) // pool).unique()] = True
            taus, fps, els = [], [], []
            for i in range(len(d["qidx"])):
                q = d["qidx"][i].float(); g = d["qgate"][i].float()
                s = score(q, g, K)
                k = min(a.cap, G)
                tau = s.topk(k).values[-1]
                Cq = (torch.clamp(g, min=0) * q.norm(dim=1)).sum()
                U = Cq * kn
                assert int(((U < tau) & (s >= tau)).sum()) == 0, "UNSOUND"
                taus.append(float(tau)); fps.append(float((s > 0).float().mean()))
                els.append(float((U < tau).float().mean()))
            elim_all += els
            print(f"{dname:<7}{int(d['lid']):>4}{G:>6}{float(kn.max()/kn.min()):>12.3f}"
                  f"{sum(taus)/len(taus):>9.2f}{sum(fps)/len(fps):>9.3f}"
                  f"{int((d['qgate'][0].float() < 0).sum()):>10}"
                  f"{sum(els)/len(els):>7.3f}")
    print(f"\nmean eliminated {sum(elim_all)/len(elim_all):.3f}  "
          f"(ceiling 1 - {a.cap}/G ~ 0.88)")


if __name__ == "__main__":
    main()
