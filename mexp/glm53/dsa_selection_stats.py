"""What DSA's own top-2048 selects, and which low-pass cutoff predicts it.

Offline, from the calibration dumps (results/glm53/cd_*): each holds a layer's
dequantised index-k (one 128-dim key per index_kpool=4 positions), the indexer's
real queries and per-head gates, and the sigma the served arm computed. So DSA's
selection can be reproduced exactly --

    s_g = sum_h gate_h * relu(<q_h, k_g>)        (mexp/glm53/indexer_prune.py)

-- and two questions answered without a GPU:

1. Does the selection have HISTORY? If a small set of groups is chosen by almost
   every query, a static cache of them is most of DSA's budget, and that is what
   tier 1's kept set is trying to be. Measured as the selection frequency's
   concentration, and its correlation with position, key norm and sigma.

2. Which LOWPASS_KAPPA ranks the frequently-selected groups best? The served
   value is 16, chosen by an attention-mass criterion. The job sigma actually
   has here is to rank what DSA will want, so the decisive form of the question
   is the recall of the frequently-selected set inside the top-m by sigma, swept
   over kappa. A kappa whose sigma does not rank them is the wrong cutoff
   whatever the energy spectrum says.
"""

import argparse
import glob
import sys

import torch


def score(qidx, gate, gkeys):
    """DSA's indexer score per group; identical to indexer_prune.score."""
    return (torch.relu(qidx.float() @ gkeys.T.float()) * gate.float()[:, None]).sum(0)


def sigma_at(keys, block, kappa):
    """blockwise_sigma with kappa as a parameter: the rFFT low-pass residual
    norm per row, computed within non-overlapping blocks of `block` rows."""
    n = (keys.shape[0] // block) * block
    if n == 0:
        return torch.zeros(keys.shape[0])
    out = torch.zeros(keys.shape[0])
    for s in range(0, n, block):
        b = keys[s : s + block].float()
        f = torch.fft.rfft(b, dim=0)
        f[kappa:] = 0
        low = torch.fft.irfft(f, n=block, dim=0)
        out[s : s + block] = (b - low).norm(dim=-1)
    return out


def spearman(a, b):
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    d = ra.norm() * rb.norm()
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", default="cd3_64k,cd3_16k")
    ap.add_argument("--topk-rows", type=int, default=2048)
    ap.add_argument("--kappas", default="2,4,8,16,32,64,128")
    ap.add_argument("--block", type=int, default=256, help="group-space sigma window (1024 tokens / pool 4)")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(8)
    kappas = [int(x) for x in a.kappas.split(",")]

    files = []
    for d in a.dirs.split(","):
        files += sorted(glob.glob(f"results/glm53/{d}/*.pt"))
    if a.limit:
        files = files[: a.limit]
    if not files:
        sys.exit("no dumps found")
    print(f"{len(files)} dumps from {a.dirs}, budget {a.topk_rows} rows\n")

    per_layer = {}
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        gk = d["index_k"]["deq"].float()
        pool = int(d["index_k"]["pool_size"])
        G = gk.shape[0]
        topg = max(1, a.topk_rows // pool)
        qs, gates = d["qidx"], d["qgate"]
        if not isinstance(qs, (list, tuple)):
            qs, gates = [qs], [gates]
        cnt = torch.zeros(G)
        for q, g in zip(qs, gates):
            s = score(q, g, gk)
            cnt[s.topk(min(topg, G)).indices] += 1
        rec = per_layer.setdefault(d["lid"], [])
        rec.append({"cnt": cnt, "nq": len(qs), "G": G, "topg": topg, "keys": gk,
                    "pool": pool, "seq": int(d["seq_len"]), "file": f})

    print("== 1. does the selection have history? (per layer, over its dumps)")
    print(f"{'lid':>4s} {'dumps':>6s} {'q':>4s} {'G':>6s} {'budget':>7s} "
          f"{'ever sel':>9s} {'always sel':>11s} {'top-budget by freq':>19s} {'rho_pos':>8s} {'rho_norm':>9s}")
    for lid in sorted(per_layer):
        R = per_layer[lid]
        # aggregate over dumps on the common prefix of groups
        Gm = min(r["G"] for r in R)
        cnt = torch.zeros(Gm)
        nq = 0
        for r in R:
            cnt += r["cnt"][:Gm]
            nq += r["nq"]
        topg = R[0]["topg"]
        ever = float((cnt > 0).float().mean())
        always = float((cnt == nq).float().mean())
        # what share of a single query's budget is covered by the globally
        # most-frequent `budget` groups
        topset = cnt.topk(min(topg, Gm)).indices
        share = float(cnt[topset].sum() / max(cnt.sum(), 1))
        pos = torch.arange(Gm).float()
        nrm = R[0]["keys"][:Gm].norm(dim=-1)
        print(f"{lid:4d} {len(R):6d} {nq:4d} {Gm:6d} {topg:7d} {ever:9.3f} {always:11.3f} "
              f"{share:19.3f} {spearman(cnt,pos):8.3f} {spearman(cnt,nrm):9.3f}")

    print("\n   ever sel   = fraction of groups chosen by at least one query")
    print("   always sel = fraction chosen by EVERY query (the static core)")
    print("   top-budget by freq = share of all selections captured by the")
    print("                        globally most-frequent `budget` groups")
    print("   rho_pos/rho_norm   = Spearman of frequency vs position / key norm")

    print(f"\n== 2. which kappa ranks the frequently-selected groups? (block={a.block} groups)")
    print(f"{'lid':>4s} " + " ".join(f"k={k:<3d}" for k in kappas) + "   <- Spearman(sigma, freq)")
    best = {}
    for lid in sorted(per_layer):
        R = per_layer[lid]
        Gm = min(r["G"] for r in R)
        cnt = torch.zeros(Gm)
        for r in R:
            cnt += r["cnt"][:Gm]
        keys = R[0]["keys"][:Gm]
        row = []
        for k in kappas:
            sg = sigma_at(keys, a.block, k)[:Gm]
            row.append(spearman(sg, cnt))
        print(f"{lid:4d} " + " ".join(f"{v:6.3f}" for v in row))
        for k, v in zip(kappas, row):
            best[k] = best.get(k, 0.0) + v
    n = len(per_layer)
    print("\n   mean over layers:")
    print("   " + "  ".join(f"k={k}: {best[k]/n:+.3f}" for k in kappas))
    bk = max(best, key=lambda k: best[k])
    print(f"\n   best kappa by this criterion: {bk}   (served LOWPASS_KAPPA = 16)")

    print(f"\n== 3. recall of the frequently-selected set inside top-m by sigma")
    print(f"{'lid':>4s} {'m=budget':>9s} " + " ".join(f"k={k:<4d}" for k in kappas))
    for lid in sorted(per_layer):
        R = per_layer[lid]
        Gm = min(r["G"] for r in R)
        cnt = torch.zeros(Gm)
        for r in R:
            cnt += r["cnt"][:Gm]
        topg = min(R[0]["topg"], Gm)
        want = set(cnt.topk(topg).indices.tolist())
        keys = R[0]["keys"][:Gm]
        row = []
        for k in kappas:
            sg = sigma_at(keys, a.block, k)[:Gm]
            got = set(sg.topk(topg).indices.tolist())
            row.append(len(want & got) / max(len(want), 1))
        print(f"{lid:4d} {topg:9d} " + " ".join(f"{v:7.3f}" for v in row))
    print("\n   a random top-m would recall m/G; anything near that means sigma")
    print("   does not rank what DSA wants at that kappa.")


if __name__ == "__main__":
    main()
