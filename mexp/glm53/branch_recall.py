"""The GLM branch rule, offline, on SGLANG_DEBUG_VESTIGEKV_DUMP_DIR snapshots.

    python mexp/glm53/branch_recall.py [--dirs cd_g2,cd_w2,cd_w3] [--cap 512]

Rule (memory glm-branch-only): tier 2 scores every pooled index-k group with
DSA's own index score I(g) = sum_h gate_h * relu(q_h . K_g), fires the groups
whose score beats the best KEPT group's score, cap `--cap` groups (512 = 2048
rows = index_topk), one recall = one group of 4 rows. No sketch, no z.

Judged against the same oracle group_oracle.py uses (latent score, best over
heads, top-2048 positions; count = positions held, mass = floor-shifted score
held), beside the as-shipped DSA selection (top-512 groups, no threshold) and
the kept set alone. Two kept sets, because the byte account turns on it:
  rho    tier 1 as served (the dump's kept table: rho*closed + sinks + recent)
  recent sinks + recent window only (260 rows), no long-term kept set
Bytes per layer-step: 33*T (the index read both arms make) + attended rows *
1 KB, against DSA's 33*T + 2048 KB.
"""
import argparse, glob, os, sys
import torch

sys.path.insert(0, "/home/user/vestigekv-wt/engine-glm521/python")
from sglang.srt.layers.attention.vestigekv import defaults as D  # noqa: E402

ROW_B = 1024  # 512 x bf16
IDX_B = 33    # pooled index-k, per token: (128 + 4) / 4


def recall(oracle, selected, rows=2048):
    T = oracle.shape[0]
    top = oracle.topk(min(rows, T))
    hit = torch.zeros(T, dtype=torch.bool)
    sel = selected[(selected >= 0) & (selected < T)]
    hit[sel] = True
    got = hit[top.indices]
    w = top.values - top.values[-1]
    tot = float(w.sum())
    mass = float((w * got).sum() / tot) if tot > 0 else float(got.float().mean())
    return float(got.float().mean()), mass


def expand(groups, pool, T):
    ids = (groups[:, None] * pool + torch.arange(pool)).reshape(-1)
    return ids[ids < T]


def dsa_score(qidx, gate, gkeys):
    # DSA's indexer: per head relu(q.k), gate-weighted sum (dsa_cert_offline)
    logits = torch.relu(qidx.float() @ gkeys.T.float())  # [H, G]
    return (logits * gate.float()[:, None]).sum(0)


def rule(score, kept_groups_mask, cap):
    thr = score[kept_groups_mask].max() if kept_groups_mask.any() else score.min() - 1
    cand = (score > thr) & ~kept_groups_mask
    n = int(cand.sum())
    if n == 0:
        return torch.zeros(0, dtype=torch.int64), 0, False
    s = score.clone(); s[~cand] = -float("inf")
    k = min(cap, n)
    return s.topk(k).indices, n, n > cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", default="cd_g2,cd_w2,cd_w3")
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--rows", type=int, default=2048)
    a = ap.parse_args()
    torch.set_num_threads(8)
    names = ["kept-rho", "rule-rho", "kept-recent", "rule-recent", "dsa-top", "dsa+rho"]
    tot = {}
    for dname in a.dirs.split(","):
        files = sorted(glob.glob(f"results/glm53/{dname}/*.pt"))
        acc = {n: torch.zeros(2) for n in names}
        fire = {"rho": [], "recent": []}; capped = {"rho": 0, "recent": 0}; zero = {"rho": 0, "recent": 0}
        bytes_ = {"rho": 0.0, "recent": 0.0, "dsa": 0.0}; nq = 0
        per_lid = {}
        for f in files:
            d = torch.load(f, map_location="cpu", weights_only=False)
            ik = d["index_k"]; pool = int(ik["pool_size"]); gkeys = ik["deq"]
            rows = d["rows"].float(); T = rows.shape[0]; G = gkeys.shape[0]
            kept = d["kept"]; kept = kept[kept < G * pool]
            recent = torch.cat([torch.arange(D.SINKS), torch.arange(max(0, T - D.RECENT_WINDOW), T)])
            recent = recent[recent < G * pool].unique()
            masks = {}
            for key, ks in (("rho", kept), ("recent", recent)):
                m = torch.zeros(G, dtype=torch.bool); m[(ks // pool).unique()] = True; masks[key] = m
            lid = int(d["lid"])
            for i in range(d["qcal"].shape[0]):
                q = d["qcal"][i].float()
                oracle = (rows @ q.T).amax(1)
                s = dsa_score(d["qidx"][i], d["qgate"][i], gkeys)
                dsa_sel = expand(s.topk(min(a.cap, G)).indices, pool, T)
                r = {}
                r["kept-rho"] = recall(oracle, kept, a.rows)
                r["kept-recent"] = recall(oracle, recent, a.rows)
                r["dsa-top"] = recall(oracle, dsa_sel, a.rows)
                r["dsa+rho"] = recall(oracle, torch.cat([dsa_sel, kept]).unique(), a.rows)
                for key, ks in (("rho", kept), ("recent", recent)):
                    fired, n, cp = rule(s, masks[key], a.cap)
                    att = torch.cat([ks, expand(fired, pool, T)]).unique()
                    r[f"rule-{key}"] = recall(oracle, att, a.rows)
                    fire[key].append(len(fired)); capped[key] += cp; zero[key] += (n == 0)
                    bytes_[key] += IDX_B * T + att.numel() * ROW_B
                bytes_["dsa"] += IDX_B * T + a.rows * ROW_B
                for n_ in names: acc[n_] += torch.tensor(r[n_])
                pl = per_lid.setdefault(lid, [torch.zeros(2), torch.zeros(2), 0])
                pl[0] += torch.tensor(r["rule-rho"]); pl[1] += torch.tensor(r["dsa-top"]); pl[2] += 1
                nq += 1
        print(f"\n== {dname}: {len(files)} snapshots, {nq} queries, T={T}, cap={a.cap} groups")
        print(f"{'selector':<12} {'count':>7} {'mass':>7}")
        for n_ in names:
            v = acc[n_] / nq; print(f"{n_:<12} {v[0]:>7.3f} {v[1]:>7.3f}")
        for key in ("rho", "recent"):
            fg = torch.tensor(fire[key]).float()
            print(f"fire[{key:6s}] mean={fg.mean():6.1f} groups  p50={fg.median():5.0f}  "
                  f"capped={capped[key]/nq:.3f}  zero={zero[key]/nq:.3f}  "
                  f"bytes/DSA={bytes_[key]/bytes_['dsa']:.3f}")
        print("per layer  rule-rho c/m   dsa-top c/m")
        for lid in sorted(per_lid):
            a0, a1, n_ = per_lid[lid]
            print(f"  lid {lid:>2}   {a0[0]/n_:.3f}/{a0[1]/n_:.3f}     {a1[0]/n_:.3f}/{a1[1]/n_:.3f}")
        tot[dname] = {n_: (acc[n_] / nq).tolist() for n_ in names}
    print("\nsummary (mass):", {k: {n_: round(v[1], 3) for n_, v in d_.items()} for k, d_ in tot.items()})


if __name__ == "__main__":
    main()
