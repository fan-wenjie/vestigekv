"""Offline certificate study over SGLANG_DEBUG_VESTIGEKV_DUMP_DIR snapshots.

    python mexp/glm53/cert_offline.py [--dir results/glm53/caldump] [--ranks 64,128,256]
                                      [--bases qpca,kpca,kmom,mix,diag] [--lids 19,23]

Per snapshot (one request slot x layer): the true causal scores of the calibration
queries against the closed prefix give, per query, the best kept score (max1) and
the best archived score; `need` is the number of archived rows a query truly
prefers to max1 (what an exact index would fetch). For every (basis, rank) the
VestigeKV certificate is refitted exactly as recall_tier.build does (conformal zp
on every query's best archived row) and the fire count per query is counted:
fallback = fraction of queries whose fire count exceeds the recall capacity.

Bases (rows of V orthonormal, r x kv):
  qpca  centered PCA of the calibration queries       (what the server uses)
  kpca  centered PCA of the archived content rows
  kmom  uncentered second moment of the content rows  (keeps the key mean / heavy dims)
  mix   top r/2 of qpca + top r/2 of kmom, re-orthonormalized
  diag  the r coordinates with the largest E[q_i^2] E[k_i^2]  (axis-aligned)
"""

import argparse
import glob
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "engine", "python"))

from sglang.srt.layers.attention.vestigekv import defaults as D  # noqa: E402

NEG = torch.finfo(torch.float32).min


def orthonormal_rows(M):
    q, _ = torch.linalg.qr(M.T)
    return q.T.contiguous()


def basis(kind, r, qc, kc, kraw):
    if kind == "qpca":
        x = qc - qc.mean(0, keepdim=True)
        _, evecs = torch.linalg.eigh(x.T @ x)
        return evecs[:, -r:].T.flip(0).contiguous()
    if kind == "kpca":
        x = kc - kc.mean(0, keepdim=True)
        _, evecs = torch.linalg.eigh(x.T @ x)
        return evecs[:, -r:].T.flip(0).contiguous()
    if kind == "kmom":
        _, evecs = torch.linalg.eigh(kraw.T @ kraw)
        return evecs[:, -r:].T.flip(0).contiguous()
    if kind == "mix":
        return orthonormal_rows(torch.cat([basis("qpca", r // 2, qc, kc, kraw), basis("kmom", r - r // 2, qc, kc, kraw)]))
    if kind == "diag":
        w = (qc.square().mean(0)) * (kraw.square().mean(0))
        idx = w.topk(r).indices
        V = torch.zeros(r, qc.shape[1])
        V[torch.arange(r), idx] = 1.0
        return V
    raise ValueError(kind)


def q(t, p):
    return float(t.float().quantile(p))


def study(snap, ranks, bases, capacity):
    kv = snap["geom"]["kv_lora_rank"]
    sc = float(snap["scale"])
    rows = snap["rows"].float()  # [T, latent]
    T = rows.shape[0]
    keep = torch.isin(snap["row_slots"], snap["kept"])
    qe = snap["qcal"].reshape(-1, rows.shape[1]).float()  # [nH, latent]
    H = snap["qcal"].shape[1]
    qpos = snap["qpos"].repeat_interleave(H)
    S = (qe @ rows.T) * sc
    S.masked_fill_(torch.arange(T)[None, :] > qpos[:, None], NEG)
    kept_S = S.masked_fill(~keep[None, :], NEG)
    max1 = kept_S.max(-1).values
    arch_S = S.masked_fill(keep[None, :], NEG)
    abest, atgt = arch_S.max(-1)
    need = (arch_S > max1[:, None]).sum(-1)
    has = abest > NEG
    content = rows[:, :kv]
    qc = qe[:, :kv]
    arch_idx = (~keep).nonzero().flatten()
    kc = content[arch_idx]
    out = {"T": T, "A": int(arch_idx.numel()), "nq": int(qe.shape[0]),
           "need_p50": q(need, 0.5), "need_p90": q(need, 0.9), "need_max": int(need.max()),
           "key_mean_energy": float(content.mean(0).square().sum() / content.square().sum(-1).mean()),
           "rows": []}
    for kind in bases:
        for r in ranks:
            V = basis(kind, r, qc, kc, content)
            csk = content @ V.T
            rho = (content - csk @ V).norm(dim=-1)
            qsk = qc @ V.T
            qres = (qc - qsk @ V).norm(dim=-1)
            idxs = (qsk @ csk.T) * sc  # [nq, T]
            cert = (qres[:, None] * rho[None, :]) * sc / (kv - r) ** 0.5
            it = idxs[torch.arange(idxs.shape[0]), atgt][has]
            ct = cert[torch.arange(idxs.shape[0]), atgt][has].clamp_min(D.ENTROPY_EPS)
            z_req = (abest[has] - it) / ct
            k = D.conformal_k(int(has.sum()), D.RECALL_TARGET)
            zp = min(float(z_req.kthvalue(k).values), D.Z_MAX)
            score = idxs + zp * cert
            score.masked_fill_(keep[None, :] | (torch.arange(T)[None, :] > qpos[:, None]), NEG)
            fire = (score > max1[:, None]).sum(-1)
            fire0 = (idxs.masked_fill(keep[None, :] | (torch.arange(T)[None, :] > qpos[:, None]), NEG) > max1[:, None]).sum(-1)
            qe_frac = float((qsk.square().sum(-1) / qc.square().sum(-1)).mean())
            ke_frac = float((csk.square().sum(-1) / content.square().sum(-1)).mean())
            out["rows"].append({
                "basis": kind, "r": r, "zp": zp, "z_p50": q(z_req, 0.5),
                "fire_p50": q(fire, 0.5), "fire_p90": q(fire, 0.9), "fire_max": int(fire.max()),
                "fallback": float((fire > capacity).float().mean()),
                "fire0_p50": q(fire0, 0.5), "q_energy": qe_frac, "k_energy": ke_frac,
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "results", "glm53", "caldump"))
    ap.add_argument("--ranks", default="64,128,256")
    ap.add_argument("--bases", default="qpca,kpca,kmom,mix,diag")
    ap.add_argument("--lids", default="")
    ap.add_argument("--capacity", type=int, default=4096)
    args = ap.parse_args()
    ranks = [int(x) for x in args.ranks.split(",")]
    bases = args.bases.split(",")
    lids = {int(x) for x in args.lids.split(",") if x}
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    files = sorted(glob.glob(os.path.join(args.dir, "cal_*.pt")))
    if not files:
        raise SystemExit(f"no cal_*.pt under {args.dir}")
    agg = {}
    for f in files:
        snap = torch.load(f)
        if lids and snap["lid"] not in lids:
            continue
        res = study(snap, ranks, bases, args.capacity)
        print(f"== {os.path.basename(f)} T={res['T']} A={res['A']} nq={res['nq']} need p50/p90/max="
              f"{res['need_p50']:.0f}/{res['need_p90']:.0f}/{res['need_max']} key_mean_energy={res['key_mean_energy']:.3f}")
        print(f"  {'basis':6s} {'r':>4s} {'zp':>5s} {'z50':>5s} {'fire50':>7s} {'fire90':>7s} {'firemax':>7s} {'fallbk':>6s} {'fire0':>6s} {'qE':>5s} {'kE':>5s}")
        for row in res["rows"]:
            print(f"  {row['basis']:6s} {row['r']:4d} {row['zp']:5.2f} {row['z_p50']:5.2f} {row['fire_p50']:7.0f} {row['fire_p90']:7.0f} "
                  f"{row['fire_max']:7d} {row['fallback']:6.3f} {row['fire0_p50']:6.0f} {row['q_energy']:5.2f} {row['k_energy']:5.2f}")
            a = agg.setdefault((row["basis"], row["r"]), [])
            a.append(row)
    print("== mean over snapshots")
    print(f"  {'basis':6s} {'r':>4s} {'zp':>5s} {'fire50':>7s} {'fire90':>7s} {'fallbk':>6s} {'qE':>5s} {'kE':>5s}  n")
    for (kind, r), rows in agg.items():
        n = len(rows)
        m = lambda key: sum(x[key] for x in rows) / n
        print(f"  {kind:6s} {r:4d} {m('zp'):5.2f} {m('fire_p50'):7.0f} {m('fire_p90'):7.0f} {m('fallback'):6.3f} {m('q_energy'):5.2f} {m('k_energy'):5.2f}  {n}")


if __name__ == "__main__":
    main()
