"""DSA-indexer certificate study on SGLANG_DEBUG_VESTIGEKV_DUMP_DIR snapshots that carry
the indexer side (engine branch vestigekv-dsa-index).

    python mexp/glm53/dsa_cert_offline.py [--dir results/glm53/dsadump] [--deltas 0.05,0.01]

Per snapshot the calibration queries' true per-head attention (causal softmax over the
closed prefix, kept + archived) is the label. Three row proxies are scored:
  sketch   VestigeKV's content sketch (query-PCA rank r, residual certificate); a row's
           "z needed" is min over heads of (max1_h - idxs_h) / cert_h, as the scan fires it
  idx_tok  DSA index score per token, I(t,s) = sum_j w_tj relu(q_tj . k_s); z needed =
           max over kept rows of I(t, .) - I(t, s)
  idx_pool the same over KPool-compressed keys (softmax(gate + ape) over each group of 4,
           tail rows per token), the score of a group given to its member rows
under two criteria:
  max      fire every archived row some head truly scores above its best kept row
           (VestigeKV's recall need)
  mass<d>  fire archived rows until the worst head's uncovered attention mass <= delta
z is the conformal quantile of the per-query z needed (D.conformal_k at D.RECALL_TARGET);
the fire counts at that z are then reported over the same queries (in-sample), with the
oracle count (fewest rows by true mass) for the mass criterion.
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


def q(t, p):
    return float(t.float().quantile(p))


def labels(snap):
    kv = snap["geom"]["kv_lora_rank"]
    sc = float(snap["scale"])
    rows = snap["rows"].float()
    T = rows.shape[0]
    keep = torch.isin(snap["row_slots"], snap["kept"])
    qcal = snap["qcal"].float()  # [n, H, latent]
    n, H = qcal.shape[:2]
    qpos = snap["qpos"]
    S = torch.einsum("nhd,td->nht", qcal, rows) * sc  # [n, H, T]
    causal = torch.arange(T)[None, :] <= qpos[:, None]  # [n, T]
    S = S.masked_fill(~causal[:, None, :], NEG)
    P = torch.softmax(S, dim=-1)  # true attention per head
    max1 = S.masked_fill(~keep[None, None, :], NEG).max(-1).values  # [n, H]
    need = (S > max1[..., None]) & (~keep)[None, None, :]  # [n, H, T]
    return dict(rows=rows, kv=kv, sc=sc, T=T, keep=keep, qcal=qcal, n=n, H=H, qpos=qpos,
                S=S, P=P, max1=max1, need=need.any(1), causal=causal)


def z_sketch(L, r):
    qc = L["qcal"][..., : L["kv"]]  # [n, H, kv]
    x = qc.reshape(-1, L["kv"])
    x = x - x.mean(0, keepdim=True)
    _, ev = torch.linalg.eigh(x.T @ x)
    V = ev[:, -r:].T.flip(0)
    content = L["rows"][:, : L["kv"]]
    csk = content @ V.T
    rho = (content - csk @ V).norm(dim=-1)  # [T]
    qsk = qc @ V.T
    qres = (qc - qsk @ V).norm(dim=-1)  # [n, H]
    idxs = torch.einsum("nhr,tr->nht", qsk, csk) * L["sc"]
    cert = (qres[..., None] * rho[None, None, :]) * L["sc"] / (L["kv"] - r) ** 0.5
    zneed = (L["max1"][..., None] - idxs) / cert.clamp_min(D.ENTROPY_EPS)  # [n, H, T]
    return zneed.min(1).values  # [n, T]: fired by any head


def index_scores(snap, L, pooled):
    k = snap["idx_k"].float()  # [T, d]
    T = L["T"]
    if pooled:
        kp = snap["idx_kpool"] if "idx_kpool" in snap else None
        gate = snap["idx_gate"].float()
        ape = snap["idx_ape"].float()  # [4, d]
        G = ape.shape[0]
        full = (T // G) * G
        kg = k[:full].view(-1, G, k.shape[1])
        sg = gate[:full].view(-1, G, k.shape[1]) + ape[None]
        w = torch.softmax(sg, dim=1)
        pooled_k = (w * kg).sum(1)  # [T//G, d]
        keys = torch.cat([pooled_k.repeat_interleave(G, 0), k[full:]])  # per-row key
    else:
        keys = k
    qi, wi = snap["idx_q"].float(), snap["idx_w"].float()  # [m, HI, d], [m, HI]
    I = torch.einsum("mjd,td->mjt", qi, keys).relu()
    I = (wi[..., None] * I).sum(1)  # [m, T]
    return I


def z_index(L, I):
    m_t = I.masked_fill(~L["keep"][None, :], NEG).max(-1).values  # [m]
    return m_t[:, None] - I  # [m, T]


def evaluate(L, zneed, deltas):
    """zneed [n, T] (archived rows only matter). Returns rows of results."""
    n, T = zneed.shape
    arch = ~L["keep"]
    zneed = zneed.masked_fill(~arch[None, :] | ~L["causal"], float("inf"))
    out = []
    # max criterion
    z_q = torch.where(L["need"], zneed, torch.full_like(zneed, -float("inf"))).max(-1).values
    z_q = z_q.clamp_min(0.0)
    k = D.conformal_k(n, D.RECALL_TARGET)
    zp = float(z_q.kthvalue(min(k, n)).values)
    fire = (zneed <= zp).sum(-1)
    out.append(("max", zp, q(fire, 0.5), q(fire, 0.9), int(fire.max()), float("nan")))
    # mass criteria
    P = L["P"]  # [n, H, T]
    for delta in deltas:
        z_q = torch.zeros(n)
        oracle = torch.zeros(n)
        for i in range(n):
            r = zneed[i]
            order = torch.argsort(r)
            pa = P[i][:, order] * arch[order][None, :]  # archived mass in fire order
            suffix = pa.flip(-1).cumsum(-1).flip(-1)  # mass of rows at rank >= k
            U = suffix.max(0).values  # worst head
            U = torch.cat([U, U.new_zeros(1)])
            kstar = int((U <= delta).nonzero()[0])  # rows fired
            z_q[i] = r[order[kstar - 1]] if kstar > 0 else 0.0
            # oracle: fewest archived rows by true mass (per head worst, greedy by max-head mass)
            pm = (P[i] * arch[None, :]).max(0).values
            oo = torch.argsort(pm, descending=True)
            po = P[i][:, oo] * arch[oo][None, :]
            so = po.flip(-1).cumsum(-1).flip(-1).max(0).values
            so = torch.cat([so, so.new_zeros(1)])
            oracle[i] = int((so <= delta).nonzero()[0])
        zp = float(z_q.kthvalue(min(k, n)).values)
        fire = (zneed <= zp).sum(-1)
        out.append((f"mass{delta:g}", zp, q(fire, 0.5), q(fire, 0.9), int(fire.max()), float(fire.float().mean() / oracle.mean().clamp_min(1))))
    return out


def rel_error(L, sel):
    """Latent-space attention output error of attending only `sel` [n, T] rows
    (softmax renormalized over them) vs the full causal softmax; the MLA output
    is W_UV applied to this latent aggregate, so the relative error carries over
    up to that fixed linear map. Returns [n, H]."""
    P = L["P"]  # [n, H, T]
    rows = L["rows"][:, : L["kv"]]
    o_true = torch.einsum("nht,td->nhd", P, rows)
    Ps = P * sel[:, None, :]
    o_sel = torch.einsum("nht,td->nhd", Ps, rows) / Ps.sum(-1, keepdim=True).clamp_min(1e-12)
    return (o_sel - o_true).norm(dim=-1) / o_true.norm(dim=-1).clamp_min(1e-12)


def error_table(L, proxies, I_pool, topk):
    """Rows attended by each selection rule at a common budget and the resulting
    latent output error (p50 / p90 over query-heads)."""
    n, T = proxies["sketch"].shape
    keep = L["keep"]
    arch = ~keep
    causal = L["causal"]
    out = []

    def report(name, sel, budget):
        e = rel_error(L, sel & causal)
        out.append((name, float(budget), q(e, 0.5), q(e, 0.9), float(e.max())))

    # (a) kept + tail only
    report("kept+tail", keep[None, :].expand(n, T), float(keep.sum()))
    # (b) VestigeKV: kept + rows fired by the sketch at its conformal z (max criterion)
    k = D.conformal_k(n, D.RECALL_TARGET)
    zs = proxies["sketch"].masked_fill(~arch[None, :] | ~causal, float("inf"))
    z_q = torch.where(L["need"], zs, torch.full_like(zs, -float("inf"))).max(-1).values.clamp_min(0.0)
    zp = float(z_q.kthvalue(min(k, n)).values)
    fired = zs <= zp
    budget = float((keep[None, :] | fired).sum(-1).float().mean())
    report("kept+sketch-fire", keep[None, :] | fired, budget)
    # (c) the indexer at the same per-query budget: kept + top fired-count rows by I
    nf = fired.sum(-1)
    zi = proxies["idx_pool"].masked_fill(~arch[None, :] | ~causal, float("inf"))
    order = torch.argsort(zi, dim=-1)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(T)[None, :].expand(n, T))
    report("kept+idx-fire(same n)", keep[None, :] | (rank < nf[:, None]), budget)
    # (d) DSA as shipped: top-k rows by pooled index score over the causal prefix, tail included
    sc_rank = torch.empty_like(order)
    ordI = torch.argsort(I_pool.masked_fill(~causal, NEG), dim=-1, descending=True)
    sc_rank.scatter_(1, ordI, torch.arange(T)[None, :].expand(n, T))
    tail = torch.arange(T)[None, :] >= (L["qpos"][:, None] - 63)  # KPool always-selected tail
    report(f"dsa-top{topk}", (sc_rank < topk) | tail, float(min(topk, T)))
    # (e) oracle at VestigeKV's budget: kept + the highest true max-head-mass archived rows
    pm = (L["P"] * arch[None, None, :]).max(1).values  # [n, T]
    ordm = torch.argsort(pm, dim=-1, descending=True)
    mrank = torch.empty_like(ordm)
    mrank.scatter_(1, ordm, torch.arange(T)[None, :].expand(n, T))
    report("kept+oracle-fire(same n)", keep[None, :] | (mrank < nf[:, None]), budget)
    # (f) dense-equivalent check
    report("all", causal, float(T))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "results", "glm53", "dsadump"))
    ap.add_argument("--deltas", default="0.05,0.01")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--lids", default="")
    ap.add_argument("--topk", type=int, default=2048, help="DSA index_topk for the as-shipped selection")
    ap.add_argument("--every", type=int, default=1, help="use every k-th snapshot file")
    args = ap.parse_args()
    deltas = [float(x) for x in args.deltas.split(",")]
    lids = {int(x) for x in args.lids.split(",") if x}
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    files = sorted(glob.glob(os.path.join(args.dir, "cal_*.pt")))[:: args.every]
    agg = {}
    for f in files:
        snap = torch.load(f)
        if lids and snap["lid"] not in lids:
            continue
        if snap.get("idx_q") is None:
            print(f"== {os.path.basename(f)}: no indexer queries, skipped")
            continue
        L = labels(snap)
        # pair calibration queries with indexer queries by order; report the offset
        m = snap["idx_q"].shape[0]
        n = L["n"]
        if m != n:
            print(f"== {os.path.basename(f)}: {n} calibration vs {m} indexer queries, using the first {min(m, n)}")
        keepn = min(m, n)
        for key in ("qcal", "qpos", "S", "P", "max1", "need", "causal"):
            L[key] = L[key][:keepn]
        L["n"] = keepn
        off = [int(a) - int(b) for a, b in zip(snap["idx_qpos"][:keepn], snap["qpos"][:keepn].tolist())]
        good = [i for i, o in enumerate(off) if o == -1]  # the stashed query is the previous forward's token
        if len(good) < keepn:
            for key in ("qcal", "qpos", "S", "P", "max1", "need", "causal"):
                L[key] = L[key][good]
            snap["idx_q"], snap["idx_w"] = snap["idx_q"][good], snap["idx_w"][good]
            L["n"] = keepn = len(good)
        mass_arch = (L["P"] * (~L["keep"])[None, None, :]).sum(-1)  # [n, H]
        print(f"== {os.path.basename(f)} T={L['T']} A={int((~L['keep']).sum())} n={keepn} idx-qpos offset={sorted(set(off))} "
              f"archived mass p50/p90={q(mass_arch, 0.5):.3f}/{q(mass_arch, 0.9):.3f} need p90/max={q(L['need'].sum(-1), 0.9):.0f}/{int(L['need'].sum(-1).max())}")
        print(f"  {'proxy':9s} {'crit':9s} {'zp':>7s} {'fire50':>7s} {'fire90':>7s} {'firemax':>7s} {'fired/oracle':>12s}")
        proxies = {"sketch": z_sketch(L, args.rank)}
        I_pool = None
        for pooled in (False, True):
            I = index_scores(snap, L, pooled)[:keepn]
            if pooled:
                I_pool = I
            proxies["idx_pool" if pooled else "idx_tok"] = z_index(L, I)
        for name, zneed in proxies.items():
            for crit, zp, f50, f90, fmax, ratio in evaluate(L, zneed, deltas):
                print(f"  {name:9s} {crit:9s} {zp:7.2f} {f50:7.0f} {f90:7.0f} {fmax:7d} {ratio:12.2f}")
                agg.setdefault((name, crit), []).append((f50, f90, fmax, ratio))
        print(f"  {'selection':26s} {'rows':>7s} {'err50':>7s} {'err90':>7s} {'errmax':>7s}")
        for name, budget, e50, e90, emax in error_table(L, proxies, I_pool, args.topk):
            print(f"  {name:26s} {budget:7.0f} {e50:7.3f} {e90:7.3f} {emax:7.3f}")
            agg.setdefault(("err", name), []).append((budget, e50, e90, emax))
    print("== mean over snapshots")
    print(f"  {'proxy':9s} {'crit':9s} {'fire50':>7s} {'fire90':>7s} {'firemax':>7s} {'fired/oracle':>12s}  n")
    for (name, crit), rows in agg.items():
        if name == "err":
            continue
        c = len(rows)
        mean = lambda i: sum(r[i] for r in rows if r[i] == r[i]) / c
        print(f"  {name:9s} {crit:9s} {mean(0):7.0f} {mean(1):7.0f} {mean(2):7.0f} {mean(3):12.2f}  {c}")
    print(f"  {'selection':26s} {'rows':>7s} {'err50':>7s} {'err90':>7s} {'errmax':>7s}  n")
    for (tag, name), rows in agg.items():
        if tag != "err":
            continue
        c = len(rows)
        mean = lambda i: sum(r[i] for r in rows) / c
        print(f"  {name:26s} {mean(0):7.0f} {mean(1):7.3f} {mean(2):7.3f} {mean(3):7.3f}  {c}")


if __name__ == "__main__":
    main()
