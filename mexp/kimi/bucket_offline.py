"""How much of the archive a bucket bound can skip, measured on dumped snapshots.

    python mexp/kimi/bucket_offline.py --dir results/kimi/caldump [--buckets 64,256,1024]
                                       [--schemes rho,sk1,sk1rho,sign,kmeans] [--device cpu]

Exchangeability makes the archive a static point set and each decode step's
trigger a maximum-inner-product query against it, so the per-row Cauchy-Schwarz
certificate has a per-bucket form: for a bucket C with sidecar centroid s_bar
and radius R_s, sketch centroid c_bar and radius R, and largest residual
rho_max,

  score(t, u) <= lam (q_r . s_bar + |q_r| R_s)
               + lam (q_sk . c_bar + |q_sk| R)
               + z lam |q_perp| rho_max / sqrt(d_c - r)   for every u in C,

so a bucket whose bound stays below the query's best kept score can be skipped
whole, and the fired set is unchanged. This measures, per snapshot and per
calibration query, what fraction of archived rows survive that test (the rows
the scan would still have to read), and asserts soundness: every row the linear
scan fires lies in a surviving bucket.

The decision this feeds: the scan is the largest VestigeKV kernel, grows
linearly with the archive and is bandwidth-bound, so it is worth restructuring
only if the surviving fraction is small.
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


def assign(scheme, B, csk, rho, gen):
    """Bucket id per archived row. Every scheme is a function of the row alone,
    so assignment is stable for the row's lifetime (the ranking is frozen at
    capture); none of them reorders anything the attention reads."""
    A = csk.shape[0]
    if scheme == "rho":
        return torch.bucketize(rho, torch.quantile(rho, torch.linspace(0, 1, B + 1, device=rho.device)[1:-1]))
    if scheme == "sk1":
        x = csk[:, 0]
        return torch.bucketize(x, torch.quantile(x, torch.linspace(0, 1, B + 1, device=x.device)[1:-1]))
    if scheme == "sk1rho":
        b = int(B**0.5)
        a1 = assign("sk1", b, csk, rho, gen)
        a2 = assign("rho", B // b, csk, rho, gen)
        return a1 * (B // b) + a2
    if scheme == "sign":
        bits = max(1, int(round(torch.log2(torch.tensor(float(B))).item())))
        P = torch.randn(csk.shape[1], bits, generator=gen, device=csk.device)
        code = ((csk @ P) > 0).long()
        return (code * (2 ** torch.arange(bits, device=csk.device))).sum(-1)
    if scheme == "kmeans":
        # Lloyd on the sketch coordinates: the only scheme here that shrinks a
        # ball radius, since a quantile split on one coordinate leaves the other
        # r-1 unconstrained and its radius does not fall with B.
        idx = torch.randperm(A, generator=gen, device=csk.device)[:B]
        cent = csk[idx].clone()
        lab = torch.zeros(A, dtype=torch.long, device=csk.device)
        for _ in range(8):
            lab = torch.cdist(csk, cent).argmin(1)
            cnt = torch.bincount(lab, minlength=B).clamp_min(1)[:, None].float()
            cent = torch.zeros_like(cent).index_add_(0, lab, csk) / cnt
        return lab
    raise ValueError(scheme)


def survivors(lab, B, side, csk, rho, qside, qsk, qperp, max1, sc, zp, denom):
    """Fraction of archived rows inside buckets whose bound clears max1, and the
    per-row fired mask of the linear scan (for the soundness assertion)."""
    idx = (lab.clamp(0, B - 1)).to(torch.int64)
    counts = torch.bincount(idx, minlength=B).clamp_min(1)
    s_bar = torch.zeros(B, side.shape[1], device=side.device).index_add_(0, idx, side) / counts[:, None]
    c_bar = torch.zeros(B, csk.shape[1], device=csk.device).index_add_(0, idx, csk) / counts[:, None]
    R_s = torch.zeros(B, device=side.device).index_reduce_(
        0, idx, (side - s_bar[idx]).norm(dim=-1), "amax", include_self=False
    )
    R = torch.zeros(B, device=csk.device).index_reduce_(
        0, idx, (csk - c_bar[idx]).norm(dim=-1), "amax", include_self=False
    )
    rho_max = torch.zeros(B, device=rho.device).index_reduce_(0, idx, rho, "amax", include_self=False)
    # per query: bucket bound, and the exact per-row certified score
    bound = (
        (qside @ s_bar.T + qside.norm(dim=-1)[:, None] * R_s[None, :]) * sc
        + (qsk @ c_bar.T + qsk.norm(dim=-1)[:, None] * R[None, :]) * sc
        + zp * (qperp[:, None] * rho_max[None, :]) * sc / denom
    )
    live = bound > max1[:, None]  # [nq, B]
    read = (live.float() * torch.bincount(idx, minlength=B)[None, :].float()).sum(-1) / side.shape[0]
    score = (qside @ side.T + qsk @ csk.T) * sc + zp * (qperp[:, None] * rho[None, :]) * sc / denom
    fired = score > max1[:, None]
    sound = bool((fired & ~live.gather(1, idx.expand(fired.shape[0], -1))).sum() == 0)
    return read, fired.sum(-1).float(), sound


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "results", "kimi", "caldump"))
    ap.add_argument("--buckets", default="64,256,1024")
    ap.add_argument("--schemes", default="rho,sk1,sk1rho,sign,kmeans")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=4, help="snapshots to read")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.dir, "cal_*.pt")))[: args.limit]
    if not files:
        raise SystemExit(f"no cal_*.pt under {args.dir}")
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    gen = torch.Generator(device=args.device).manual_seed(args.seed)
    agg = {}
    for f in files:
        snap = torch.load(f, map_location=args.device)
        kv = snap["geom"]["kv_lora_rank"]
        rows = snap["rows"].float().to(args.device)
        sc = float(snap["scale"])
        V = snap["V"].float().to(args.device)
        r = V.shape[0]
        zp = float(snap["stats"]["zp"])
        keep = torch.isin(snap["row_slots"].to(args.device), snap["kept"].to(args.device))
        qe = snap["qcal"].reshape(-1, rows.shape[1]).float().to(args.device)
        content, side = rows[:, :kv], rows[:, kv:]
        csk_all = content @ V.T
        rho_all = (content - csk_all @ V).norm(dim=-1)
        arch = (~keep).nonzero().flatten()
        csk, rho, sided = csk_all[arch], rho_all[arch], side[arch]
        qsk = qe[:, :kv] @ V.T
        qperp = (qe[:, :kv] - qsk @ V).norm(dim=-1)
        max1 = ((qe @ rows[keep].T) * sc).max(-1).values
        denom = (kv - r) ** 0.5
        print(f"== {os.path.basename(f)} archive {arch.numel()} queries {qe.shape[0]} zp {zp:.2f}")
        for scheme in args.schemes.split(","):
            for B in (int(b) for b in args.buckets.split(",")):
                lab = assign(scheme, B, csk, rho, gen)
                read, fired, sound = survivors(
                    lab, B, sided, csk, rho, qe[:, kv:], qsk, qperp, max1, sc, zp, denom
                )
                key = (scheme, B)
                a = agg.setdefault(key, [])
                a.append((float(read.mean()), float(read.median()), float(fired.mean()), sound))
                print(f"   {scheme:8s} B={B:5d} rows read mean {read.mean():6.1%} median {read.median():6.1%} "
                      f"fired/query {fired.mean():7.1f} sound={sound}")
    print("== mean over snapshots (rows the scan still reads)")
    print(f"   {'scheme':8s} {'B':>6s} {'mean':>8s} {'median':>8s} {'fired':>8s}  sound")
    for (scheme, B), v in agg.items():
        n = len(v)
        print(f"   {scheme:8s} {B:6d} {sum(x[0] for x in v) / n:8.1%} {sum(x[1] for x in v) / n:8.1%} "
              f"{sum(x[2] for x in v) / n:8.1f}  {all(x[3] for x in v)}")


if __name__ == "__main__":
    main()
