#!/usr/bin/env python3
"""What the certificate's slack is made of: rank, norm pair, and bucket spread.

    python mexp/kimi/analyze_geometry.py --dir results/kimi/sigmadump/mrcr

Three questions off one dump, none of which needs a GPU job of its own. The
calibration dump carries the pool rows, the calibration queries and the fitted
basis, so any rank and any norm can be refitted offline.

THE IDENTITY THIS RESTS ON. With V's rows orthonormal the content splits as
c = V'Vc + c_res and the query as q = V'Vq + q_res, the cross terms vanish, and

    q.c - (Vq).(Vc) = q_res . c_res

exactly. So the certificate's error is a single inner product and its bound
is Cauchy-Schwarz on it: |q_res . c_res| <= ||q_res|| rho. The bound's
looseness is therefore ONE number -- the cosine between the two residuals --
and everything below is a way of measuring it.

  1. RANK. rho, the slack ||q_res||*rho, and that cosine, at r = 0, 4, 8, 16,
     32, 64, 128. r=0 is the branch arm: nothing projected, rho is the whole
     content norm. The measured claim this tests is that branch-only is loose
     by roughly 1/sqrt(residual energy fraction) per side, which the rotation
     ablation puts at 0.8-9.0% for rank 64 -- so 3.3-11x each, and the product
     on the slack.

  2. NORM PAIR. Cauchy-Schwarz is Hoelder at p=q=2, and the other pairs are
     equally sound. Which is tightest on THIS data is not a matter of opinion;
     it is |q.c| against each of the three products. L2 also buys Pythagoras
     (the prologue's back-projection GEMM disappears), so a tighter pair has
     to beat that too, and this prints what it would have to beat.

  3. BUCKET SPREAD. The scan reads 1024-row buckets. If a bucket's largest
     ||s_u|| and largest rho are far above its typical ones, a per-bucket bound
     prunes nothing; if buckets are homogeneous and differ from each other,
     whole buckets can be skipped without reading a row. That is the
     precondition for making the scan sublinear, and it is a property of the
     data, so it is measured before any kernel is written.
"""
from __future__ import annotations

import argparse
import glob
import os

import torch


def fit_basis(qcal, kv, r):
    """The basis build() fits: top-r eigenvectors of the centred query gram."""
    if r == 0:
        return torch.zeros(0, kv, dtype=torch.float64)
    qe = qcal.reshape(-1, qcal.shape[-1]).double()[:, :kv]
    qc = qe - qe.mean(0, keepdim=True)
    _, evecs = torch.linalg.eigh(qc.T @ qc)
    return evecs[:, -r:].T.flip(0)


def stats(v):
    s = v.sort().values
    n = s.numel()
    g = lambda p: float(s[min(n - 1, int(p * n))])
    return g(0.5), g(0.9), float(s[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ranks", default="0,4,8,16,32,64,128")
    ap.add_argument("--bucket", type=int, default=1024)
    ap.add_argument("--max-rows", type=int, default=32768)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.dir, "cal_*.pt")))
    if not files:
        raise SystemExit(f"ABORT: no cal_*.pt under {a.dir}; the dump did not run")
    d = torch.load(files[0], map_location="cpu", weights_only=False)
    kv = d["geom"]["kv_lora_rank"]
    sd = d["geom"]["side_dim"]
    rows = d["rows"].double()
    if rows.shape[0] > a.max_rows:
        rows = rows[:: rows.shape[0] // a.max_rows][: a.max_rows]
    content, side = rows[:, :kv], rows[:, kv : kv + sd]
    qcal = d["qcal"]
    q = qcal.reshape(-1, qcal.shape[-1]).double()
    qc = q[:, :kv]
    print(f"{os.path.basename(files[0])}: {len(files)} dumps, "
          f"{rows.shape[0]} rows x {kv}+{sd}, {qc.shape[0]} calibration queries, "
          f"index_rank={d['index_rank']}")

    # --- 1. rank sweep -------------------------------------------------
    true = (qc @ content.T).abs()          # [nq, T] the exact content term
    print(f"\n{'rank':>5} {'resid energy':>13} {'rho p50':>9} {'slack p50':>11} "
          f"{'|cos| p50':>10} {'slack/|q.c| p50':>16}")
    for r in (int(x) for x in a.ranks.split(",")):
        if r > kv:
            continue
        V = fit_basis(qcal, kv, r)
        if r:
            cres = content - (content @ V.T) @ V
            qres = qc - (qc @ V.T) @ V
        else:
            cres, qres = content, qc
        rho = cres.norm(dim=-1)
        qn = qres.norm(dim=-1)
        frac = float((rho**2).sum() / (content**2).sum())
        slack = qn[:, None] * rho[None, :]
        err = (qres @ cres.T).abs()
        cos = (err / slack.clamp_min(1e-30))
        print(f"{r:>5} {100 * frac:>12.2f}% {stats(rho)[0]:>9.3f} "
              f"{stats(slack.flatten())[0]:>11.3f} {stats(cos.flatten())[0]:>10.4f} "
              f"{stats((slack / true.clamp_min(1e-30)).flatten())[0]:>16.2f}")

    # --- 2. Hoelder pairs, at r=0 (the branch arm's case) ---------------
    print(f"\nnorm pair at r=0, ratio bound/|q.c| (1.0 would be exact)")
    q2, qi, q1 = qc.norm(dim=-1), qc.abs().amax(-1), qc.abs().sum(-1)
    c2, ci, c1 = content.norm(dim=-1), content.abs().amax(-1), content.abs().sum(-1)
    t = true.clamp_min(1e-30)
    for name, b in (("L2 x L2  (current)", q2[:, None] * c2[None, :]),
                    ("Linf x L1", qi[:, None] * c1[None, :]),
                    ("L1 x Linf", q1[:, None] * ci[None, :])):
        p50, p90, mx = stats((b / t).flatten())
        print(f"  {name:<20} p50 {p50:>9.2f}   p90 {p90:>9.2f}   max {mx:>10.2f}")

    # --- 3. bucket spread ----------------------------------------------
    V = fit_basis(qcal, kv, d["index_rank"])
    rho = (content - (content @ V.T) @ V).norm(dim=-1)
    sn = side.norm(dim=-1)
    nb = max(1, rows.shape[0] // a.bucket)
    print(f"\nbucket spread, {nb} buckets of {a.bucket} rows (r={d['index_rank']})")
    for nm, v in (("||side||", sn), ("rho", rho)):
        b = v[: nb * a.bucket].reshape(nb, -1)
        within = (b.amax(1) / b.median(1).values.clamp_min(1e-30))
        across = b.amax(1)
        print(f"  {nm:<10} max/median WITHIN a bucket p50 {float(within.median()):.2f}; "
              f"bucket maxima ACROSS buckets p50 {float(across.median()):.3f} "
              f"min {float(across.min()):.3f} max {float(across.max()):.3f}")
    print("\n  A bucket bound prunes only where the across-bucket spread of the "
          "maxima exceeds the within-bucket spread; equal spreads mean every "
          "bucket looks alike from outside and nothing is skippable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
