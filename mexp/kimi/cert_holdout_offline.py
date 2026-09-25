"""Does the certificate still recover the rows it needs OUT OF SAMPLE?

    python mexp/kimi/cert_holdout_offline.py [--dir results/kimi/caldump]

`ent_margin_offline.py` and `z_topk_offline.py` both evaluate the fitted zp on
the very queries zp was fitted on. In sample a conformal quantile recovers its
target by construction -- that is what the order statistic is -- so "1081 of
1081 rows recovered" says nothing about a decode step the fit never saw. Every
conclusion drawn from that number inherits the error, including the ruling that
the recall step is not where multi-key rows are lost.

This splits the calibration points, fits zp on one half with EXACTLY the
delivered rule (recall_tier.py: z_req = (abest - idxs_t)/cert_t over queries
with an archived row, then the conformal_k-th order statistic, clamped at
Z_MAX), and scores the other half. Two quantities, because they answer
different questions:

  row recall     of the archived rows whose true score beats the kept maximum
                 -- the only rows that reach the output -- how many fire.
  query recall   of the queries needing at least one such row, how many get
                 ALL of them. A k-key question needs every one, so this is the
                 quantity the multi-key gap is made of, and a marginal
                 per-row guarantee does not deliver it.

Both are reported in sample and out, so the gap between them IS the in-sample
optimism. Split modes: `random` (the exchangeability the fit assumes) and
`pos` (earliest calibration positions fit, latest score) -- though note the
dumped queries are 8 CONSECUTIVE decode steps, so `pos` tests a two-token
extrapolation, not a long one.
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


def fit_zp(need_z, target):
    """The delivered rule, on whatever subset it is handed."""
    n = need_z.numel()
    if n == 0:
        return D.Z_MAX
    k = D.conformal_k(n, target)
    return min(float(need_z.sort().values[min(k, n) - 1]), D.Z_MAX)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "results", "kimi", "caldump"))
    ap.add_argument("--split", default="random", choices=("random", "pos"))
    ap.add_argument("--target", type=float, default=D.RECALL_TARGET)
    ap.add_argument("--limit", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.dir, "cal_*.pt")))[: args.limit]
    if not files:
        raise SystemExit(f"no cal_*.pt under {args.dir}")
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    g = torch.Generator().manual_seed(args.seed)

    # [in-sample, out-of-sample] x [rows fired, rows needed, queries all-fired,
    # queries needing]; plus the two zp values, to show what the fit itself does.
    acc = {s: [0, 0, 0, 0] for s in ("in", "out")}
    zs = {"in": [], "out": []}
    by_k = {}  # needed-row count -> [out-of-sample queries satisfied, queries]

    for f in files:
        snap = torch.load(f, map_location=args.device, weights_only=False)
        kv = snap["geom"]["kv_lora_rank"]
        rows = snap["rows"].float()
        sc = float(snap["scale"])
        V = snap["V"].float()
        r = V.shape[0]
        keep = torch.isin(snap["row_slots"], snap["kept"])
        nq, H, dim = snap["qcal"].shape
        qe = snap["qcal"].reshape(-1, dim).float()
        content, side = rows[:, :kv], rows[:, kv:]
        csk_all = content @ V.T
        rho_all = (content - csk_all @ V).norm(dim=-1)
        arch = (~keep).nonzero().flatten()
        if arch.numel() == 0:
            continue
        csk, rho, sided = csk_all[arch], rho_all[arch], side[arch]
        qsk = qe[:, :kv] @ V.T
        qperp = (qe[:, :kv] - qsk @ V).norm(dim=-1)

        max1 = ((qe @ rows[keep].T) * sc).max(-1).values
        true = (qe @ rows[arch].T) * sc
        idxs = (qe[:, kv:] @ sided.T + qsk @ csk.T) * sc
        cert = ((qperp[:, None] * rho[None, :]) * sc / (kv - r) ** 0.5).clamp_min(
            D.ENTROPY_EPS
        )

        # The delivered target: the single BEST archived row per query.
        abest, atgt = true.max(-1)
        z_req = (abest - idxs.gather(1, atgt[:, None]).squeeze(1)) / cert.gather(
            1, atgt[:, None]
        ).squeeze(1)

        # What actually reaches the output, and so what recall is judged on.
        needed = true > max1[:, None]
        nneed = needed.sum(1)

        # Split the (query, head) points. Heads of one query are not
        # independent, so `pos` splits whole queries.
        idx = torch.arange(qe.shape[0])
        if args.split == "pos":
            cut = nq // 2
            fit_mask = (idx // H) < cut
        else:
            perm = torch.randperm(qe.shape[0], generator=g)
            fit_mask = torch.zeros(qe.shape[0], dtype=torch.bool)
            fit_mask[perm[: qe.shape[0] // 2]] = True
        hold = ~fit_mask

        for tag, zp, ev in (
            ("in", fit_zp(z_req, args.target), torch.ones_like(fit_mask)),
            ("out", fit_zp(z_req[fit_mask], args.target), hold),
        ):
            zs[tag].append(zp)
            fired = (idxs + zp * cert) > max1[:, None]
            hit = (fired & needed)[ev]
            nd = needed[ev]
            acc[tag][0] += int(hit.sum())
            acc[tag][1] += int(nd.sum())
            has = nneed[ev] > 0
            allfired = (hit.sum(1) == nd.sum(1))[has]
            acc[tag][2] += int(allfired.sum())
            acc[tag][3] += int(has.sum())
            if tag == "out":
                for kk, ok in zip(nneed[ev][has].tolist(), allfired.tolist()):
                    b = by_k.setdefault(min(kk, 8), [0, 0])
                    b[0] += int(ok)
                    b[1] += 1

    print(f"split={args.split}  target={args.target}  snapshots={len(files)}")
    print(f"  zp  in-sample {sum(zs['in'])/len(zs['in']):.3f}   "
          f"fit-half {sum(zs['out'])/len(zs['out']):.3f}")
    print(f"\n{'':>4s} {'row recall':>22s} {'query recall (ALL rows)':>28s}")
    for tag in ("in", "out"):
        rf, rn, qf, qn = acc[tag]
        print(f"{tag:>4s} {rf:8d}/{rn:<8d} {rf/max(rn,1):6.1%} "
              f"{qf:10d}/{qn:<8d} {qf/max(qn,1):6.1%}")
    print(f"\nout-of-sample query recall by how many rows the query needs")
    print(f"{'k':>3s} {'satisfied':>12s} {'rate':>8s}")
    for k in sorted(by_k):
        ok, n = by_k[k]
        print(f"{k:3d} {ok:6d}/{n:<6d} {ok/max(n,1):8.1%}")


if __name__ == "__main__":
    main()
