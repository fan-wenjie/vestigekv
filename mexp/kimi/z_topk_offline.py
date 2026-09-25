"""What z has to be to certify the top-k archived rows, not just the best one.

    python mexp/kimi/z_topk_offline.py --dir results/kimi/caldump [--ks 1,2,3,4,8]

The conformal calibration solves, per calibration query, for the z that makes
the certified score of its **best** archived row reach that row's true score:

    z_req = (true_best - idxs_best) / cert_best

so the guarantee it buys is marginal and about one row. A multi-key question
needs k distinct archived rows at once, and a marginal guarantee does not
compose: at recall target tau it gives roughly tau^k. The measured RULER means
fit that shape -- multikey_2 0.920 and multikey_3 0.860 back out a consistent
per-key 0.959 and 0.951 -- while every single-needle task sits at 1.000.

This measures the cost of fixing it at the source: take the max of z_req over
the top-k archived rows instead of the best one, so the quantile certifies all
k. It reports, per k, the calibrated z and how many rows the scan then fires,
which is what the change trades.

Reads the same snapshots as bucket_offline.py and reuses its decomposition:
content rows project through the rank-r basis V into a sketch, the residual
norm is rho, and the certificate term is z * |q_perp| * rho / sqrt(kv - r).
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "results", "kimi", "caldump"))
    ap.add_argument("--ks", default="1,2,3,4,8")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--target", type=float, default=D.RECALL_TARGET)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.dir, "cal_*.pt")))[: args.limit]
    if not files:
        raise SystemExit(f"no cal_*.pt under {args.dir}")
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    ks = [int(x) for x in args.ks.split(",")]

    for f in files:
        snap = torch.load(f, map_location=args.device, weights_only=False)
        kv = snap["geom"]["kv_lora_rank"]
        rows = snap["rows"].float()
        sc = float(snap["scale"])
        V = snap["V"].float()
        r = V.shape[0]
        keep = torch.isin(snap["row_slots"], snap["kept"])
        qe = snap["qcal"].reshape(-1, rows.shape[1]).float()
        content, side = rows[:, :kv], rows[:, kv:]
        csk_all = content @ V.T
        rho_all = (content - csk_all @ V).norm(dim=-1)
        arch = (~keep).nonzero().flatten()
        csk, rho, sided = csk_all[arch], rho_all[arch], side[arch]
        qsk = qe[:, :kv] @ V.T
        qperp = (qe[:, :kv] - qsk @ V).norm(dim=-1)
        denom = (kv - r) ** 0.5
        max1 = ((qe @ rows[keep].T) * sc).max(-1).values

        # per (query, archived row): the true score, the index score the scan
        # can see, and the per-unit-z certificate slack
        true = (qe @ rows[arch].T) * sc
        idxs = (qe[:, kv:] @ sided.T + qsk @ csk.T) * sc
        cert = (qperp[:, None] * rho[None, :]) * sc / denom
        need = (true - idxs) / cert.clamp_min(1e-9)  # z that certifies that row

        nq, A = true.shape
        print(f"== {os.path.basename(f)}  archive {A}  queries {nq}  rank {r}")
        # The set that actually matters: a row only changes the output if its
        # true score beats the best kept row. Its size is data-determined per
        # query, not a chosen k, and certifying exactly it is the target the
        # guarantee should have.
        beats = true > max1[:, None]
        nb = beats.sum(1).float()
        zb = torch.where(beats, need, torch.full_like(need, -1e30)).max(1).values
        zb = zb[nb > 0]
        jb = D.conformal_k(max(zb.numel(), 1), args.target)
        z_beat = float(zb.sort().values[min(jb, max(zb.numel(), 1) - 1)]) if zb.numel() else 0.0
        fired_b = ((idxs + z_beat * cert) > max1[:, None]).sum(-1).float()
        print(f"   beats max1: count p50 {nb.median():.0f} p90 {nb.quantile(0.9):.0f} "
              f"max {nb.max():.0f} | queries with none {(nb==0).float().mean():.0%}")
        print(f"   {'target':>10s} {'z':>8s} {'fired p50':>10s} {'p90':>8s}")
        print(f"   {'beats max1':>10s} {z_beat:8.3f} {fired_b.median():10.0f} {fired_b.quantile(0.9):8.0f}")
        order = true.argsort(dim=1, descending=True)
        for k in ks:
            topk = order[:, :k]
            # the query is satisfied only when every one of its top-k archived
            # rows is certified, so its requirement is the largest of the k
            zq = need.gather(1, topk).max(dim=1).values
            j = D.conformal_k(nq, args.target)
            z = float(zq.sort().values[min(j, nq - 1)])
            fired = ((idxs + z * cert) > max1[:, None]).sum(-1).float()
            print(f"   {('top-%d' % k):>10s} {z:8.3f} {fired.median():10.0f} {fired.quantile(0.9):8.0f}")


if __name__ == "__main__":
    main()
