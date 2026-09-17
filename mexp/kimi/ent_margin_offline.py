"""What gain to give the entropy margin, from the dumped calibration snapshots.

    python mexp/kimi/ent_margin_offline.py [--dir results/kimi/caldump]

The scan fires an archived row when its certified score clears the kept
maximum less a margin. `--vestigekv-entropy-margin-gain` adds margin per nat of
how flat the kept distribution is, measured as the log-sum-exp of the kept
scores minus their maximum: 0 when one kept row holds the attention mass,
log(n) when none does. A query with no dominant kept row is the one whose
maximum says least and the one that needs several archived rows rather than
one.

The gain is not guessed. This measures, per snapshot and per calibration query,
the flatness, and then how many archived rows each candidate gain would fire
and how many of the rows the query actually needs -- those whose true score
beats the kept maximum -- it would recover. The gain to pick is the smallest
that recovers what is recoverable; firing more than that buys nothing and costs
the fallback rate.
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
    ap.add_argument("--gains", default="0,0.25,0.5,1,2")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.dir, "cal_*.pt")))[: args.limit]
    if not files:
        raise SystemExit(f"no cal_*.pt under {args.dir}")
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    gains = [float(g) for g in args.gains.split(",")]

    agg = {g: [0, 0, 0.0] for g in gains}  # recovered, needed, fired
    for f in files:
        snap = torch.load(f, map_location=args.device, weights_only=False)
        kv = snap["geom"]["kv_lora_rank"]
        rows = snap["rows"].float()
        sc = float(snap["scale"])
        V = snap["V"].float()
        r = V.shape[0]
        zp = float(snap["stats"]["zp"])
        keep = torch.isin(snap["row_slots"], snap["kept"])
        qe = snap["qcal"].reshape(-1, rows.shape[1]).float()
        content, side = rows[:, :kv], rows[:, kv:]
        csk_all = content @ V.T
        rho_all = (content - csk_all @ V).norm(dim=-1)
        arch = (~keep).nonzero().flatten()
        csk, rho, sided = csk_all[arch], rho_all[arch], side[arch]
        qsk = qe[:, :kv] @ V.T
        qperp = (qe[:, :kv] - qsk @ V).norm(dim=-1)

        skept = (qe @ rows[keep].T) * sc
        max1 = skept.max(-1).values
        flat = torch.logsumexp(skept, -1) - max1  # the gain's multiplier
        true = (qe @ rows[arch].T) * sc
        idxs = (qe[:, kv:] @ sided.T + qsk @ csk.T) * sc
        cert = (qperp[:, None] * rho[None, :]) * sc / (kv - r) ** 0.5
        certified = idxs + zp * cert
        needed = true > max1[:, None]  # the rows that reach the output

        print(f"== {os.path.basename(f)}  flat p50 {flat.median():.3f} "
              f"p90 {flat.quantile(0.9):.3f} max {flat.max():.3f}  "
              f"needed p90 {needed.sum(1).float().quantile(0.9):.0f}")
        for g in gains:
            thr = (max1 - g * flat)[:, None]
            fired = certified > thr
            rec = int((fired & needed).sum())
            agg[g][0] += rec
            agg[g][1] += int(needed.sum())
            agg[g][2] += float(fired.sum(1).float().median())

    print(f"\n{'gain':>6s} {'needed rows recovered':>22s} {'fired rows p50 (mean)':>22s}")
    for g in gains:
        rec, need, fired = agg[g]
        pct = rec / need if need else float("nan")
        print(f"{g:6.2f} {rec:9d} / {need:<8d} {pct:6.1%} {fired / len(files):22.1f}")


if __name__ == "__main__":
    main()
