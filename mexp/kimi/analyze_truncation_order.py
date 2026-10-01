#!/usr/bin/env python3
"""Positional-prefix truncation against bound-ordered truncation, on r=0.

    python mexp/kimi/analyze_truncation_order.py [--dir results/kimi/sigmadump/prose]

When a lane fires more rows than the fetch buffer holds, the compaction "keeps
the first W in position order" (fused_prologue.compact_fired) and raises the
overflow flag that drives the fence. So the COST is already capped at W; what is
arbitrary is WHICH rows survive -- archive position has nothing to do with how
likely a row is to beat the kept maximum.

The branch arm is where this bites, because it is the arm that over-fires: it
fires 979 rows a step on average against the served arm's 36.5, with a median of
12 and a quarter of its steps above 2048. A flat negative margin tightens every
step including that cheap median, which is where its fidelity cost comes from
(first divergence 30 -> 7). Re-ordering the truncation touches only the steps
that overflow.

This decides whether that is worth a kernel change, without one. For each
calibration query it reconstructs the r=0 certificate exactly -- V is zero, so
q_res is the whole content query and rho_u is the whole content norm -- and
compares, at a given capacity W:

  positional: the first W fired rows by archive position
  by bound:   the W fired rows with the largest idxs + zp*cert

scored on how many rows that TRULY beat the kept maximum survive each. If
bound order keeps materially more of them at the same W, the truncation is
losing recall for free and the fix is an ordering, not a threshold.

Caveat kept in view: qcal is the prompt's last few positions, which are out of
distribution for answer steps (0.1% against 22.85% miss). That biases the
absolute recall numbers, not the comparison between two orderings of the same
fired set.
"""
from __future__ import annotations

import argparse
import glob
import os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/kimi/sigmadump/prose")
    ap.add_argument("--zp", type=float, default=8.0)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--caps", default="512,1024,2048")
    ap.add_argument("--device", default="cpu",
                    help="cpu by default ON PURPOSE: this reads dumps, and a "
                         "CUDA run here contends with whatever the queue is "
                         "serving -- it OOMed a live server once already")
    ap.add_argument("--positions", type=int, default=8,
                    help="calibration positions per file (8 = all)")
    ap.add_argument("--files", type=int, default=3)
    a = ap.parse_args()
    caps = [int(x) for x in a.caps.split(",")]
    dev = a.device

    print(f"  r=0 certificate, zp={a.zp}, margin={a.margin}, device={dev}")

    tot = {w: [0, 0, 0, 0] for w in caps}  # beat, pos, bound, last
    for p in sorted(glob.glob(os.path.join(a.dir, "cal_*.pt")))[: a.files]:
        d = torch.load(p, map_location="cpu")
        rows = d["rows"].to(dev).float()            # [T, 576]
        # `kept` holds POOL ROW IDS; row_slots maps position -> pool row id,
        # so positions come from a searchsorted, not from kept directly. Using
        # it as an index into `rows` is an out-of-bounds read.
        rs = d["row_slots"].contiguous()
        kept = torch.searchsorted(rs, d["kept"].contiguous()).to(dev).long()
        kept = kept.clamp_(0, rows.shape[0] - 1)
        sc = float(d["scale"])
        kv = rows.shape[1] - 64
        q = d["qcal"].to(dev).float().reshape(-1, rows.shape[1])   # [n*H, 576]
        T = rows.shape[0]
        arch_mask = torch.ones(T, dtype=torch.bool, device=dev)
        arch_mask[kept] = False
        arch = arch_mask.nonzero(as_tuple=True)[0]
        if arch.numel() == 0:
            continue
        ar = rows.index_select(0, arch)
        kr = rows.index_select(0, kept)
        # r=0: idxs is the sidecar term only; the certificate is the whole
        # content block bounded by Cauchy-Schwarz, scaled by zp/sqrt(kv).
        rho = ar[:, :kv].norm(dim=-1)                               # [A]
        side_a = ar[:, kv:]                                         # [A, 64]
        name = os.path.basename(p)
        for qi in range(0, min(a.positions, 8) * 16, 16):   # one position at a time
            qq = q[qi:qi + 16]
            if qq.numel() == 0:
                continue
            qres = qq[:, :kv].norm(dim=-1)                          # [H]
            idxs = (qq[:, kv:] @ side_a.T) * sc                     # [H, A]
            cert = (qres[:, None] * rho[None, :]) * sc / (kv ** 0.5)
            bound = idxs + a.zp * cert
            true_a = (qq @ ar.T) * sc
            max1 = ((qq @ kr.T) * sc).max(-1).values                # [H]
            thr = max1 - a.margin
            fired = bound > thr[:, None]
            beat = true_a > max1[:, None]
            for w in caps:
                for h in range(qq.shape[0]):
                    fi = fired[h].nonzero(as_tuple=True)[0]
                    if fi.numel() <= w:
                        continue                 # no truncation, nothing to compare
                    b = beat[h]
                    n_beat = int(b.sum())
                    if n_beat == 0:
                        continue
                    pos_keep = fi[:w]                               # first W
                    last_keep = fi[-w:]                             # last W
                    ord_b = bound[h, fi].argsort(descending=True)[:w]
                    bnd_keep = fi[ord_b]                            # top W by bound
                    tot[w][0] += n_beat
                    tot[w][1] += int(b[pos_keep].sum())
                    tot[w][2] += int(b[bnd_keep].sum())
                    tot[w][3] += int(b[last_keep].sum())
        pass
        del rows, ar, kr
    print("\n  TOTAL over all dumps and heads that actually overflowed:")
    print(f"    {'W':<6} {'beat':>5} {'first W':>9} {'last W':>9} {'top W by bound':>15}")
    for w in caps:
        tb, sp, sb, sl = tot[w]
        if not tb:
            print(f"    W={w:<5} nothing truncated")
            continue
        print(f"    {w:<6} {tb:>5} {100*sp/tb:>8.1f}% {100*sl/tb:>8.1f}% "
              f"{100*sb/tb:>14.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
