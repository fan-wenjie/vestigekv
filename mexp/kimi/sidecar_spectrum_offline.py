#!/usr/bin/env python3
"""Does tier-1's low-pass miss periodic structure the real sidecar carries?

    python mexp/kimi/sidecar_spectrum_offline.py results/kimi/sidecardump

THE QUESTION, AND WHAT IS ALREADY SETTLED. A pre-review reader asked whether
packing feature dimensions into complex numbers and low-passing the complex
spectrum would beat the real low-pass, by analogy with RoPE. Two thirds of
that needs no data. The branch is un-roped -- SIDECAR_DIM is "the un-roped
half of the latent row", which is the paper's own NoPE argument -- so there is
no rotation to exploit. And a symmetric complex low-pass IS the current
per-channel real one: lowpass(a+ib) = lowpass(a) + i lowpass(b), identical to
8.9e-16 and unchanged when the pairing is permuted, which is the lock -- a
transform that used the pairing could not be indifferent to which dimension
pairs with which.

What no algebra settles is the question underneath: sigma keeps the residual
after projecting a 4096-row block onto 2*kappa-1 trigonometric components
along the TOKEN axis. If the real sidecar carries energy at frequencies above
that cutoff -- periodic structure, whatever its origin -- then the residual is
partly that periodicity rather than the per-row anomaly sigma is read as. This
script measures it on dumped rows.

THREE MEASUREMENTS, in increasing order of what they decide:

  spectrum     energy per frequency bin along the token axis, and the share
               above the kappa cutoff. High share is not itself a defect: a
               residual is supposed to hold what the low-pass drops. It is a
               defect only if that share is PERIODIC rather than broadband,
               because a peak is shared across rows and cannot separate them.
  peakiness    the largest single bin above the cutoff against the median bin
               there. Broadband residual: near 1. A standing peak: large.
  ranking      what actually matters. sigma's top-m against the rows the
               request's own calibration queries score highest. If a periodic
               component is inflating sigma, the overlap falls when that
               component is notched out; if it is not, notching changes
               nothing and the question is answered the other way.

The dump is whole 576-dim pool rows, so the sidecar is rows[:, 512:576]; the
geometry travels in the record rather than being assumed here.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys


def load(path):
    import torch

    blob = torch.load(path, map_location="cpu", weights_only=False)
    geom = blob["geom"]
    rows, slots = blob["rows"], blob["row_slots"]
    # A time-axis transform needs the rows in time order. Pool slots are an
    # allocator's business, not the sequence's, so this is checked rather than
    # assumed: a shuffled block would make every spectrum below meaningless
    # while looking entirely normal.
    if not bool((slots[1:] > slots[:-1]).all()):
        raise SystemExit(
            f"ABORT: {os.path.basename(path)} has non-monotone row_slots, so the "
            "pool order is not the sequence order and the token axis is not an "
            "axis. Dump the positions alongside if this is expected.")
    side = rows[:, geom["kv_lora_rank"]:].float()
    return blob, side


def spectrum(side, kappa, block):
    import torch

    n = side.shape[0] // block
    if n == 0:
        raise SystemExit(f"ABORT: {side.shape[0]} rows is under one {block}-row block")
    out = []
    for b in range(n):
        seg = side[b * block:(b + 1) * block]
        f = torch.fft.rfft(seg - seg.mean(0, keepdim=True), dim=0).abs().pow(2).sum(1)
        out.append(f)
    return torch.stack(out).mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", help="directory of cal_*.pt")
    ap.add_argument("--kappa", type=int, default=16)
    ap.add_argument("--block", type=int, default=4096)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.dump, "cal_*.pt")))
    if not files:
        raise SystemExit(f"ABORT: no cal_*.pt under {a.dump}; run sidecardump-vestigekv")
    print(f"{len(files)} snapshots under {a.dump}\n")
    print(f"{'file':44}{'rows':>7}{'>kappa share':>14}{'peakiness':>11}")
    for path in files:
        blob, side = load(path)
        e = spectrum(side, a.kappa, a.block)
        lo, hi = e[:a.kappa].sum(), e[a.kappa:].sum()
        above = e[a.kappa:]
        peak = float(above.max() / above.median()) if above.numel() else float("nan")
        print(f"{os.path.basename(path)[:42]:44}{side.shape[0]:>7}"
              f"{float(hi / (lo + hi)):>13.1%}{peak:>11.1f}")
    print("\npeakiness near 1 means the residual above the cutoff is broadband --")
    print("nothing periodic for the low-pass to have missed. A large value means a")
    print("standing component shared across rows, which inflates every sigma alike")
    print("and so cannot separate them; that is the case worth acting on.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
