#!/usr/bin/env python3
"""Redraw both serving figures from the same logs the macros are read from.

    ~/.venvs/plot/bin/python mexp/kimi/make_figures.py --emit

WHY THIS EXISTS. The captions cite generated macros and the panels were drawn
by hand, so the two could drift apart without anything noticing -- and they
did: the figures in the submission directory were five days older than every
run they claimed to show, produced before the GRAPH_BS correction moved the
256k ratio from 1.236 to 1.269 and the 512k one from 1.368 to 1.446. A reader
comparing a plotted point against the number beside it would have found them
disagreeing, which is the worst way for a reader to find out.

So the panels are computed here by importing the two macro generators and
calling the same parsers they use -- not by re-implementing them. If a panel
and its caption ever disagree again it will be because a generator changed,
and both move together.

The plotting environment is deliberately NOT the serving environment:
matplotlib lives in ~/.venvs/plot so that installing it can never perturb an
in-flight run (see the note in docs/ about not touching sglang-dev while the
queue is live).
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PAPER = os.path.expanduser("~/vestigekv_paper")
sys.path.insert(0, HERE)

import make_serving_speed_numbers as speed  # noqa: E402
import make_throughput_numbers as tput  # noqa: E402

# The two generators key their arms differently -- the latency one by the
# display name, the throughput one by the launch name -- so the style table is
# keyed by neither and each panel says which key it is reading. Guessing that
# they agree is how the right panel came out with a KeyError the first time.
ARM_STYLE = {"dense": ("#444444", "o", "Dense MLA"),
             "vestigekv": ("#1b6ca8", "s", "VestigeKV")}
LAT_KEY = {"dense": "dense", "vestigekv": "vestigekv"}
THR_KEY = {"dense": "baseline", "vestigekv": "vestigekv"}


def latency_series():
    """(context, ms/token) per arm, on the buckets the log actually carries."""
    out = {}
    for arm, name in speed.LOGS.items():
        rows = speed.read(os.path.join(speed.RESULTS, name))
        top = max(n for n, _ in rows)
        # `top` is where the stream actually stopped and is not on the 4096
        # lattice, so it has to be appended rather than stepped to. The macro
        # table's last row IS this point; dropping it would put the curve's
        # end 0.3%/1.8% away from the number printed beside it, which is the
        # exact class of drift this script exists to remove.
        grid = list(range(8192, top, 4096)) + [top]
        pts = [(c, speed.window(rows, c)) for c in grid]
        out[arm] = [(c, v) for c, v in pts if v is not None]
        if not out[arm]:
            raise SystemExit(f"ABORT: {name} yielded no latency bucket")
    return out


def throughput_series():
    """{ctx: {live: {arm: tok/s}}} -- exactly the admitted points, no others."""
    have, refused = tput.survey()
    for r in refused:
        print(f"  (refused, so not plotted) {r}")
    return have


def draw(lat, thr, out_dir, emit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- left: per-token latency against context ----
    fig, ax = plt.subplots(figsize=(4.4, 3.0), dpi=200)
    for arm, (color, marker, label) in ARM_STYLE.items():
        xs = [c / 1024 for c, _ in lat[LAT_KEY[arm]]]
        ys = [v for _, v in lat[LAT_KEY[arm]]]
        ax.plot(xs, ys, color=color, lw=1.4, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks([8, 32, 128, 512])
    ax.set_xticklabels(["8k", "32k", "128k", "512k"])
    ax.set_xlabel("context (tokens)")
    ax.set_ylabel("ms per decoded token")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    p1 = os.path.join(out_dir, "fig_latency_curve.png")
    if emit:
        fig.savefig(p1)
        print(f"wrote {p1}")
    plt.close(fig)

    # ---- right: throughput against the concurrency the server actually ran ----
    fig, ax = plt.subplots(figsize=(4.4, 3.0), dpi=200)
    dash = {65536: "-", 131072: "--"}
    for ctx in sorted(thr):
        if not thr[ctx]:
            continue
        lives = sorted(thr[ctx])
        for arm, (color, marker, label) in ARM_STYLE.items():
            ys = [thr[ctx][b][THR_KEY[arm]] for b in lives]
            ax.plot(lives, ys, dash[ctx], color=color, marker=marker,
                    ms=3.5, lw=1.3,
                    label=f"{label}, {ctx // 1024}k")
    ax.set_xlabel("live requests (measured)")
    ax.set_ylabel("decode throughput (token/s)")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    p2 = os.path.join(out_dir, "fig_throughput.png")
    if emit:
        fig.savefig(p2)
        print(f"wrote {p2}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--out", default=PAPER)
    a = ap.parse_args()

    lat = latency_series()
    for arm in sorted(lat):
        c0, v0 = lat[arm][0]
        c1, v1 = lat[arm][-1]
        print(f"  latency {arm:10} {len(lat[arm])} buckets  "
              f"{c0 // 1024}k {v0:.3f} ms -> {c1 // 1024}k {v1:.3f} ms")
    thr = throughput_series()
    for ctx in sorted(thr):
        print(f"  throughput {ctx // 1024}k: live {sorted(thr[ctx])}")
    draw(lat, thr, a.out, a.emit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
