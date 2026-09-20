#!/usr/bin/env python3
"""The bs=1 latency curve's macros, reduced from the two server logs.

    python mexp/kimi/make_serving_speed_numbers.py --emit

Until today these five values were hand-maintained, and their trailing
comments named server logs that no surviving tree contains -- the exact
provenance hole audit_provenance.py measures. Worse, the pair they summarised
was never a pair: the vestigekv arm came from td-stream-512k
(MAMBA_SLOTS=8, GRAPH_BS=1, RADIX=on) and the dense arm from a different job at
a different context with RADIX off, so three launch knobs varied alongside the
backend. The ratio that comparison produced, 1.53x at 508k, was measured across
that mismatch, not across the backend.

This script reads the paired run instead: one script, one sitting, both arms
identical apart from the backend, both logs shipped. The ratio it produces is
lower than the hand-maintained one and it is the number the paper prints.

THE METRIC. The server's own `gen throughput (token/s)` lines, median over a
+/-2k-token window around each context. The client-side ITL is not the metric
here and the reason is in stream_itl_is_trustworthy(): past ~64k tokens the
per-token SSE cost can put the client behind the server, and the contamination
penalises whichever arm decodes faster.

THE KNOB THAT WAS WRONG, AND WHAT IT COST. The first attempt at this pair
captured a cuda graph for bs=2 while decoding at bs=1. Both arms carried it, so
the comparison was controlled and the numbers looked plausible; what gave it
away was that the dense arm reproduced two independent earlier records while
the vestigekv arm did not. Re-run at GRAPH_BS=1, dense moved 0.1% -- which is
this measurement's run-to-run noise -- and vestigekv moved 2.6% at 256k and
5.6% at 496k. The surplus lane is not a fixed per-step cost: the scan grid is
baked at capture over max_bs times the context, so an unused lane scans rows in
proportion to how long the context has grown. That is why the error hid at
short contexts and mattered most exactly where the paper makes its claim.
"""
from __future__ import annotations

import argparse
import os
import re
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "results", "kimi")
PAPER = os.path.expanduser("~/vestigekv_paper")
OUT = os.path.join(PAPER, "serving_speed_numbers.tex")

LOGS = {
    "dense": "server_baseline_paired512k-bs1-dense.log",
    "vestigekv": "server_vestigekv_paired512k-bs1-vestigekv.log",
}
HALFWIDTH = 2048
LINE = re.compile(r"#full token:\s*(\d+).*?gen throughput \(token/s\):\s*([\d.]+)")


def read(path):
    rows = []
    for line in open(path, errors="ignore"):
        m = LINE.search(line)
        if m and float(m.group(2)) > 0:
            rows.append((int(m.group(1)), 1000.0 / float(m.group(2))))
    if not rows:
        raise SystemExit(f"ABORT: no gen-throughput lines in {path}")
    return rows


def control_check(paths):
    """Both arms must have been launched with the same knobs. The arm mismatch
    this script replaces is exactly what happens when nothing checks."""
    knobs = ("cuda_graph_max_bs_decode", "max_mamba_cache_size",
             "disable_radix_cache", "context_length", "max_running_requests")
    seen = {}
    for arm, p in paths.items():
        text = open(p, errors="ignore").read(400000)
        seen[arm] = {k: (re.search(rf"'{k}':\s*([^,}}]+)", text) or [None, "?"])[1]
                     for k in knobs}
    a, b = seen["dense"], seen["vestigekv"]
    differing = {k: (a[k], b[k]) for k in knobs if a[k] != b[k]}
    if differing:
        raise SystemExit(
            "ABORT: the two arms were not launched alike, so the ratio would "
            f"measure the launch and not the backend: {differing}")
    return a


def window(rows, ctx):
    w = [ms for n, ms in rows if abs(n - ctx) <= HALFWIDTH]
    return statistics.median(w) if w else None


def slope(rows, lo, hi):
    """ns per decoded step per cached token, over the span the figure plots."""
    a, b = window(rows, lo), window(rows, hi)
    return (b - a) * 1e6 / (hi - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", action="store_true")
    a = ap.parse_args()

    paths = {arm: os.path.join(RESULTS, f) for arm, f in LOGS.items()}
    for p in paths.values():
        if not os.path.exists(p):
            raise SystemExit(f"ABORT: missing server log {p}")
    knobs = control_check(paths)
    arms = {arm: read(p) for arm, p in paths.items()}

    grid = [8192, 16384, 20480, 24576, 28672, 32768, 65536, 131072, 262144, 507904]
    table = []
    for ctx in grid:
        d, v = window(arms["dense"], ctx), window(arms["vestigekv"], ctx)
        if d and v:
            table.append((ctx, d, v, d / v))

    # crossover: the first bucket whose ratio reaches 1.0, reported to the
    # bucket and not interpolated -- the curve is flat here and a fitted
    # crossing would claim a precision the 4k buckets do not have
    cross = next((c for c, _, _, r in table if r >= 1.0), None)
    at256 = next(r for c, _, _, r in table if c == 262144)
    at508 = next(r for c, _, _, r in table if c == 507904)
    sl_v = slope(arms["vestigekv"], 8192, 507904)
    sl_d = slope(arms["dense"], 8192, 507904)

    # The memory-time product divides cache growth by this same speedup, so it
    # is not a separate measurement and must not be a separately kept number:
    # it moved from 0.75 to 0.83 when the speedup moved, and nothing would have
    # caught that while both were hand-maintained.
    nums = open(os.path.join(PAPER, "numbers.tex")).read()
    growth = float(re.search(r"\\newcommand\{\\memGrowth\}\{([\d.]+)\}", nums).group(1))
    memtime = growth / at508

    print(f"launch knobs shared by both arms: {knobs}")
    print(f"{'ctx':>6} {'dense ms':>9} {'vk ms':>8} {'ratio':>7}")
    for c, d, v, r in table:
        print(f"{c // 1024:>5}k {d:>9.3f} {v:>8.3f} {r:>7.3f}")
    print(f"crossover {cross // 1024}k   slope vk {sl_v:.1f} ns  dense {sl_d:.1f} ns")

    body = [
        f"% Generated by mexp/kimi/make_serving_speed_numbers.py from",
    ] + [f"%   results/kimi/{LOGS[arm]}" for arm in sorted(LOGS)] + [
        f"% Both arms one script, one sitting, one tree; knobs: {knobs}",
        "% Contexts below are labelled in units of 1024 tokens, so the figure's",
        "% rightmost point (507904 tokens) is the 496k row.",
        "",
        f"\\newcommand{{\\serveCrossover}}{{$\\sim${cross // 1024}k}}",
        f"\\newcommand{{\\serveMaxCtx}}{{{507904 // 1024}k}}",
        f"\\newcommand{{\\serveSpeedTwoFiveSix}}{{{at256:.2f}$\\times$}}",
        f"\\newcommand{{\\serveSpeedMax}}{{{at508:.2f}$\\times$}}",
        f"\\newcommand{{\\serveSpeedMaxBare}}{{{at508:.2f}}}",
        f"\\newcommand{{\\serveSlopeVk}}{{{sl_v:.1f}}}",
        f"\\newcommand{{\\serveSlopeDense}}{{{sl_d:.1f}}}",
        f"\\newcommand{{\\memTimeCross}}{{{memtime:.2f}}}",
        "",
        "% the reduced curve, for anyone checking the figure against the logs:",
    ] + [f"%   {c // 1024:>5}k  dense {d:.3f}  vk {v:.3f}  ratio {r:.3f}"
         for c, d, v, r in table]

    if a.emit:
        open(OUT, "w").write("\n".join(body) + "\n")
        print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
