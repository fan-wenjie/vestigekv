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

THE METRIC. The time to decode a 4096-token bucket, divided by 4096. Each
`gen throughput (token/s)` line covers the 40 tokens since the previous one,
so 40/throughput is that interval's duration and the bucket's total is their
sum: every millisecond the server spent is inside it.

It used to be the median of those lines instead, and the median is not a cost
-- it is a cost with the expensive steps removed. That matters here in one
direction only. Dense decode has no periodic work and its tail is flat (p50
7.115, max 7.130 ms at 509k); VestigeKV closes a block and rebuilds an index
every 4096 tokens, and its tail is not (p50 4.119, max 4.548 at 64k). A
summary that drops the tail drops our own periodic cost and nobody else's,
which is the one accusation a speed claim cannot answer. The client-side ITL is not the metric
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
import collections
import datetime
import os
import re

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
STAMP = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")


def read(path):
    """Steady-state decode lines: (context, tokens/s).

    The first line of a stream is dropped. Its reported rate covers everything
    since the server last logged, which is the whole prefill plus a handful of
    decoded tokens -- 598.8 ms/token here against a steady 3.9 -- so it is not
    a decode interval and a mean that includes it is not a decode cost. The
    rule is general rather than "drop line one": keep a line only if the
    previous one sits exactly one logging interval behind it, which is also
    what excludes a restart or a second request mid-log."""
    raw = []
    for line in open(path, errors="ignore"):
        m = LINE.search(line)
        if m and float(m.group(2)) > 0:
            raw.append((int(m.group(1)), float(m.group(2))))
    if len(raw) < 2:
        raise SystemExit(f"ABORT: no gen-throughput lines in {path}")
    step = collections.Counter(b - a for (a, _), (b, _) in zip(raw, raw[1:])).most_common(1)[0][0]
    rows = [cur for prev, cur in zip(raw, raw[1:]) if cur[0] - prev[0] == step]
    dropped = len(raw) - len(rows)
    if dropped > 0.02 * len(raw):
        raise SystemExit(
            f"ABORT: {os.path.basename(path)} drops {dropped}/{len(raw)} lines as "
            "non-steady-state; that is too many to be prefill boundaries")
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


def check_clock(path, rows, step):
    """The server's reported rates must add up to the run's wall clock.

    Each bucket here spans 4096 tokens, about 17 seconds at bs=1, and the log
    stamps whole seconds -- so differencing stamps per bucket would carry 12%
    of quantisation and the per-line reported rate is the only usable timer at
    that granularity. That rate is the server's own, though, and the
    throughput reducer stopped trusting an unchecked one for good reason.

    It is checkable over the whole run instead, where the same stamps are
    worth 0.08%: the interval durations the rates imply must sum to the
    elapsed time between the first and last decode line.
    """
    stamps = []
    for line in open(path, errors="ignore"):
        m, t = LINE.search(line), STAMP.search(line)
        if m and t and float(m.group(2)) > 0:
            stamps.append(datetime.datetime.strptime(t.group(1), "%Y-%m-%d %H:%M:%S"))
    if len(stamps) < 2:
        return
    elapsed = (stamps[-1] - stamps[0]).total_seconds()
    implied = sum(step / tput for _, tput in rows)
    if abs(implied - elapsed) > 0.01 * elapsed:
        raise SystemExit(
            f"ABORT: {os.path.basename(path)} reports rates that imply "
            f"{implied:.0f}s of decoding over an elapsed {elapsed:.0f}s; one of "
            "them is not measuring what it says")
    print(f"   clock check {os.path.basename(path)[:44]:46s} "
          f"implied {implied:.0f}s vs elapsed {elapsed:.0f}s "
          f"({100 * abs(implied - elapsed) / elapsed:.2f}%)")


def check_even(rows):
    """The log lines must cover equal numbers of tokens, and here is why.

    A bucket's cost is the time it took divided by the tokens it decoded:
    sum(step/tput) / sum(step). When every line covers the same step, the step
    cancels and the cost is just the mean of the per-line ms/token -- no
    constant enters the arithmetic and none can go stale. When the lines cover
    different spans that identity fails and an unweighted mean silently
    overweights the short intervals, so this refuses rather than computing it.
    """
    gaps = {b - a for (a, _), (b, _) in zip(rows, rows[1:]) if b > a}
    if len(gaps) != 1:
        raise SystemExit(f"ABORT: decode log lines are not evenly spaced: {sorted(gaps)[:5]}")
    return gaps.pop()


def window(rows, ctx):
    """Milliseconds per token over the 4096 tokens ENDING at ctx: that
    bucket's total decode time divided by the tokens it decoded. Equal log
    spacing (check_even) is what makes that the plain mean below.

    Trailing and not centred, which is what lets the curve reach the end of
    the stream. A centred bucket at 512k would be half empty -- the stream
    stops there -- and the previous version dealt with that by stopping the
    curve at 509k instead. Centring one point and trailing another is the
    special case; trailing every point costs about 0.3%, in the direction of
    a smaller reported speedup, because a trailing bucket averages over a
    slightly shorter context than the one it is labelled with."""
    ms = [1000.0 / tput for n, tput in rows if ctx - 2 * HALFWIDTH < n <= ctx]
    return sum(ms) / len(ms) if ms else None


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
    steps = {arm: check_even(rows) for arm, rows in arms.items()}
    if len(set(steps.values())) != 1:
        raise SystemExit(f"ABORT: the arms logged at different intervals: {steps}")
    step = next(iter(steps.values()))  # reported, not used in the mean: it cancels
    for arm, rows in arms.items():
        check_clock(paths[arm], rows, step)

    # The rightmost point is the end of the stream, derived rather than
    # chosen: the last logged context rounded up to the kilo-token it is
    # within one logging interval of. The constant it replaced, 507904, came
    # from a hand-written macro comment and 13k of context short of the end,
    # and nobody could have said why.
    last = min(max(n for n, _ in rows) for rows in arms.values())
    top = -(-last // 1024) * 1024
    if top - last > step:
        raise SystemExit(
            f"ABORT: the log stops at {last}, more than one {step}-token logging "
            f"interval short of {top}; the last bucket would be incomplete")
    grid = [8192, 16384, 20480, 24576, 28672, 32768, 65536, 131072, 262144, top]
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
    at508 = next(r for c, _, _, r in table if c == top)
    sl_v = slope(arms["vestigekv"], 8192, top)
    sl_d = slope(arms["dense"], 8192, top)

    # The memory-time product divides cache growth by this same speedup, so it
    # is not a separate measurement and must not be a separately kept number:
    # it moved from 0.75 to 0.83 when the speedup moved, and nothing would have
    # caught that while both were hand-maintained.
    nums = open(os.path.join(PAPER, "numbers.tex")).read()
    growth = float(re.search(r"\\newcommand\{\\memGrowth\}\{([\d.]+)\}", nums).group(1))
    memtime = growth / at508

    print(f"launch knobs shared by both arms: {knobs}")
    print(f"decode log interval: {step} tokens")
    print(f"{'ctx':>6} {'dense ms':>9} {'vk ms':>8} {'ratio':>7}")
    for c, d, v, r in table:
        print(f"{c // 1024:>5}k {d:>9.3f} {v:>8.3f} {r:>7.3f}")
    print(f"crossover {cross // 1024}k   slope vk {sl_v:.1f} ns  dense {sl_d:.1f} ns")

    body = [
        f"% Generated by mexp/kimi/make_serving_speed_numbers.py from",
    ] + [f"%   results/kimi/{LOGS[arm]}" for arm in sorted(LOGS)] + [
        f"% Both arms one script, one sitting, one tree; knobs: {knobs}",
        "% A row at N is the 4096 tokens ENDING at N -- the 512k row is the",
        "% 508k-512k interval -- labelled in units of 1024 tokens.",
        f"% The rightmost row is the end of the stream ({top} tokens; the last",
        f"% logged step lands at {last}).",
        "",
        f"\\newcommand{{\\serveCrossover}}{{$\\sim${cross // 1024}k}}",
        f"\\newcommand{{\\serveMaxCtx}}{{{top // 1024}k}}",
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
