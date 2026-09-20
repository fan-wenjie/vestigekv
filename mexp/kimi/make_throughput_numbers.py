#!/usr/bin/env python3
"""Figure 2's right panel: decode throughput against batch, at two contexts.

    python mexp/kimi/make_throughput_numbers.py            # report
    python mexp/kimi/make_throughput_numbers.py --emit     # write the macros
    python mexp/kimi/make_throughput_numbers.py --check    # exit 1 on a mismatch

THE METRIC, AND THE TWO THINGS IT REPLACED. A point is the mean of the
server's own `gen throughput (token/s)` over a window in which `#running-req`
is constant and equal to the batch the point claims. That column is the
batch's decoded tokens per second, so it excludes prefill by construction and
assumes nothing about how many requests were resident.

It replaces `output_throughput` from the client, which divides decoded tokens
by the whole benchmark -- a third of which is prefill at bs=1, and prefill is
where VestigeKV is slower, so the dilution ran against us.

It also replaces `bs / mean_itl`, which excluded prefill correctly and was
still wrong: an average ITL says nothing about how many requests produced it.
The server log says the batch frequently was not the batch requested --

    64k   bs=1..24  #running-req exactly bs on every decode line   valid
    64k   bs=32     31 for 102 lines, then one request alone       invalid
    128k  bs=4      2 for all 102 lines, 8080 decoded of 16382     invalid
    128k  bs=8      5 for all 102 lines                            invalid

-- because 131072 tokens of prefill per request take long enough that early
requests finish their 4096 decoded tokens before late ones finish prefilling.
At 64k the scheduler completed every prefill before decoding began, the batch
was exact, and the defect was invisible.

So the plateau is not a convenience: it is the only part of a run where the
number on the x-axis is true. A point with no plateau is refused rather than
averaged, because an average over a collapsing batch looks entirely reasonable
and means nothing.
"""
from __future__ import annotations

import argparse
import datetime
import os
import re
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "results", "kimi")
PAPER = os.path.expanduser("~/vestigekv_paper")
OUT = os.path.join(PAPER, "serving_numbers.tex")

BATCHES = [1, 2, 4, 8, 12, 16, 24, 32]
WORD = {1: "One", 2: "Two", 4: "Four", 8: "Eight", 12: "Twelve", 16: "Sixteen",
        24: "TwentyFour", 32: "ThirtyTwo"}
CONTEXTS = {65536: ("SixtyFour", "tput32"), 131072: ("OneTwentyEight", "tput128k")}
ARMS = {"baseline": "dense", "vestigekv": "vestigekv"}
# 40 decode-log lines is 1600 forward steps: long enough that a rate is a rate
# and not a moment, short enough that a genuine plateau is not thrown away.
# The window's DURATION is what bounds the error, though, not its line count:
# see plateau() on the one-second stamps, and the sweep's output lengths, which
# are chosen so every point measures a comparable stretch of wall clock.
MIN_PLATEAU = 40
# And a floor on the window's DURATION, which is what actually bounds the
# error. The log stamps whole seconds, so a window of W seconds carries about
# 2/W of quantisation no matter how many lines it holds: at bs=1 the first
# sweep's plateau lasted 18 seconds and the counted and reported rates
# disagreed by 6%, which is the quantisation and not the machine. 120 seconds
# puts that under 2%, and the sweep's output lengths are chosen per point to
# reach it -- that is what makes two points comparable.
MIN_SPAN_S = 120

DECODE = re.compile(r"#running-req:\s*(\d+).*?#full token:\s*(\d+)"
                    r".*?gen throughput \(token/s\):\s*([\d.]+)")
STAMP = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")


def plateau(path, bs):
    """Tokens generated per second while exactly `bs` requests were live.

    Counted the direct way: how many tokens entered the pool over how much
    wall clock. `#full token` is the pool's occupancy and during pure decode
    it grows by one token per live request per step, so its increase over a
    window IS the tokens that window generated. Nothing here trusts a rate the
    server computed for itself.

    That rate is still read, as a cross-check. The two agree to 0.15-2.9% on
    the runs measured so far, and where they disagree most the window is
    shortest -- the log stamps whole seconds, so a 18-second window carries
    5% of quantisation on its own. The tolerance below is that quantisation,
    not a fudge factor: two seconds of stamp error over the window's length.
    """
    rows = []
    for line in open(path, errors="ignore"):
        if "Decode batch" not in line:
            continue
        m, t = DECODE.search(line), STAMP.search(line)
        if m and t:
            rows.append((datetime.datetime.strptime(t.group(1), "%Y-%m-%d %H:%M:%S"),
                         int(m.group(1)), int(m.group(2)), float(m.group(3))))
    best, cur = [], []
    for row in rows:
        if row[1] == bs and row[3] > 0:
            cur.append(row)
        else:
            best, cur = (cur if len(cur) > len(best) else best), []
    best = cur if len(cur) > len(best) else best
    if len(best) < MIN_PLATEAU:
        seen = sorted({live for _, live, _, _ in rows})
        return None, f"no run of {MIN_PLATEAU}+ decode lines at {bs} live (saw {seen})"
    # drop the plateau's first line: its interval starts before the batch
    # reached this width, so it prices part of another regime
    seg = best[1:]
    span = (seg[-1][0] - seg[0][0]).total_seconds()
    if span < MIN_SPAN_S:
        return None, (f"plateau lasted {span:.0f}s, under the {MIN_SPAN_S}s floor; "
                      f"one-second stamps put {200 / max(span, 1):.0f}% of "
                      "quantisation on a window that short")
    counted = (seg[-1][2] - seg[0][2]) / span
    reported = statistics.fmean(r[3] for r in seg)
    tol = max(0.02, 2.0 / span)
    if abs(counted - reported) > tol * reported:
        return None, (f"tokens counted ({counted:.1f}/s) and the server's reported "
                      f"rate ({reported:.1f}/s) differ by more than {100 * tol:.1f}%")
    return counted, len(seg)


def survey(reps=(None, 2, 3)):
    have, refused = {}, []
    for ctx, (_, prefix) in CONTEXTS.items():
        have[ctx] = {}
        for bs in BATCHES:
            point = {}
            for arm, tag in ARMS.items():
                runs = []
                for rep in reps:
                    job = f"{prefix}-bs{bs}{'' if not rep else 'r%d' % rep}-{tag}"
                    p = os.path.join(RESULTS, f"server_{arm}_{job}.log")
                    if not os.path.exists(p):
                        continue
                    val, why = plateau(p, bs)
                    if val is None:
                        refused.append(f"{job}: {why}")
                    else:
                        runs.append(val)
                if runs:
                    point[arm] = runs
            if len(point) == 2:
                have[ctx][bs] = point
    return have, refused


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    have, refused = survey()
    lines = ["% Generated by mexp/kimi/make_throughput_numbers.py.",
             "% Decoded tokens per second from the server's own decode log, over a",
             "% window where #running-req is constant and equal to the batch; prefill",
             "% is outside it by construction. Ratios are vestigekv/dense."]
    for ctx in sorted(have):
        word = CONTEXTS[ctx][0]
        if not have[ctx]:
            continue
        print(f"  {ctx // 1024}k prefill:")
        for bs in sorted(have[ctx]):
            d = statistics.fmean(have[ctx][bs]["baseline"])
            v = statistics.fmean(have[ctx][bs]["vestigekv"])
            n = min(len(have[ctx][bs]["baseline"]), len(have[ctx][bs]["vestigekv"]))
            print(f"    bs={bs:<3} dense {d:>8.1f}  vk {v:>8.1f}  {v / d:.3f}x  (n={n})")
            lines.append(f"\\newcommand{{\\srvBatch{word}{WORD[bs]}}}{{{v / d:.2f}}}"
                         f"  % {ctx // 1024}k, bs={bs}, n={n}: {v:.1f}/{d:.1f} tok/s")
        if 12 in have[ctx]:
            g = (statistics.fmean(have[ctx][12]["vestigekv"])
                 / statistics.fmean(have[ctx][12]["baseline"]) - 1)
            lines.append(f"\\newcommand{{\\tputGain{word}Twelve}}{{{g * 100:.1f}\\%}}")
    for r in refused:
        print(f"  REFUSED {r}")

    if not any(have.values()):
        raise SystemExit("ABORT: no point has a usable plateau")
    if a.emit:
        open(OUT, "w").write("\n".join(lines) + "\n")
        print(f"wrote {OUT}")
        nums = os.path.join(PAPER, "numbers.tex")
        text = open(nums).read()
        kept = [l for l in text.splitlines()
                if not re.match(r"\\newcommand\{\\tputGainTwelve\}", l.strip())]
        if len(kept) != len(text.splitlines()):
            open(nums, "w").write("\n".join(kept) + "\n")
            print("removed the hand-maintained \\tputGainTwelve from numbers.tex")
    if a.check:
        cur = open(OUT).read()
        for line in lines:
            m = re.match(r"\\newcommand\{\\(\w+)\}", line)
            if m and line.split("%")[0].strip() not in cur:
                raise SystemExit(f"ABORT: the paper's \\{m.group(1)} does not match "
                                 "the sweep on disk. Re-run with --emit.")
        print("check: the throughput macros match the records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
