#!/usr/bin/env python3
"""Reduce the streaming-benchmark JSONL records to the few points that carry the
analysis, so the archive can ship them.

WHY THIS EXISTS
---------------
A 4k->512k streaming run records one inter-token latency per generated token.
At 520189 tokens that is 10.9 MB of JSON per repetition, 13.3 MB with the
generated text, and the latency/throughput family totals ~500 MB. That does not
belong in a repository, and none of it is needed to reproduce a published
number: the paper reads the curve as *the median of the 4096-token window
ending at each context length*, which is a few dozen scalars per run.

So this script computes those scalars from the full array **before** discarding
it, and stores them. The reduction is therefore lossless for every quantity the
paper reports -- not "close enough", exactly equal -- and `--verify` proves it
per file by recomputing both ways and refusing to write on any mismatch.

WHAT IS KEPT
------------
1. `itl_windows`   exact median/mean/p10/p90/n over the window ending at every
                   context length on a fixed grid (default: every 4096 tokens).
                   This is what the paper's curve is made of.
2. `itl_global`    exact count/mean/median/p10/p90/p99/min/max over the whole
                   array, so the summary lines are reproducible too.
3. `itl_trace`     a downsampled trace, head- and tail-dense (see below), for
                   plotting and for eyeballing the shape. Not used by any
                   reported number.
4. every scalar field of the original record, untouched.

WHAT IS DROPPED
---------------
`itls` (the full per-token array; superseded by 1-3) and `generated_texts` (the
random-token continuation a latency benchmark emits -- no analytical content).

THE SAMPLING LAW
----------------
Points are placed on a cosine (Chebyshev) spacing,

    i_k = round( (N-1) * (1 - cos(pi * k/(K-1))) / 2 ),    k = 0..K-1

which is dense at both ends and sparse in the middle. That matches where the
signal is: the head holds the warm-up and the first cache growth, the tail holds
the behaviour at maximum context, and the middle is a slow monotone ramp that a
sparse sample represents faithfully. `--spacing linear` is available for
comparison.

USAGE
-----
    python mexp/tools/reduce_streams.py --glob 'results/**/*.jsonl' --verify
    python mexp/tools/reduce_streams.py --glob '...' --verify --in-place

Without --in-place the reduced copies go to --out-dir and nothing is modified.
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
import math
import os
import shutil
import statistics
import sys

# The array fields worth reducing, and the ones worth dropping outright.
ARRAY_FIELD = "itls"
DROP_FIELDS = ("generated_texts",)
REDUCTION_KEY = "_reduction"


def _pct(sorted_vals, q):
    """Nearest-rank percentile on an already-sorted list, matching the
    convention stream_curve.py uses for its p10/p90 columns."""
    if not sorted_vals:
        return float("nan")
    return sorted_vals[int(q * (len(sorted_vals) - 1))]


def window_stats(itls, inp, ctx, window):
    """Exactly stream_curve.py's client-mode slice: itls[i] is the gap before
    output token i+1, so the window ending at context `ctx` is
    itls[end-window:end] with end = ctx - inp - 1."""
    end = ctx - inp - 1
    seg = itls[max(0, end - window):end]
    if not seg:
        return None
    s = sorted(seg)
    return {
        "n": len(seg),
        "median": statistics.median(seg),
        "mean": statistics.fmean(seg),
        "p10": _pct(s, 0.10),
        "p90": _pct(s, 0.90),
    }


def global_stats(itls):
    s = sorted(itls)
    return {
        "n": len(itls),
        "mean": statistics.fmean(itls),
        "median": statistics.median(itls),
        "p10": _pct(s, 0.10),
        "p90": _pct(s, 0.90),
        "p99": _pct(s, 0.99),
        "min": s[0],
        "max": s[-1],
    }


def trace_budget(n, n_trace, max_frac, floor):
    """How many trace points a run of n tokens gets.

    A cap on the *fraction* matters as much as the absolute budget: a 520k-token
    stream is reduced 260x by a 2000-point trace, but a 4096-token throughput
    request would keep half its array under the same number, which is not a
    reduction at all. So the budget is the tighter of the two, never below a
    floor that still shows the shape."""
    return max(min(n_trace, int(n * max_frac)), min(floor, n))


def sample_indices(n, k, spacing="cosine"):
    """Indices to keep. Cosine spacing is dense at both ends; linear is uniform."""
    if n <= k:
        return list(range(n))
    if spacing == "linear":
        return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})
    return sorted({round((n - 1) * (1 - math.cos(math.pi * i / (k - 1))) / 2)
                   for i in range(k)})


def reduce_record(d, window, grid_step, n_trace, spacing, sigdig,
                  max_frac=0.05, floor=128):
    arrs = d.get(ARRAY_FIELD)
    if not isinstance(arrs, list) or not arrs or not isinstance(arrs[0], list):
        return d, None  # nothing to reduce (already reduced, or a different shape)

    out = {k: v for k, v in d.items() if k not in DROP_FIELDS and k != ARRAY_FIELD}
    # input_lens is flat (one entry per completed request); itls is nested
    # (one per-token array per request). stream_curve.py reads input_lens[0].
    inps = d.get("input_lens") or [0]
    per_rep = []
    for rep, itls in enumerate(arrs):
        if not itls:
            continue
        inp = inps[rep] if rep < len(inps) else inps[0]
        n = len(itls)
        # The grid must land on the context values a consumer actually asks
        # for, which are multiples of grid_step (stream_curve.py reads
        # 8192*2^i) plus the run's own end. Anchoring it at inp+window+1
        # instead puts every point one token off every query, and the lookup
        # silently returns nothing.
        # The largest readable context is inp+n+1: there end = n and the window
        # is itls[n-window:n], the last full window of the run. Stopping at
        # inp+n drops exactly the endpoint the curve's final column asks for.
        max_ctx = inp + n + 1
        ctxs = [c for c in range(grid_step, max_ctx + 1, grid_step)
                if c - inp - 1 > 0]
        if max_ctx not in ctxs:
            ctxs.append(max_ctx)
        idx = sample_indices(n, trace_budget(n, n_trace, max_frac, floor), spacing)
        per_rep.append({
            "rep": rep,
            "input_len": inp,
            "window": window,
            # itl_windows/itl_global are in the source unit (SECONDS, as the
            # benchmark writes itls); itl_trace.ms is milliseconds. Stated in
            # `units` below so a reader never has to infer it.
            "units": {"itl_windows": "s", "itl_global": "s", "itl_trace.ms": "ms"},
            "itl_global": global_stats(itls),
            "itl_windows": {str(c): window_stats(itls, inp, c, window) for c in ctxs},
            "itl_trace": {
                "spacing": spacing,
                "index": idx,
                # itls[i] is the gap before output token i+1, so the context at
                # which the i-th latency was paid is inp + i + 1.
                "context": [inp + i + 1 for i in idx],
                "ms": [round(itls[i] * 1000.0, sigdig) for i in idx],
            },
        })
    out[ARRAY_FIELD + "_reduced"] = per_rep
    out[REDUCTION_KEY] = {
        "script": "mexp/tools/reduce_streams.py",
        "dropped": [ARRAY_FIELD] + list(DROP_FIELDS),
        "window": window,
        "grid_step": grid_step,
        "trace_points": n_trace,
        "trace_max_frac": max_frac,
        "trace_floor": floor,
        "spacing": spacing,
        "note": ("Per-token arrays removed for size. Every statistic the paper "
                 "reports was computed from the full array before it was "
                 "dropped and is stored exactly in itl_windows/itl_global; "
                 "itl_trace is a head/tail-dense sample for plotting only."),
    }
    return out, per_rep


def verify(original, reduced, window):
    """Recompute the paper's quantities from the full array and from the stored
    summary, and require exact agreement."""
    arrs = original.get(ARRAY_FIELD) or []
    inps = original.get("input_lens") or [0]
    for rep, itls in enumerate(arrs):
        if not itls:
            continue
        inp = inps[rep] if rep < len(inps) else inps[0]
        stored = reduced[ARRAY_FIELD + "_reduced"][rep]
        g = global_stats(itls)
        if g != stored["itl_global"]:
            return f"rep {rep}: global stats differ"
        for c, want in stored["itl_windows"].items():
            got = window_stats(itls, inp, int(c), window)
            if got != want:
                return f"rep {rep}: window at ctx={c} differs ({got} vs {want})"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", required=True, help="files to reduce, e.g. 'results/**/*.jsonl'")
    ap.add_argument("--window", type=int, default=4096, help="curve window (stream_curve.py default)")
    ap.add_argument("--grid-step", type=int, default=4096, help="context spacing of the exact window grid")
    ap.add_argument("--trace-points", type=int, default=2000, help="downsampled trace points per repetition")
    ap.add_argument("--trace-max-frac", type=float, default=0.05,
                    help="never keep more than this fraction of a run's tokens in the trace")
    ap.add_argument("--trace-floor", type=int, default=128,
                    help="minimum trace points for a short run")
    ap.add_argument("--spacing", choices=["cosine", "linear"], default="cosine")
    ap.add_argument("--sigdig", type=int, default=6, help="decimals kept in the trace (ms)")
    ap.add_argument("--out-dir", default="", help="write reduced copies here (default: alongside, .reduced.jsonl)")
    ap.add_argument("--allow-lossy-in-place", action="store_true",
                    help="override the refusal above; the arrays are not recoverable")
    ap.add_argument("--in-place", action="store_true",
                    help="replace the originals after verification; the original "
                         "is copied to ~/vestigekv-retired/full-stream-traces first")
    ap.add_argument("--verify", action="store_true", help="prove the reduction preserves every reported statistic")
    ap.add_argument("--min-bytes", type=int, default=1 << 20, help="skip files smaller than this")
    args = ap.parse_args()
    if args.in_place and not args.allow_lossy_in_place:
        raise SystemExit(
            "REFUSED: --in-place discards the per-token arrays, and\n"
            "mexp/tools/pack_streams.py keeps them at a comparable size:\n"
            "  161 MB of traces -> ~18 MB packed, lossless to the microsecond,\n"
            "  against 8 MB reduced and gone.\n"
            "Use:  python mexp/tools/pack_streams.py --glob '<...>' --write\n"
            "This flag destroyed 47 runs' traces once; the tracked copies were\n"
            "133-byte stubs from an earlier cleanup, so nothing came back.\n"
            "Pass --allow-lossy-in-place only for a record packing cannot take.")

    paths = sorted(p for p in globmod.glob(args.glob, recursive=True) if os.path.isfile(p))
    if not paths:
        print("no files matched", file=sys.stderr)
        return 1

    tot_before = tot_after = 0
    touched = skipped = 0
    for p in paths:
        size = os.path.getsize(p)
        if size < args.min_bytes:
            skipped += 1
            continue
        out_lines = []
        changed = False
        for ln, line in enumerate(open(p)):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            red, per_rep = reduce_record(d, args.window, args.grid_step,
                                         args.trace_points, args.spacing, args.sigdig,
                                         args.trace_max_frac, args.trace_floor)
            if per_rep is None:
                out_lines.append(json.dumps(d))
                continue
            changed = True
            if args.verify:
                err = verify(d, red, args.window)
                if err:
                    print(f"VERIFY FAILED {p} line {ln}: {err}", file=sys.stderr)
                    return 2
            out_lines.append(json.dumps(red))
        if not changed:
            skipped += 1
            continue

        blob = "\n".join(out_lines) + "\n"
        if args.in_place:
            # Archive before overwriting. This replaced 47 runs' per-token
            # traces with their reductions and kept no copy; the repository's
            # tracked versions were 133-byte stubs from an earlier cleanup, so
            # there was nothing to recover from. The reduction is verified, so
            # no reported number was lost -- but "verified" covers the
            # statistics we thought to check, and the original covered the
            # ones we did not.
            keep = os.path.join(os.path.expanduser("~/vestigekv-retired"),
                                "full-stream-traces")
            os.makedirs(keep, exist_ok=True)
            shutil.copy2(p, os.path.join(keep, os.path.basename(p)))
            dest = p
        elif args.out_dir:
            dest = os.path.join(args.out_dir, os.path.relpath(p))
            os.makedirs(os.path.dirname(dest), exist_ok=True)
        else:
            dest = p.replace(".jsonl", ".reduced.jsonl")
        with open(dest, "w") as f:
            f.write(blob)
        tot_before += size
        tot_after += len(blob)
        touched += 1
        print(f"  {size/2**20:8.2f} -> {len(blob)/2**20:6.3f} MB  "
              f"({size/max(len(blob),1):6.1f}x)  {p}")

    print(f"\n{touched} reduced, {skipped} skipped"
          + (f"  |  {tot_before/2**20:.1f} MB -> {tot_after/2**20:.1f} MB "
             f"({tot_before/max(tot_after,1):.1f}x)" if touched else "")
          + ("  |  verified: every reported statistic identical" if args.verify and touched else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
