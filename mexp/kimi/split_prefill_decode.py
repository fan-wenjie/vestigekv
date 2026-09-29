#!/usr/bin/env python3
"""Prefill and decode as two records, never one, into results/kimi/speed/.

    python mexp/kimi/split_prefill_decode.py                 # rebuild and print
    python mexp/kimi/split_prefill_decode.py --table         # print only
    python mexp/kimi/split_prefill_decode.py --job <id> ...  # restrict

A merged tokens-per-second is the wrong unit for this method and the record
has to make that impossible to quote by accident. VestigeKV is SLOWER in
prefill and faster in decode, so any figure that divides decoded tokens by a
wall clock containing prefill reports the two halves cancelling, in a
proportion set by the prompt length rather than by the method. That is how a
1.56x decode gain was reported here as 1.09.

THE SPLIT. The first token ends prefill. TTFT therefore belongs to prefill --
it contains the prompt's forward pass and the index build that follows it --
and decode is measured from the first token onward, which is exactly what the
client's ITL/TPOT already exclude TTFT to mean. No number in this file spans
the boundary.

    prefill   TTFT median/p99 (ms), from the client
    decode    ITL median/p99 (ms), from the client
              tok/s, from the SERVER's own `gen throughput` over a window
              where #running-req is constant -- see below

WHY THE SERVER FOR THROUGHPUT. The client's `Output token throughput` divides
by the whole benchmark, prefill included; it is recorded here as
`merged_tok_s` with a field name that says not to quote it. The server's
decode log reports the batch's decoded tokens per second directly, so prefill
is outside it by construction. Windows are labelled by the concurrency that
ACTUALLY held, not the one the job requested: at 128k a request's prefill
takes long enough that early requests finish decoding before late ones start,
so a job asking for 4 can run 2 (make_throughput_numbers.plateaus, whose
implementation this imports rather than copies).

AUDITABILITY. Every row carries the log paths it came from, the window's span
in seconds, and the actual concurrency. A row whose window is shorter than
MIN_SPAN_S is written with `decode_tok_s: null` and a reason, not dropped and
not averaged -- the log stamps whole seconds, so a short window carries
quantisation that a reader cannot see once it has become a ratio.

The paper is frozen; this writes to results/kimi/speed/ and nothing else, so
the numbers accumulate in one place for whenever it reopens.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "results", "kimi")
OUT_DIR = os.path.join(RESULTS, "speed")
OUT = os.path.join(OUT_DIR, "points.jsonl")

sys.path.insert(0, HERE)
from make_throughput_numbers import MIN_SPAN_S, plateaus  # noqa: E402

# One entry per timed sweep: (context, arm label, client-log glob, server log).
# The server log is named separately because the runner keeps ONE server across
# consecutive jobs with the same env, so a whole sweep's decode lines land in
# the log of whichever job launched it. plateaus() labels by actual
# concurrency, so a shared log is usable -- but only if it is named honestly.
SWEEPS = [
    (65536, "dense", "stream_baseline_tputA-bs{bs}-dense",
     "server_baseline_tputA-bs{bs}-dense"),
    (65536, "vestigekv", "stream_vestigekv_tputA-bs{bs}-vestigekv",
     "server_vestigekv_tputA-bs{bs}-vestigekv"),
    (65536, "branch", "stream_vestigekv_tputAbr-bs{bs}",
     "server_vestigekv_tputAbr-bs1"),
    (131072, "dense", "stream_baseline_tputB87-bs{bs}-dense",
     "server_baseline_tputB87-bs{bs}-dense"),
    (131072, "vestigekv", "stream_vestigekv_tputB87-bs{bs}-vestigekv",
     "server_vestigekv_tputB87-bs{bs}-vestigekv"),
    (131072, "branch", "stream_vestigekv_tputBbr-bs{bs}",
     "server_vestigekv_tputBbr-bs1"),
]
BATCHES = [1, 2, 4, 8, 12, 16, 24, 32]

CLIENT_FIELDS = {
    "ttft_median_ms": r"Median TTFT \(ms\):\s*([\d.]+)",
    "ttft_p99_ms": r"P99 TTFT \(ms\):\s*([\d.]+)",
    "itl_median_ms": r"Median ITL \(ms\):\s*([\d.]+)",
    "itl_p99_ms": r"P99 ITL \(ms\):\s*([\d.]+)",
    "tpot_median_ms": r"Median TPOT \(ms\):\s*([\d.]+)",
    # Recorded, never quoted: it spans the boundary this file exists to keep.
    "merged_tok_s_DO_NOT_QUOTE": r"Output token throughput \(tok/s\):\s*([\d.]+)",
}


def read_client(path):
    if not os.path.exists(path):
        return None
    text = open(path, errors="ignore").read()
    out = {}
    for name, pat in CLIENT_FIELDS.items():
        m = re.search(pat, text)
        out[name] = float(m.group(1)) if m else None
    return out


def read_decode(path, bs):
    """Decoded tok/s at the concurrency that actually held, and its window."""
    if not os.path.exists(path):
        return None, None, "server log missing"
    hits = [p for p in plateaus(path) if p[0] == bs]
    if not hits:
        return None, 0.0, f"no decode window ran at {bs} live requests"
    tok = sum(p[1] for p in hits)
    sec = sum(p[2] for p in hits)
    if sec < MIN_SPAN_S:
        return None, sec, (f"window {sec:.0f}s is under the {MIN_SPAN_S}s floor; "
                           "the log stamps whole seconds, so a ratio from it "
                           "would carry unshowable quantisation")
    return tok / sec, sec, ""


def build(jobs=None):
    rows = []
    for ctx, arm, cpat, spat in SWEEPS:
        for bs in BATCHES:
            cpath = os.path.join(RESULTS, cpat.format(bs=bs) + ".log")
            spath = os.path.join(RESULTS, spat.format(bs=bs) + ".log")
            if jobs and not any(j in cpat.format(bs=bs) for j in jobs):
                continue
            client = read_client(cpath)
            if client is None:
                continue
            tok_s, span, why = read_decode(spath, bs)
            rows.append({
                "context": ctx, "arm": arm, "batch_requested": bs,
                "prefill": {"ttft_median_ms": client["ttft_median_ms"],
                            "ttft_p99_ms": client["ttft_p99_ms"]},
                "decode": {"itl_median_ms": client["itl_median_ms"],
                           "itl_p99_ms": client["itl_p99_ms"],
                           "tpot_median_ms": client["tpot_median_ms"],
                           "tok_s": tok_s, "window_s": span,
                           "refused_because": why or None},
                "merged_tok_s_DO_NOT_QUOTE":
                    client["merged_tok_s_DO_NOT_QUOTE"],
                "source": {"client": os.path.relpath(cpath, ROOT),
                           "server": os.path.relpath(spath, ROOT)},
            })
    return rows


def table(rows):
    by = {(r["context"], r["arm"], r["batch_requested"]): r for r in rows}
    for ctx in sorted({r["context"] for r in rows}):
        print(f"\n=== {ctx // 1024}k prompt")
        print(f"{'bs':>3} | {'PREFILL: TTFT median ms':>34} | "
              f"{'DECODE: ITL median ms':>30} | {'DECODE: tok/s':>34}")
        print(f"{'':>3} | {'dense':>10}{'vk':>11}{'branch':>11} | "
              f"{'dense':>9}{'vk':>10}{'branch':>10} | "
              f"{'dense':>10}{'vk':>11}{'branch':>11}")
        for bs in BATCHES:
            got = [by.get((ctx, a, bs)) for a in ("dense", "vestigekv", "branch")]
            if not any(got):
                continue
            def cell(r, group, key, w, p=1):
                if r is None or r[group][key] is None:
                    return " " * (w - 1) + "-"
                return f"{r[group][key]:>{w}.{p}f}"
            print(f"{bs:>3} | " +
                  "".join(cell(r, "prefill", "ttft_median_ms", 11 if i else 10, 0)
                          for i, r in enumerate(got)) + " | " +
                  "".join(cell(r, "decode", "itl_median_ms", 10 if i else 9, 2)
                          for i, r in enumerate(got)) + " | " +
                  "".join(cell(r, "decode", "tok_s", 11 if i else 10, 1)
                          for i, r in enumerate(got)))
    refused = [r for r in rows if r["decode"]["refused_because"]]
    if refused:
        print(f"\n{len(refused)} point(s) carry no decode tok/s:")
        for r in refused[:8]:
            print(f"  {r['context']//1024}k {r['arm']:<10} bs={r['batch_requested']:<3} "
                  f"{r['decode']['refused_because']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", action="store_true", help="print without writing")
    ap.add_argument("--job", action="append", default=None)
    a = ap.parse_args()
    rows = build(a.job)
    if not a.table:
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(OUT, "w") as f:
            for r in rows:
                f.write(json.dumps(r, sort_keys=True) + "\n")
        print(f"wrote {len(rows)} points -> {os.path.relpath(OUT, ROOT)}")
    table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
