#!/usr/bin/env python3
"""Compare two MRCR arms and say WHICH failure each one has.

    python mexp/kimi/analyze_mrcr.py \
        --dense results/kimi/mrcr/results_baseline_2needle_mrcr2-dense.json \
        --vk    results/kimi/mrcr/results_vestigekv_2needle_mrcr2-vestigekv.json

A mean MRCR score hides three different things, and they call for three
different fixes:

  the prefix is missing          -- the model did not follow the instruction;
                                    nothing about the cache is implicated
  the prefix is there and the    -- the model retrieved the WRONG INSTANCE:
  response matches a SIBLING        it answered "the first essay" when asked
  answer better than the target     for the second. For a compressed cache
                                    this is the diagnostic that matters: the
                                    rows it served were a near-duplicate of
                                    the right ones, which is exactly what a
                                    rank-r sketch fitted to the bulk cannot
                                    separate
  the prefix is there and the    -- fidelity: the right rows, reproduced badly
  response matches nothing well

Only the second says "fix the sketch". Reporting a single mean would let a
regression in any one of them be read as a regression in the others, and the
algorithm change that follows would be aimed at the wrong thing.

Siblings are recovered from the dataset rather than stored: run_mrcr.py takes
an even stride over each bin, which is deterministic given (rows, n), so the
same stride reproduces which source row each sample came from. A mismatch
between the record's answer and the reconstructed row aborts rather than
guesses.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
from difflib import SequenceMatcher

REPO = "giulio98/MRCR_v2_common"
ORD = re.compile(r"Prepend \S+ to the (\w+)\b", re.I)


def take(rows, n):
    if n >= len(rows):
        return list(range(len(rows)))
    step = len(rows) / n
    return [int(i * step) for i in range(n)]


def load_bin(needles, b):
    from huggingface_hub import list_repo_files, hf_hub_download
    import pyarrow.parquet as pq
    cfg = f"{needles}needle_in_{b}"
    files = sorted(f for f in list_repo_files(REPO, repo_type="dataset")
                   if f.startswith(cfg + "/"))
    rows = []
    for f in files:
        t = pq.read_table(hf_hub_download(REPO, f, repo_type="dataset"))
        cols = t.column_names
        for i in range(t.num_rows):
            rows.append({c: t.column(c)[i].as_py() for c in cols})
    return rows


def ratio(a, b):
    return float(SequenceMatcher(None, a, b).ratio())


def lcp(a, b):
    """Characters of verbatim agreement from the start.

    The graded ratio mixes "found the right passage" with "kept copying it".
    A response that opens correctly and then diverges has done the first and
    failed the second, which is neither a selection error nor uniform noise,
    and only this separates it from both.
    """
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def classify(rec, samples, needles):
    """Per sample: prefix_ok, target ratio, best sibling ratio."""
    out = []
    by_bin = {}
    for b in rec["bins"]:
        rows = load_bin(needles, b)
        idx = take(rows, rec["n_per_bin"])
        by_bin[b] = (rows, idx)
    for s in samples:
        rows, idx = by_bin[s["bin"]]
        src = rows[idx[s["i"]]]
        ans = src["answers"][0] if isinstance(src["answers"], list) else src["answers"]
        if ans != s["answer"]:
            raise SystemExit(
                f"ABORT: bin {s['bin']} sample {s['i']} does not match the "
                f"reconstructed source row; the stride assumption is wrong and "
                f"every sibling below would be the wrong comparison")
        # siblings: same context, different question
        sibs = [r for r in rows
                if r["context"] == src["context"] and r["question"] != src["question"]]
        resp = s["response"]
        has = resp.startswith(s["prefix"])
        body = resp.removeprefix(s["prefix"]) if has else resp
        tgt = ratio(body, ans.removeprefix(s["prefix"]))
        best_sib, best_r = None, 0.0
        for r in sibs:
            a2 = r["answers"][0] if isinstance(r["answers"], list) else r["answers"]
            m = re.search(r"Prepend (\S+) to ", r["question"])
            a2b = a2.removeprefix(m.group(1)) if m else a2
            rr = ratio(body, a2b)
            if rr > best_r:
                best_r, best_sib = rr, r["question"]
        om = ORD.search(src["question"])
        full = ans.removeprefix(s["prefix"])
        out.append({"bin": s["bin"], "i": s["i"], "prefix_ok": has,
                    "target": tgt, "sibling": best_r,
                    "copy_frac": lcp(body, full) / max(len(full), 1),
                    "len_ratio": len(body) / max(len(full), 1),
                    "asked": om.group(1).lower() if om else None,
                    "n_siblings": len(sibs),
                    "sibling_q": best_sib})
    return out


def summarize(name, rec, cls):
    print(f"\n=== {name}  ({rec['arm']}, {rec['needles']}needle, "
          f"n={rec['n_per_bin']}/bin, {rec['wall_s']:.0f}s)")
    print(f"{'bin':>16} {'score':>7} {'prefix':>7} {'target':>7} {'sibling':>8} "
          f"{'wrong-inst':>11} {'fidelity':>9} {'copy':>6} {'full':>7} {'len':>5}")
    for b in rec["bins"]:
        v = [c for c in cls if c["bin"] == b]
        if not v:
            continue
        rows = [r for r in rec["rows"] if r["bin"] == b]
        # wrong instance: prefix right, and a sibling answer explains the
        # response better than the target does
        wrong = [c for c in v if c["prefix_ok"] and c["sibling"] > c["target"]]
        fid = [c for c in v if c["prefix_ok"] and c["sibling"] <= c["target"]]
        print(f"{b:>16} {statistics.mean(r['score'] for r in rows):>7.3f} "
              f"{statistics.mean(1.0 if c['prefix_ok'] else 0.0 for c in v):>7.2f} "
              f"{statistics.mean(c['target'] for c in v):>7.3f} "
              f"{statistics.mean(c['sibling'] for c in v):>8.3f} "
              f"{len(wrong)}/{len(v):<9} "
              f"{statistics.mean(c['target'] for c in fid) if fid else float('nan'):>9.3f} "
              f"{statistics.median(c['copy_frac'] for c in v):>6.3f} "
              f"{sum(1 for c in v if c['copy_frac'] >= 0.999)}/{len(v):<5} "
              f"{statistics.median(c['len_ratio'] for c in v):>5.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", required=True)
    ap.add_argument("--vk", required=True)
    a = ap.parse_args()

    out = {}
    for tag, path in (("dense", a.dense), ("vestigekv", a.vk)):
        rec = json.load(open(path))
        smp = json.load(open(path.replace("results_", "samples_")))
        cls = classify(rec, smp, rec["needles"])
        summarize(tag, rec, cls)
        out[tag] = (rec, cls)

    d, v = out["dense"], out["vestigekv"]
    print("\n=== paired, per bin (vestigekv - dense)")
    print(f"{'bin':>16} {'dscore':>8} {'dtarget':>8} {'dprefix':>8} "
          f"{'wrong-inst d/v':>15}")
    for b in d[0]["bins"]:
        dr = [r for r in d[0]["rows"] if r["bin"] == b]
        vr = [r for r in v[0]["rows"] if r["bin"] == b]
        dc = [c for c in d[1] if c["bin"] == b]
        vc = [c for c in v[1] if c["bin"] == b]
        if not dr or not vr:
            continue
        dw = sum(1 for c in dc if c["prefix_ok"] and c["sibling"] > c["target"])
        vw = sum(1 for c in vc if c["prefix_ok"] and c["sibling"] > c["target"])
        print(f"{b:>16} "
              f"{statistics.mean(r['score'] for r in vr) - statistics.mean(r['score'] for r in dr):>+8.3f} "
              f"{statistics.mean(c['target'] for c in vc) - statistics.mean(c['target'] for c in dc):>+8.3f} "
              f"{statistics.mean(1.0 if c['prefix_ok'] else 0.0 for c in vc) - statistics.mean(1.0 if c['prefix_ok'] else 0.0 for c in dc):>+8.2f} "
              f"{dw}/{len(dc)} -> {vw}/{len(vc)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
