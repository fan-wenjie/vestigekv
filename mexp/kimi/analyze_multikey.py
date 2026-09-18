#!/usr/bin/env python3
"""Classify RULER needle errors by whether the right row was reached.

The paper explains the multi-key gap one way: a rank-r sketch fitted to the
archive's own content cannot separate rows drawn from the bulk it was fitted
to, so the certificate fires on the wrong row. That is a claim about SELECTION,
and it predicts a wrong answer -- a different needle's value.

The generations do not agree that this is the whole story. On
`niah_multikey_3`, whose values are 36-character UUIDs, most errors return the
CORRECT value with one character wrong:

    target  715109e9-f773-4f9b-a639-11dd1883a3d2
    got     415109e9-f773-4f9b-a639-11dd1883a3d2

UUIDs are random, so two needles sharing 35 of 36 characters does not happen by
chance. The model reached the right row and mis-emitted one token of the copy,
which is a per-DECODE-STEP failure, not a per-request selection failure. The
two are distinguishable here precisely because the answer is long: a 36-char
copy is ~20 decode steps, and one step missing its row shows up as one bad
character while its 19 neighbours stay correct.

This splits every wrong answer into:

  near   similarity to the target above --threshold: the right row, mis-copied.
  wrong  below it: a different row's value, which is the selection failure.

and reports the split per task. `niah_single_3` is the control: the same UUID
copy, the same length, but a haystack of essays rather than of key-value pairs.

    python mexp/kimi/analyze_multikey.py --samples results/kimi/ruler/samples_*.json
    python mexp/kimi/analyze_multikey.py --line kimi --examples 6
"""
from __future__ import annotations

import argparse
import difflib
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def response(s):
    r = s.get("filtered_resps") or s.get("resps") or [""]
    return (r[0] if isinstance(r, list) else r).strip()


def target(s):
    t = s.get("target")
    return (t[0] if isinstance(t, list) else str(t))


def classify(path, threshold):
    """[(task, length, doc_id, kind, target, response, similarity)] for one arm.

    Only scored items count: lm-eval writes every length key on every record and
    marks the ones this item was not run at with -1.0."""
    rows = []
    for task, recs in json.load(open(path)).items():
        for s in recs:
            for k, v in s.items():
                if not (k.isdigit() and isinstance(v, (int, float)) and v >= 0):
                    continue
                tgt, got = target(s), response(s)
                if v > 0:
                    kind = "ok"
                    sim = 1.0
                else:
                    sim = difflib.SequenceMatcher(None, tgt, got).ratio()
                    kind = "near" if sim > threshold else "wrong"
                rows.append((task, int(k), s["doc_id"], kind, tgt, got, sim))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", nargs="*", default=[],
                    help="samples_*.json files; default: every one under --line")
    ap.add_argument("--line", default="kimi")
    ap.add_argument("--threshold", type=float, default=0.8,
                    help="similarity above which a wrong answer is the right row")
    ap.add_argument("--examples", type=int, default=4)
    args = ap.parse_args()

    paths = args.samples or sorted(glob.glob(os.path.join(
        ROOT, "results", args.line, "ruler", "samples_*.json")))
    if not paths:
        print("no samples files")
        return 1

    for p in paths:
        rows = classify(p, args.threshold)
        print(f"\n## {os.path.basename(p)}")
        print(f"{'task':20s} {'n':>4} {'ok':>4} {'near':>5} {'wrong':>6}   near/err")
        for task in sorted({r[0] for r in rows}):
            t = [r for r in rows if r[0] == task]
            ok = sum(r[3] == "ok" for r in t)
            near = sum(r[3] == "near" for r in t)
            wrong = sum(r[3] == "wrong" for r in t)
            err = near + wrong
            frac = f"{near}/{err}" if err else "-"
            print(f"{task:20s} {len(t):>4} {ok:>4} {near:>5} {wrong:>6}   {frac:>8}")
        ex = [r for r in rows if r[3] == "near"][:args.examples]
        for task, l, doc, _, tgt, got, sim in ex:
            print(f"  near  {task} doc{doc} {l}  sim={sim:.2f}")
            print(f"        want {tgt}")
            print(f"        got  {got}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
