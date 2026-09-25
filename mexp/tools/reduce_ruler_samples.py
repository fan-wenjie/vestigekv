#!/usr/bin/env python3
"""Reduce RULER generations to the per-item scores the paired test needs.

`samples_<arm>_<...>.json` is the full lm-eval record: prompt, generation,
filtered response and score for every item. At the long line that is 188 MB per
arm, which cannot live in `results/results.zip` (5 MB, git LFS), and the last
reduction deleted the whole category on the reasoning that no paper number
depended on it. That reasoning was wrong: the two arms answer the *same* items,
so the correct comparison is paired, and the pairing exists only in these files.

Nothing in the paired statistic reads the generated text -- it needs the score
of item `doc_id` at length `l` in each arm, and nothing else. This writes that
projection, three orders of magnitude smaller, so the archive can carry the
evidence for `\\rulerKLPaired*` instead of the reader having to re-run 256k
prompts to check a p-value.

    python mexp/tools/reduce_ruler_samples.py --line kimi
    python mexp/tools/reduce_ruler_samples.py --line kimi --verify

`--verify` re-reads each source and checks the reduction reproduces it item for
item; run it before replacing an archive, because a silent projection error
looks exactly like a real disagreement between the arms.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_NAME = "paired_scores.json"


def project(path):
    """{task: {"<doc_id>,<length>": score}} for one samples file.

    A record carries one key per length it was scored at, with -1.0 marking a
    length this item was not run at -- lm-eval writes the full length column on
    every record, so the sentinel, not the key's absence, is what says the item
    is absent."""
    raw = json.load(open(path))
    out = {}
    for task, recs in raw.items():
        d = {}
        for s in recs:
            for k, v in s.items():
                if k.isdigit() and isinstance(v, (int, float)) and v >= 0:
                    d[f"{s['doc_id']},{k}"] = v
        if d:
            out[task] = d
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--line", default="kimi")
    ap.add_argument("--verify", action="store_true",
                    help="re-read the sources and check the reduction matches")
    args = ap.parse_args()

    d = os.path.join(ROOT, "results", args.line, "ruler")
    srcs = sorted(glob.glob(os.path.join(d, "samples_*.json")))
    if not srcs:
        print(f"no samples_*.json under {d}", file=sys.stderr)
        return 1

    out, prov = {}, {}
    for p in srcs:
        # The key is the samples filename minus prefix and suffix, which is the
        # same string make_ruler_numbers.py builds from (arm, n, lengths, tag).
        key = os.path.basename(p)[len("samples_"):-len(".json")]
        out[key] = project(p)
        prov[key] = {"source": os.path.basename(p), "bytes": os.path.getsize(p),
                     "items": sum(len(v) for v in out[key].values()),
                     "tasks": len(out[key])}
        print(f"  {key}: {prov[key]['tasks']} tasks, {prov[key]['items']} items, "
              f"{prov[key]['bytes'] / 1e6:.0f} MB -> reduced")

    dest = os.path.join(d, OUT_NAME)
    if args.verify:
        if not os.path.exists(dest):
            print(f"nothing to verify: {dest} absent", file=sys.stderr)
            return 1
        have = json.load(open(dest))["scores"]
        bad = 0
        for key, want in out.items():
            got = have.get(key)
            if got != want:
                bad += 1
                print(f"  MISMATCH {key}")
        print(f"verify: {len(out) - bad}/{len(out)} reductions reproduce")
        return 1 if bad else 0

    json.dump({"scores": out, "provenance": prov}, open(dest, "w"))
    print(f"wrote {dest} ({os.path.getsize(dest) / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
