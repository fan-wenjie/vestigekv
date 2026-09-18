#!/usr/bin/env python3
"""Reconstruct results/<line>/queue_state.jsonl after it is lost.

The runner decides what to run from this file alone: `done` is the set of ids
whose last record has status done or failed, and anything else in queue.jsonl is
fair game. So losing the file does not stall the queue -- it silently re-runs
the entire thing, which is far worse, and the loss is invisible until hours of
GPU time have been spent re-deriving numbers that already existed.

Two independent sources say a job finished, and this uses both, because neither
is complete on its own:

1. **The runner log** (`results/<line>_queue_runner.log`), whose lines read
   `[HH:MM:SS] <id> done rc=0 wall=1967s`. Authoritative for this runner
   process, but blind to anything finished before it started.
2. **The result artifacts** a finished job leaves behind, whose filenames carry
   the job id as a tag. Survives across runner restarts, and can be read from
   `results.zip` when the loose files have been archived.

A job that either source calls finished is written as done. That is the safe
direction: wrongly marking a finished job pending costs a re-run, while wrongly
marking a pending job done silently skips it -- so the reconstruction is
deliberately biased toward "it ran", and `--report` prints the evidence per job
so the bias can be checked rather than trusted.

    python mexp/tools/rebuild_queue_state.py --line kimi --report
    python mexp/tools/rebuild_queue_state.py --line kimi --write
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DONE_RE = re.compile(r"^\[[\d:]+\]\s+(\S+)\s+(done|failed)\b(?:.*?wall=(\d+)s)?")


def from_runner_log(path):
    """id -> (status, wall_s) for every completion this runner logged."""
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path, errors="replace"):
        m = DONE_RE.match(line.strip())
        if m:
            out[m.group(1)] = (m.group(2), int(m.group(3)) if m.group(3) else None)
    return out


def from_artifacts(results_dir, ids):
    """ids whose tag appears in a result filename, loose or inside results.zip.

    Job tags are substrings of one another (`ruler-vestigekv` inside
    `ruler-vestigekv-n50-tgt`), so a bare `in` test would mark the shorter job
    done on the longer one's output. The tag must be delimited by `_` or `.`
    the way the writers spell it."""
    names = []
    for d, _, fs in os.walk(results_dir):
        for f in fs:
            names.append(f)
    zp = os.path.join(results_dir, "results.zip")
    if os.path.exists(zp):
        with zipfile.ZipFile(zp) as z:
            names += [os.path.basename(n) for n in z.namelist()]
    blob = "\n".join(names)
    found = set()
    for i in ids:
        if re.search(r"(?:^|[_/])" + re.escape(i) + r"(?:[._]|$)", blob, re.M):
            found.add(i)
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--line", default="kimi")
    ap.add_argument("--queue", default="", help="default: mexp/<line>/queue.jsonl")
    ap.add_argument("--write", action="store_true", help="write queue_state.jsonl")
    ap.add_argument("--merge", action="store_true",
                    help="append to an existing queue_state.jsonl instead of refusing. "
                         "The runner recreates the file the moment it starts a job, so "
                         "after a loss the file usually exists again with one record in "
                         "it -- refusing there would leave the whole queue re-running.")
    ap.add_argument("--report", action="store_true", help="print the evidence per job")
    ap.add_argument("--reopen", default="", metavar="ID[,ID...]",
                    help="drop these ids' records so the runner runs them again. For a job "
                         "that failed for a reason that was not its own -- a server killed "
                         "out from under it, a machine reboot. The runner treats failed as "
                         "final, which is right for a job that failed on its own merits and "
                         "wrong here, and there is no other way to undo it.")
    args = ap.parse_args()

    queue = args.queue or os.path.join(ROOT, "mexp", args.line, "queue.jsonl")
    results = os.path.join(ROOT, "results", args.line)
    runner_log = os.path.join(ROOT, "results", f"{args.line}_queue_runner.log")
    state = os.path.join(results, "queue_state.jsonl")

    jobs = [json.loads(l) for l in open(queue) if l.strip()]
    ids = [j["id"] for j in jobs]

    if args.reopen:
        want = {x.strip() for x in args.reopen.split(",") if x.strip()}
        if not os.path.exists(state):
            print(f"no {state} to reopen from", file=sys.stderr)
            return 1
        # Read fully before writing: this file is the queue's only memory.
        recs = [json.loads(l) for l in open(state) if l.strip()]
        keep = [r for r in recs if r.get("id") not in want]
        dropped = len(recs) - len(keep)
        unknown = want - {r.get("id") for r in recs}
        if not args.write:
            print(f"  would drop {dropped} record(s) for {sorted(want)}"
                  + (f"; not present: {sorted(unknown)}" if unknown else ""))
            print("\n(dry run; pass --write)")
            return 0
        with open(state, "w") as f:
            for r in keep:
                f.write(json.dumps(r) + "\n")
        print(f"  dropped {dropped} record(s); {len(keep)} remain"
              + (f"; not present: {sorted(unknown)}" if unknown else ""))
        return 0

    logged = from_runner_log(runner_log)
    arts = from_artifacts(results, ids)

    recs, n_log, n_art, pending = [], 0, 0, []
    for i in ids:
        src = []
        status, wall = None, None
        if i in logged:
            status, wall = logged[i]
            src.append("runner-log")
            n_log += 1
        if i in arts:
            src.append("artifact")
            n_art += 1
            status = status or "done"
        if status is None:
            pending.append(i)
            if args.report:
                print(f"  PENDING  {i}")
            continue
        rec = {"id": i, "status": status, "note": "reconstructed by "
               "mexp/tools/rebuild_queue_state.py from " + "+".join(src)}
        if wall is not None:
            rec["wall_s"] = wall
        recs.append(rec)
        if args.report:
            print(f"  {status:6s}   {i}  [{'+'.join(src)}]")

    print(f"\n{len(ids)} jobs: {len(recs)} finished, {len(pending)} pending")
    print(f"  evidence: runner log {n_log}, artifacts {n_art}, "
          f"both {n_log + n_art - len(recs) if recs else 0}")
    if pending:
        print(f"  pending: {', '.join(pending[:8])}{' ...' if len(pending) > 8 else ''}")

    if args.write:
        exists = os.path.exists(state)
        if exists and not args.merge:
            print(f"\nrefusing to overwrite existing {state}; pass --merge", flush=True)
            return 1
        os.makedirs(results, exist_ok=True)
        have = set()
        if exists:
            # Read fully before opening for append: the runner owns this file and
            # its records are the authority for anything it ran itself.
            have = {json.loads(l)["id"] for l in open(state) if l.strip()}
        add = [r for r in recs if r["id"] not in have]
        with open(state, "a" if exists else "w") as f:
            for r in add:
                f.write(json.dumps(r) + "\n")
        print(f"\n{'appended to' if exists else 'wrote'} {state}: "
              f"{len(add)} records added, {len(have)} already present")
    else:
        print("\n(dry run; pass --write to create the file)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
