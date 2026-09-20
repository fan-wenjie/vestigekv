#!/usr/bin/env python3
"""Initialise an experiment queue from the registry in README.md.

    python mexp/exp/init_queue.py --line kimi            # show what it would write
    python mexp/exp/init_queue.py --line kimi --write    # write queue.jsonl

WHICH FILE IS THE EXPERIMENT SET. README.md is. The `registered launches` block
there is the set of experiments this paper needs run, written as the exact
launches; `mexp/<line>/queue.jsonl` is what the runner consumes and is derived
from it. Before this script the two were maintained by hand in parallel and
audit_queue.py compared them -- which catches a divergence but still asks a
person to fix it in two places, and the person is the reason there was a
divergence.

So: edit README.md, run this, run the audit, start the runner. A queue row
that nobody wrote in the registry cannot survive an --write, and a registry
entry nobody queued cannot be forgotten.

WHAT IT REFUSES. Editing the queue out from under a live runner. Adding rows
while one runs is supported and normal -- it re-reads the file between jobs,
and the write here is atomic, so it sees either the old file or the new one.
Removing or editing a row is not: that is how a job in flight becomes a job
the queue no longer admits to having started, and how this project lost its
queue state once already.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_queue import canonical, load, registered  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def running(state_path, known):
    """Ids whose LAST recorded status is 'running'.

    Last, not "started and never finished": an id can appear many times --
    abandoned, then started again -- and set arithmetic cancels the restart
    against the abandonment, which hides exactly the job that is live now.

    Restricted to ids the queue or the registry knows about, because the state
    file is append-only history and carries orphaned 'running' rows from runs
    that were killed long ago. One of those must not block the queue forever.
    """
    if not os.path.exists(state_path):
        return set()
    last = {}
    for row in load(state_path):
        if "id" in row and "status" in row:
            last[row["id"]] = row["status"]
    return {i for i, s in last.items() if s == "running" and i in known}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--line", default="kimi")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    queue = os.path.join(ROOT, "mexp", a.line, "queue.jsonl")
    state = os.path.join(ROOT, "results", a.line, "queue_state.jsonl")
    reg = registered(open(os.path.join(ROOT, "README.md")).read())
    if not reg:
        raise SystemExit("ABORT: README.md has no registered launches")

    rows = [json.loads(v) for v in reg.values()]
    have = {j["id"]: canonical(j) for j in load(queue)} if os.path.exists(queue) else {}
    live = running(state, set(have) | set(reg))

    added = [r["id"] for r in rows if r["id"] not in have]
    changed = [r["id"] for r in rows if r["id"] in have and have[r["id"]] != canonical(r)]
    removed = [i for i in have if i not in reg]

    print(f"registry: {len(rows)} launches   queue: {len(have)} rows")
    for label, ids in (("add", added), ("rewrite", changed), ("remove", removed)):
        for i in ids:
            print(f"  {label:>8}  {i}")
    if not (added or changed or removed):
        print("  (queue already matches the registry)")

    stuck = live & set(removed + changed)
    if stuck:
        raise SystemExit(
            f"ABORT: {sorted(stuck)} started and has not finished. Let it finish or "
            "record it as abandoned in queue_state.jsonl first -- a job cannot be "
            "edited out from under the runner that is running it.")
    if a.write and live and (removed or changed):
        raise SystemExit(
            f"ABORT: the runner is mid-job on {sorted(live)} and this write is not "
            "purely additive. Adding rows under a live runner is the supported "
            "pattern -- it re-reads the file between jobs -- but removing or "
            "editing one is not.")
    if a.write:
        tmp = queue + ".tmp"
        with open(tmp, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        os.replace(tmp, queue)  # atomic: never a half file for the runner to read
        print(f"\nwrote {os.path.relpath(queue, ROOT)} ({len(rows)} rows)")
        print("next: python mexp/exp/audit_queue.py && "
              "python mexp/glm53/queue_runner.py --line " + a.line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
