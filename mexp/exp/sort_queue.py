"""Reorder mexp/<line>/queue.jsonl shortest-job-first.

    python mexp/exp/sort_queue.py [--line kimi] [--write]

WHY THE FILE AND NOT THE RUNNER. queue_runner takes jobs[0] and re-reads the
file before every job, so file order IS the schedule and a sort is the whole
implementation. Putting the policy in the runner would hide it from the audit,
which reads the same file.

THE KEY IS MEASURED, NOT ESTIMATED. An id's cost is the wall_s of its last
SUCCEEDING row in queue_state.jsonl -- these jobs have run before, so their
durations are recorded rather than guessed from input_len x num_prompts, which
misprices a ruler job against a stream one by an order of magnitude.

Only a `done` row counts. A failed row's wall_s is time-to-failure, which is
short for the wrong reason: taking it would promote the jobs most likely to
fail again straight to the front. diag-long128k-bs4-dense failed in 107 s and
sorted fifth of 98 before this rule. An id with no succeeding run therefore has
an UNKNOWN cost and sorts LAST, which is also where a job that has only ever
failed belongs when the point is to finish the most work in a fixed window.

WHAT IT COSTS. The runner keeps a server alive between consecutive jobs with the
same arm and env; sorting by duration interleaves arms and pays a restart
(~50 s) where grouping would not. That is the trade: with the queue longer than
the window, finishing the most jobs matters more than the restarts, and paired
arms of one config have similar durations so they stay adjacent anyway.

Lines are preserved byte for byte -- only their order changes -- so a re-serialize
cannot quietly rewrite a job's keys.
"""

import argparse
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def read_raw(path):
    """(parsed, raw_line) pairs, skipping blanks; raw keeps the original bytes."""
    out = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                out.append((json.loads(s), s))
            except json.JSONDecodeError:
                out.append((None, s))
    return out


def measured_wall(state_path):
    """id -> wall_s of its last SUCCEEDING row. Failures carry no cost signal."""
    wall, failed = {}, set()
    for rec, _ in read_raw(state_path):
        if rec is None:
            continue
        if rec.get("status") == "done" and isinstance(rec.get("wall_s"), (int, float)):
            wall[rec["id"]] = rec["wall_s"]
            failed.discard(rec["id"])
        elif rec.get("status") == "failed":
            failed.add(rec["id"])          # a later failure retires the old cost
    for jid in failed:
        wall.pop(jid, None)
    return wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--line", default="kimi")
    ap.add_argument("--write", action="store_true", help="apply; default is a dry run")
    a = ap.parse_args()

    queue = os.path.join(ROOT, "mexp", a.line, "queue.jsonl")
    state = os.path.join(ROOT, "results", a.line, "queue_state.jsonl")
    wall = measured_wall(state)
    rows = read_raw(queue)

    UNKNOWN = float("inf")
    keyed = []
    for i, (rec, raw) in enumerate(rows):
        jid = rec.get("id") if rec else None
        keyed.append((wall.get(jid, UNKNOWN), i, jid, raw))
    keyed.sort(key=lambda t: (t[0], t[1]))  # stable: ties keep their file order

    known = [k for k in keyed if k[0] is not UNKNOWN]
    print(f"{len(rows)} jobs, {len(known)} with a measured cost, "
          f"{len(rows) - len(known)} unknown (sorted last)")
    if known:
        print(f"total {sum(k[0] for k in known) / 3600:.1f} h\n")
    cum = 0.0
    for w, _, jid, _ in keyed[:12]:
        cum += 0 if w is UNKNOWN else w
        print(f"  {jid:<34} {'?' if w is UNKNOWN else f'{w:>6.0f}s'}  cumulative {cum/3600:>5.2f} h")
    if len(keyed) > 12:
        print(f"  ... {len(keyed) - 12} more")

    if not a.write:
        print("\ndry run; pass --write to apply")
        return
    with open(queue, "w") as f:
        for _, _, _, raw in keyed:
            f.write(raw + "\n")
    print(f"\nwrote {queue}")


if __name__ == "__main__":
    main()
