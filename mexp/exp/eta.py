#!/usr/bin/env python3
"""How long the pending queue will take, from measured wall times only.

    python mexp/exp/eta.py [--line kimi]

Three estimates were given in this project from a progress bar's instantaneous
rate and all three were wrong -- once by a factor of forty (a job reported as
"six hours remaining" finished in nine minutes, then the same job's rate went
back up and it took another hour). A tqdm ETA extrapolates the last few
seconds, and these jobs are not uniform: RULER sweeps several context lengths
in one run, so the rate swings by 3x within a job and every instantaneous
reading is a different job's worth of prediction.

So this reads finished jobs instead. A pending job is matched to completed
ones by (client, shape, arm-class), where shape is the setup name or the
length list and arm-class separates the branch arm because it is 1.3x slower
than dense at the same shape. With a match it reports the mean and the spread
of the matches. Without one it says so and scales from the nearest shape,
labelled EXTRAPOLATED, because an extrapolation a reader can see is worth more
than a number they cannot check.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "mexp", "exp"))


def shape(job):
    a = job.get("args", {})
    for k in ("setup", "lengths"):
        if a.get(k):
            return str(a[k])
    if job.get("client") == "stream":
        return f"in{a.get('input_len')}/out{a.get('output_len')}/c{a.get('concurrency')}"
    return "-"


def arm_class(job):
    env = job.get("env", {})
    if job.get("arm") == "baseline":
        return "dense"
    return "branch" if env.get("SGLANG_DEBUG_VESTIGEKV_BRANCH_ONLY") else "served"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--line", default="kimi")
    a = ap.parse_args()
    import naming

    here = os.path.join(ROOT, "mexp", a.line)
    results = os.path.join(ROOT, "results", a.line)
    jobs = [json.loads(l) for l in open(os.path.join(here, "queue.jsonl")) if l.strip()]
    epoch = naming.tree_epoch(ROOT)

    walls, done = {}, set()
    for l in open(os.path.join(results, "queue_state.jsonl")):
        try:
            r = json.loads(l)
        except ValueError:
            continue
        if r.get("status") in ("done", "failed") and r.get("end", "") >= epoch:
            done.add(r["id"])
        if r.get("status") == "done" and r.get("wall_s"):
            walls[r["id"]] = r["wall_s"]

    by_key = {}
    for j in jobs:
        w = walls.get(j["id"])
        if w:
            by_key.setdefault((j["client"], shape(j), arm_class(j)), []).append(w)

    pending = [j for j in jobs if not j.get("skip") and j["id"] not in done]
    total = 0.0
    unknown = []
    print(f"{'job':<20} {'client':<8} {'shape':<14} {'arm':<7} {'estimate':>10}  basis")
    for j in pending:
        key = (j["client"], shape(j), arm_class(j))
        hits = by_key.get(key)
        if hits:
            est = statistics.mean(hits)
            spread = (f"+/-{(max(hits) - min(hits)) / 60:.0f}m" if len(hits) > 1 else "")
            basis = f"{len(hits)} run(s) of the same shape and arm {spread}"
        else:
            # nearest shape, same client and arm: scale by context if both parse
            same = {k: v for k, v in by_key.items()
                    if k[0] == j["client"] and k[2] == key[2]}
            if not same:
                same = {k: v for k, v in by_key.items() if k[0] == j["client"]}
            if not same:
                unknown.append(j["id"])
                print(f"{j['id']:<20} {j['client']:<8} {shape(j):<14} {key[2]:<7} "
                      f"{'unknown':>10}  no completed job of this client")
                continue
            ref_key = sorted(same)[0]
            est = statistics.mean(same[ref_key])
            ctx_of = lambda s: next((int(x) for x in s.replace("-", " ").replace("_", " ")
                                     .split() if x.isdigit()), None)
            c_new, c_ref = ctx_of(shape(j)), ctx_of(ref_key[1])
            if c_new and c_ref:
                est *= c_new / c_ref
            basis = f"EXTRAPOLATED from {ref_key[1]} ({ref_key[2]})"
        total += est
        print(f"{j['id']:<20} {j['client']:<8} {shape(j):<14} {key[2]:<7} "
              f"{est / 60:>8.0f} m  {basis}")
    print(f"\n{len(pending)} pending, about {total / 3600:.1f} h serial"
          + (f"; {len(unknown)} with no basis at all" if unknown else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
