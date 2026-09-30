#!/usr/bin/env python3
"""Which recorded results were served by the PROVISIONAL index, not the fitted one.

    python mexp/exp/check_builds.py [--line kimi] [--fail-on-any]

The async calibrated build runs on a worker that catches Exception and keeps
the provisional index serving -- deliberately, because a failed fit must not
take the server down, and the provisional index over-fetches rather than
under-recalling, so the output stays correct. The cost is that a broken build
is indistinguishable from a working one in every number the run reports except
fetch count and throughput.

That is not hypothetical. Five recorded Kimi runs turned out to be serving the
provisional index, from three unrelated bugs, and one of them (r2-32k-branch)
had about half its builds failing while its RULER v2 score went into a rebuttal
table as if it were the branch arm's. Nothing flagged it: the runner saw rc=0,
the client wrote a score, the audit passes on the queue and never reads a
server log.

So the gate belongs here, after the fact, over every log at once -- and it has
to report a RATE, not a count. A handful of failures at the start of a request
is the build racing prefill and is normal; a rate near the success count means
the arm never calibrated and the number is not the arm's.

VKZP is throttled to every 200th build (recall_tier._log_zp), so successes are
estimated as 200x its line count. That makes the rate approximate and its
direction safe: it under-states the failure rate, never over-states it.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

_FAIL = "async calibrated build failed"
_ZP_EVERY = 200  # _log_zp logs when _ZP_SEEN % 200 == 1


def scan(path):
    fails, zp = 0, 0
    causes = {}
    with open(path, errors="replace") as fh:
        for line in fh:
            if _FAIL in line:
                fails += 1
                m = re.search(r"build failed \(([^)]*)", line)
                if m:
                    causes[m.group(1)] = causes.get(m.group(1), 0) + 1
            elif "VKZP" in line:
                zp += 1
    return fails, zp * _ZP_EVERY, causes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--line", default="kimi")
    ap.add_argument("--fail-on-any", action="store_true",
                    help="exit 1 if any run had a single failed build")
    a = ap.parse_args()

    pat = os.path.join("results", a.line, "server_*.log")
    bad = []
    for path in sorted(glob.glob(pat)):
        fails, est, causes = scan(path)
        if not fails:
            continue
        job = os.path.basename(path)[len("server_"):-len(".log")]
        bad.append((job, fails, est, causes))

    if not bad:
        print(f"{a.line}: every run's calibrated builds installed")
        return 0

    print(f"{a.line}: {len(bad)} run(s) served the provisional index at least "
          f"once\n")
    print(f"  {'job':<40} {'failed':>7} {'~ok':>8} {'rate':>7}  cause")
    for job, fails, est, causes in sorted(bad, key=lambda r: -r[1]):
        rate = fails / (fails + est) if (fails + est) else 1.0
        top = max(causes, key=causes.get)
        print(f"  {job:<40} {fails:>7} {est:>8} {100 * rate:>6.1f}%  {top[:44]}")
    print("\n  A rate in the tens of percent means the arm largely did not "
          "calibrate:\n  its score is not that arm's score, and its fetch "
          "counts are the provisional\n  index's, which over-fetches by "
          "construction.")
    return 1 if a.fail_on_any else 0


if __name__ == "__main__":
    sys.exit(main())
