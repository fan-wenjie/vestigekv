#!/usr/bin/env python3
"""Rebuild results/results.zip from the run records on disk.

The archive was assembled by hand, which is why its contents drifted from the
directory twice and why one category was dropped on a rationale that later
turned out to be false. The rules belong in code where they can be read and
argued with:

**Excluded, and why.**
  `caldump/`          7.1 GB of activation dumps; the paper says they are
                      excluded and names the script that regenerates them.
  `*/ruler/samples_*.json`
                      raw RULER generations, 188 MB per arm. NOT dropped as
                      worthless -- the paired test needs their per-item scores,
                      and `mexp/tools/reduce_ruler_samples.py` puts exactly
                      those in `paired_scores.json`, which IS archived. Keep the
                      originals on disk; the archive carries the projection.
                      The rule is scoped to `ruler/` on purpose: the LongBench
                      samples next door are 75 KB and `make_lb2_numbers.py`
                      reads them per question, so a bare `samples_*` glob takes
                      out a load-bearing file. The archive-may-only-grow check
                      is what caught that.
  `health.log`, `*_queue_runner.log`
                      the runner's own monitoring. It describes how the queue
                      was driven, not what any job measured, and neither has
                      ever been in the archive. `queue_state.jsonl` is NOT in
                      this list: `rebuild_queue_state.py` reads it out of the
                      archive to reconstruct what has run.
  `*.png`, `README.md`, `results.zip`
                      kept beside the archive so they render without unpacking.

Everything else goes in, including the server and client logs: several macros
are extracted from `server_*.log` VKSTATS lines and from the `continue_*.log`
summary lines, and a deletion of that category has already cost this paper nine
cited numbers once.

**The archive may only grow.** A rebuild that would drop a path already in the
archive stops, because that is what both losses looked like. Pass `--allow-drop`
with the paths to remove one deliberately.

    python mexp/tools/build_results_archive.py            # dry run: print the plan
    python mexp/tools/build_results_archive.py --write
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "results")
ARCHIVE = os.path.join(RESULTS, "results.zip")

EXCLUDE_DIRS = {"caldump", ".git"}
# Matched against the basename.
EXCLUDE_GLOBS = ("health.log", "*_queue_runner.log", "*.png", "README.md",
                 "*.zip")
# Matched against the path relative to results/, so the rule can name a
# directory. Scoping matters here -- see the module docstring.
EXCLUDE_PATHS = ("*/ruler/samples_*.json",)


def included():
    out = []
    for r, ds, fs in os.walk(RESULTS):
        ds[:] = [d for d in ds if d not in EXCLUDE_DIRS]
        for f in fs:
            rel = os.path.relpath(os.path.join(r, f), RESULTS)
            if any(fnmatch.fnmatch(f, g) for g in EXCLUDE_GLOBS):
                continue
            if any(fnmatch.fnmatch(rel, g) for g in EXCLUDE_PATHS):
                continue
            out.append(rel)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--allow-drop", nargs="*", default=[],
                    help="paths that may disappear from the archive")
    args = ap.parse_args()

    # Pack before listing, so a record that still carries raw per-token arrays
    # is packed rather than shipped at twenty times its size -- or, worse,
    # reduced by hand later and lost. Safe here and nowhere else: this runs
    # only with the queue stopped, checked below.
    if args.write:
        subprocess.run([sys.executable,
                        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "pack_streams.py"),
                        "--glob", os.path.join(RESULTS, "*", "latency_stream_*.jsonl"),
                        "--write"], check=True)

    want = included()
    have = []
    if os.path.exists(ARCHIVE):
        have = sorted(zipfile.ZipFile(ARCHIVE).namelist())

    added = [p for p in want if p not in have]
    dropped = [p for p in have if p not in want and p not in args.allow_drop]

    print(f"on disk: {len(want)} files, {sum(os.path.getsize(os.path.join(RESULTS, p)) for p in want) / 1e6:.1f} MB")
    print(f"archive: {len(have)} files")
    for p in added:
        print(f"  + {p}")
    for p in dropped:
        print(f"  - {p}   <-- WOULD BE LOST")
    if dropped:
        print("\nrefusing to write: the archive may only grow. Name the paths in "
              "--allow-drop if the removal is deliberate.", file=sys.stderr)
        return 1
    if not args.write:
        print("\ndry run; pass --write to rebuild")
        return 0

    # A live runner writes into this directory, and rebuilding while it does
    # captures a half-written record -- the same directory-under-a-live-runner
    # hazard that cost a queue once.
    if subprocess.run(["pgrep", "-f", "[q]ueue_runner.py"],
                      capture_output=True).returncode == 0:
        print("queue runner is alive; not rebuilding the archive", file=sys.stderr)
        return 1

    tmp = ARCHIVE + ".new"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in want:
            z.write(os.path.join(RESULTS, p), p)
        # The archive carries its own reader: packed arrays are lzma, which is
        # stdlib, so this is one file and no install.
        z.write(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "unpack_streams.py"), "unpack_streams.py")
    os.replace(tmp, ARCHIVE)
    print(f"\nwrote {ARCHIVE}: {len(want)} files, "
          f"{os.path.getsize(ARCHIVE) / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
