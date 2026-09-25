#!/usr/bin/env python3
"""Re-materialise a log file whose writer still holds it open after deletion.

A process that has a file open keeps writing to the inode after the path is
unlinked. It does not crash and it reports no error; the output simply stops
appearing at the path, and every reader -- a health check, a tail, a person --
sees a log frozen at the moment of deletion while the job it describes runs on.
That is how deleting `results/` under a live queue runner hides the queue from
its own monitoring.

While the descriptor is open the data is still readable at `/proc/<pid>/fd/<n>`,
so this copies it back. It is a recovery tool, not a fix: new writes keep going
to the unlinked inode, so the path goes stale again the moment this returns.
Only restarting the writer restores a self-maintaining log, and for the queue
runner that means interrupting the queue -- so the working arrangement is to run
this before reading, which is what the health check does.

    python mexp/tools/sync_runner_log.py --line kimi
    python mexp/tools/sync_runner_log.py --all      # every deleted log any
                                                    # runner/server still holds
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WRITER_PATTERNS = ("queue_runner.py", "sglang.launch_server", "run_ruler.py")


def writer_pids():
    """pids of the processes that write the logs we care about."""
    pids = []
    for pat in WRITER_PATTERNS:
        out = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True).stdout
        pids += [int(x) for x in out.split()]
    return sorted(set(pids))


def deleted_fds(pid):
    """[(fd_path, original_path)] for this pid's deleted regular files."""
    found = []
    for fd in glob.glob(f"/proc/{pid}/fd/*"):
        try:
            target = os.readlink(fd)
        except OSError:
            continue
        m = re.match(r"^(.*) \(deleted\)$", target)
        if m and m.group(1).startswith("/"):
            found.append((fd, m.group(1)))
    return found


def restore(fd_path, dest, dry):
    """Copy the open descriptor's contents back to its path.

    Read the whole thing before opening the destination: the destination may be
    the same path, and opening it for write first would truncate what is being
    read -- the exact shape of an earlier data loss here."""
    try:
        with open(fd_path, "rb") as f:
            data = f.read()
    except OSError as e:
        return None, f"unreadable: {e}"
    if dry:
        return len(data), "would write"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    existing = b""
    if os.path.exists(dest):
        existing = open(dest, "rb").read()
    if existing == data:
        return len(data), "unchanged"
    # Never shrink a path that already holds more than the descriptor offers:
    # a second writer, or an already-restored longer copy, would be truncated.
    if len(existing) > len(data):
        return len(data), f"skipped, path has more ({len(existing)} B)"
    with open(dest, "wb") as f:
        f.write(data)
    return len(data), f"+{len(data) - len(existing)} B"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--line", default="", help="restrict to results/<line>* paths")
    ap.add_argument("--all", action="store_true", help="every deleted log, any path")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pids = writer_pids()
    if not pids:
        print("no runner/server processes found", file=sys.stderr)
        return 1

    n = 0
    seen = set()
    for pid in pids:
        for fd, dest in deleted_fds(pid):
            if dest in seen:
                continue
            if not args.all:
                if not dest.startswith(os.path.join(ROOT, "results")):
                    continue
                if args.line and f"/{args.line}" not in dest:
                    continue
            seen.add(dest)
            size, note = restore(fd, dest, args.dry_run)
            rel = os.path.relpath(dest, ROOT) if dest.startswith(ROOT) else dest
            print(f"  pid {pid:>7}  {size if size is not None else '-':>10} B  {note:<28} {rel}")
            n += 1
    if not n:
        print("nothing deleted-but-open matched; logs are live")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
