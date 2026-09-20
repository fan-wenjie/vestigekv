"""Where a job's record lands. One implementation, imported by both users.

The runner names the file it writes and the auditor names the file it checks
for. While those were two copies of the same rules they could disagree, and on
2026-09-20 they did: `rep` was added to the auditor and to the runner source,
but the runner process had been started before the edit, so it kept computing
the pre-`rep` name. The auditor cleared `tput32-bs4r2-dense` because
`...-x4-c4-r2_baseline.jsonl` did not exist; the running process wrote
`...-x4-c4_baseline.jsonl`, which did, and the client appended -- so a repeat
landed inside the record of the point it was repeating.

A shared module does not fix a stale process. What fixes that is the runner
refusing to write over an existing record, which is also in this file.
"""
from __future__ import annotations

import os


def stream_shape(args):
    """The part of a stream record's name that describes the measurement."""
    n_in = int(args.get("input_len", 4096))
    n_out = int(args.get("output_len", 126976))
    n_req = int(args.get("num_prompts", 1))
    conc = int(args.get("concurrency", 1))
    shape = f"{n_in // 1024}k-{n_out}"
    if n_req > 1:
        shape += f"-x{n_req}"
    if conc > 1:
        shape += f"-c{conc}"
    if args.get("rep"):
        shape += f"-r{int(args['rep'])}"
    return shape


def stream_out(results, job):
    args, arm = job.get("args", {}), job["arm"]
    tag = "_stats" if str(job.get("env", {}).get(
        "SGLANG_DEBUG_VESTIGEKV_STATS", "0")) == "1" else ""
    if job.get("server_args"):
        tag += "_" + job["id"]
    return os.path.join(results, f"latency_stream_{stream_shape(args)}_{arm}{tag}.jsonl")


def ruler_out(results, job):
    args = job.get("args", {})
    lens = args.get("lengths", "4096,8192,16384,32768,65536").split(",")
    tag = f"{job['arm']}_n{args.get('n', 10)}_{'-'.join(lens)}"
    if job.get("server_args") or "tasks" in args or "lengths" in args:
        tag += f"_{job['id']}"
    return os.path.join(results, "ruler", f"results_{tag}.json")


def refuse_existing(path, job_id):
    """The runner's own last line of defence.

    The auditor runs before the runner starts and cannot see what a process
    already running will compute. This check is inside the process that writes,
    so it holds even when that process is older than the rules it is following.
    """
    if os.path.exists(path):
        raise SystemExit(
            f"ABORT: {job_id} would write {os.path.basename(path)}, which exists. "
            "The client appends, so this would put two measurements in one "
            "record. If this process predates a naming change, restart it.")
