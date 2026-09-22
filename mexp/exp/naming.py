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
    args, arm, env = job.get("args", {}), job["arm"], job.get("env", {})
    tag = "_stats" if str(env.get("SGLANG_DEBUG_VESTIGEKV_STATS", "0")) == "1" else ""
    # Anything that changes what the backend does has to change the record's
    # name. Flags did; debug env keys did not, so fbstream-on and fbstream-off
    # -- identical but for SGLANG_DEBUG_VESTIGEKV_NO_OVERFLOW_FALLBACK -- both
    # claimed latency_stream_4k-126976_vestigekv_stats.jsonl.
    ablating = [k for k in env if k.startswith("SGLANG_DEBUG_VESTIGEKV_")
                and k != "SGLANG_DEBUG_VESTIGEKV_STATS"]
    if job.get("server_args") or ablating:
        tag += "_" + job["id"]
    return os.path.join(results,
                        f"latency_stream_{stream_shape(args)}_{arm}{_checkpoint(job)}{tag}.jsonl")


def _checkpoint(job):
    """A marker for a non-default checkpoint, or "" for the line's own.

    Without it a Base run and an Instruct run at the same n and lengths write
    one file, and the second silently becomes the first. The paper compares
    the two checkpoints in a table, so both exist by design.
    """
    model = job.get("env", {}).get("MODEL", "")
    return "_" + model.rsplit("-", 1)[-1].lower() if model else ""


def ruler_out(results, job):
    args = job.get("args", {})
    lens = args.get("lengths", "4096,8192,16384,32768,65536").split(",")
    tag = f"{job['arm']}{_checkpoint(job)}_n{args.get('n', 10)}_{'-'.join(lens)}"
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
        # RuntimeError, not SystemExit: the runner catches Exception, and
        # SystemExit is not one. Raising it here took the whole queue down
        # while correctly refusing a single job.
        raise RuntimeError(
            f"ABORT: {job_id} would write {os.path.basename(path)}, which exists. "
            "The client appends, so this would put two measurements in one "
            "record. If this process predates a naming change, restart it.")


def tree_epoch(root):
    """When the engine tree's current lineage began -- its rebase point.

    A job is skipped because `queue_state.jsonl` says done, and that row does
    not know which tree produced it. Two LongBench v2 runs from three days
    before the v0.5.20 rebase were skipped for exactly that reason: their ids
    were registered, so the queue believed them covered, and their records
    could not be reproduced by anything shipped.

    So `done` means done on this tree. The question is which TREE produced a
    result, not which commit, and those are different: the first version of
    this read the HEAD commit's date, so committing a CPU-only test to the
    engine moved the epoch to now and reclassified every result of that day as
    stale. Sixty-three finished jobs were queued for a re-run by a one-line
    test file.

    The rebase point does not move when a commit is added on top, which is the
    property wanted: results are invalidated by the history being rewritten
    under them, not by the tree growing.
    """
    import subprocess

    # The tree being SERVED, not the one the submodule happens to sit on:
    # common.sh takes the engine from ENGINE when it is set, and staleness has
    # to be judged against the checkout a run actually loaded. Latent rather
    # than observed -- today both branches share their rebase point, so both
    # answer the same epoch and nothing has diverged yet. It bites the first
    # time one of them rebases onto a different tag, and then it bites
    # silently, by calling a result current because some other branch did not
    # move.
    engine = os.environ.get("ENGINE") or os.path.join(root, "engine")
    out = subprocess.run(
        ["git", "-C", engine, "log", "v0.5.20..HEAD",
         "--format=%cI"], capture_output=True, text=True)
    dates = [l for l in out.stdout.splitlines() if l.strip()]
    return dates[-1][:19].replace("T", " ") if out.returncode == 0 and dates else ""
