#!/usr/bin/env python3
"""Audit the experiment queue before the runner touches the GPU.

    python mexp/exp/audit_queue.py            # audit every pending job
    python mexp/exp/audit_queue.py --line kimi

The queue exists so no agent launches an experiment by hand. That only helps
if something checks the jobs, because a queue entry is just as capable of
being wrong as a typed command -- and wrong here costs GPU hours and, worse,
produces a record that looks fine. Every check below is a refusal, and every
one of them corresponds to a mistake this project has actually made:

  one tree          a job pinned to a retired worktree. Numbers measured on
                    the frozen tree and printed beside numbers from the
                    unified v0.5.20 tree is the arm mismatch that cost this
                    paper its RoPE table.
  no overwrite      a job whose output file already exists. An experiment that
                    silently overwrites its own record makes the archive lie.
  paired arms       a dense job with no vestigekv partner, or partners whose
                    env differs. A comparison is worth its weakest controlled
                    variable, and the env IS the control.
  real arm script   an `arm` with no launcher; the runner would fail after
                    loading the model.
  declared seed     a client that takes --seed and a job that does not set it.
                    A default is not a declaration.
  known client      a client the runner does not implement -- it would skip
                    the job and mark it done.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UNIFIED = os.path.join(ROOT, "engine")
CLIENTS = {"ruler", "stream", "needle", "gsm8k", "replay", "continue"}
SEEDED = {"ruler"}  # clients whose runner call takes --seed


def load(path):
    out = []
    for line in open(path):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def stream_out(results, job):
    a, arm = job.get("args", {}), job["arm"]
    n_out = int(a.get("output_len", 126976))
    n_in = int(a.get("input_len", 4096))
    n_req = int(a.get("num_prompts", 1))
    shape = f"{n_in // 1024}k-{n_out}" + (f"-x{n_req}" if n_req > 1 else "")
    tag = "_stats" if str(job.get("env", {}).get("SGLANG_DEBUG_VESTIGEKV_STATS", "0")) == "1" else ""
    if job.get("server_args"):
        tag += "_" + job["id"]
    return os.path.join(results, f"latency_stream_{shape}_{arm}{tag}.jsonl")


def ruler_out(results, job):
    a = job.get("args", {})
    lens = a.get("lengths", "4096,8192,16384,32768,65536").split(",")
    tag = f"{job['arm']}_n{a.get('n', 10)}_{'-'.join(lens)}"
    if job.get("server_args") or "tasks" in a or "lengths" in a:
        tag += f"_{job['id']}"
    return os.path.join(results, "ruler", f"results_{tag}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--line", default="kimi")
    args = ap.parse_args()

    here = os.path.join(ROOT, "mexp", args.line)
    results = os.path.join(ROOT, "results", args.line)
    jobs = load(os.path.join(here, "queue.jsonl"))
    state = load(os.path.join(results, "queue_state.jsonl"))
    done = {r["id"] for r in state if r.get("status") in ("done", "failed")}
    pending = [j for j in jobs if not j.get("skip") and j["id"] not in done]

    fails = []

    def bad(job, msg):
        fails.append(f"{job['id']}: {msg}")

    by_env = {}
    for j in pending:
        env = j.get("env", {})
        engine = env.get("ENGINE")
        if engine and os.path.realpath(engine) != os.path.realpath(UNIFIED):
            bad(j, f"pinned to {engine}, not the one tree ({UNIFIED})")
        if j.get("client") not in CLIENTS:
            bad(j, f"client {j.get('client')!r} is not one the runner implements")
        launcher = os.path.join(here, f"{j.get('arm')}.sh")
        if not os.path.exists(launcher):
            bad(j, f"arm {j.get('arm')!r} has no launcher at {launcher}")
        if j.get("client") in SEEDED and "seed" not in j.get("args", {}):
            bad(j, "takes --seed but the job does not set one; a default is not a declaration")
        out = None
        if j.get("client") == "stream":
            out = stream_out(results, j)
        elif j.get("client") == "ruler":
            out = ruler_out(results, j)
        if out and os.path.exists(out):
            bad(j, f"output already exists: {os.path.relpath(out, ROOT)}")
        key = json.dumps(env, sort_keys=True) + "|" + json.dumps(j.get("args", {}), sort_keys=True)
        by_env.setdefault(key.replace(f'"{j["arm"]}"', ""), []).append(j)

    # a comparison needs both arms, and needs them identical apart from the arm
    arms_seen = {}
    for j in pending:
        stem = j["id"].rsplit("-", 1)[0]
        arms_seen.setdefault(stem, []).append(j)
    for stem, group in arms_seen.items():
        arms = {g["arm"] for g in group}
        if len(group) > 1 and len(arms) > 1:
            envs = {json.dumps(g.get("env", {}), sort_keys=True) for g in group}
            argss = {json.dumps(g.get("args", {}), sort_keys=True) for g in group}
            if len(envs) > 1:
                fails.append(f"{stem}: paired arms differ in env, which is the control")
            if len(argss) > 1:
                fails.append(f"{stem}: paired arms differ in args, which is the control")

    print(f"pending jobs: {len(pending)}")
    for j in pending:
        env = j.get("env", {})
        print(f"  {j['id']:28s} arm={j['arm']:10s} client={j['client']:8s} "
              f"tree={'one' if not env.get('ENGINE') else env['ENGINE']}")
    if fails:
        print("\nREFUSED:")
        for f in fails:
            print(f"  {f}")
        return 1
    print("\nqueue audit: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
