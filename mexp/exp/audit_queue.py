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
  declared seed     a client that takes --seed and a job that does not set it,
                    or one that sets a value other than the line's. A default
                    is not a declaration: it is upstream's choice, silently,
                    for as long as upstream keeps it.
  known client      a client the runner does not implement -- it would skip
                    the job and mark it done.
  registered        a job whose launch is not written in README.md verbatim.
                    The rule has always been that an experiment is registered
                    before it runs; until now the only thing enforcing it on
                    the queue -- which is what actually launches experiments --
                    was remembering to. Registering the name is not enough
                    either: the knobs can be edited after the name is written
                    down and the name still matches. The canonical launch is
                    compared, so the row that runs is the row that was
                    declared.
  graph width       a job whose captured graph is wider than the batch it
                    decodes. Checking that the two ARMS agree is not enough --
                    they did agree, at GRAPH_BS=2 against MAX_REQS=1, and that
                    pair's vestigekv arm came out 3.7% slower at 256k than two
                    independent records taken at the matched width while its
                    dense arm reproduced them. A line has a convention and a
                    job has to match the line, not just its partner.
  radix off         a performance-line job that turns the radix cache on. A
                    reused prefix is an answer attention never produced, and
                    this line's claim is about what attention costs. The knob
                    also has to agree across runs, not just within a pair: the
                    latency pair that was discarded differed from its partner
                    in this knob among three.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UNIFIED = os.path.join(ROOT, "engine")
def _runner_clients():
    """The clients the runner implements, read from the runner.

    A hand-kept list drifts: this one was missing longbench2 and profile, so
    two jobs the runner runs fine would have been refused as unimplemented."""
    src = open(os.path.join(ROOT, "mexp", "glm53", "queue_runner.py")).read()
    return set(re.findall(r'client == "(\w+)"', src))


CLIENTS = _runner_clients()

from naming import ruler_out, stream_out  # noqa: E402  (same rules as the runner)
SEEDED = {"ruler", "stream"}  # clients whose runner call takes --seed
# One seed for every run on every line, and it is the first one: a paper that
# reports seed 7 has to answer what seeds 0 through 6 did, and the honest answer
# -- nothing, we never ran them -- is not available to a reader. Seed 0 costs
# nothing and removes the question.
SEED = 0


REGISTRY = "<!-- registered launches: audited verbatim against mexp/*/queue.jsonl -->"


def canonical(job):
    """The launch, with nothing that is not the launch. Key order and the
    runner's own bookkeeping must not be able to make two identical runs look
    different, or two different ones look identical."""
    keep = {k: job[k] for k in ("id", "arm", "client", "env", "args", "server_args", "probe")
            if k in job}
    return json.dumps(keep, sort_keys=True, separators=(",", ":"))


def registered(readme):
    """Every canonical launch written down in README.md, by id."""
    out = {}
    for line in readme.splitlines():
        line = line.strip().strip("`")
        if not line.startswith("{"):
            continue
        try:
            job = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "id" in job:
            out[job["id"]] = canonical(job)
    return out


def load(path):
    out = []
    for line in open(path):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


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

    readme = open(os.path.join(ROOT, "README.md")).read()
    readme_canon = registered(readme)

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
        # Registering the NAME is not registering the experiment: the knobs can
        # be edited afterwards and the name still matches. What has to appear in
        # README.md is the launch itself, canonicalised, so the row that runs
        # and the row that was declared are the same row.
        if readme_canon.get(j["id"]) != canonical(j):
            bad(j, f"its launch is not registered verbatim in README.md. Declared "
                   f"there: {readme_canon.get(j['id'], '(nothing for this id)')}\n"
                   f"        about to run: {canonical(j)}")
        conc = int(j.get("args", {}).get("concurrency", 1))
        if conc > int(env.get("MAX_REQS", 1) or 1):
            bad(j, f"asks for concurrency {conc} from a server admitting "
                   f"{env.get('MAX_REQS')}; the batch it reports would not be the "
                   "batch it ran")
        if env.get("GRAPH_BS") and env.get("MAX_REQS") and env["GRAPH_BS"] != env["MAX_REQS"]:
            bad(j, f"captures a graph for bs={env['GRAPH_BS']} but decodes at "
                   f"bs={env['MAX_REQS']}; every other job on every line sets them "
                   "equal, and the surplus lane is dispatch VestigeKV's per-step "
                   "kernels pay and the dense path does not")
        if str(env.get("RADIX", "")).lower() == "on":
            bad(j, "RADIX=on; every line runs radix off (see the RADIX POLICY block "
                   "in mexp/exp/latency-pair-512k.sh). The paper states it as a "
                   "property of all of them, so one job with it on makes that false")
        if j.get("client") in SEEDED and "seed" not in j.get("args", {}):
            bad(j, "takes --seed but the job does not set one; a default is not a "
                   "declaration, and upstream's is 42 only for as long as upstream "
                   "keeps it")
        if j.get("args", {}).get("seed", SEED) != SEED:
            bad(j, f"declares seed {j['args']['seed']}; this line runs seed {SEED} so "
                   "any two of its records are comparable")
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
