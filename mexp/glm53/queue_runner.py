"""JSONL job queue for an experiment line (mexp/<line>/queue.jsonl): edit it, run this
with --line glm53 (default) or --line kimi.

One job per line of queue.jsonl:
  {"id": "ruler-baseline", "arm": "baseline", "client": "ruler",
   "env": {"CTX": "73728", "MAX_REQS": "4", "MAMBA_SLOTS": "4", "MEM_FRAC": "0.955",
           "CHUNK": "1024", "GRAPH_BS": "4"},
   "args": {"n": 10}, "probe": true, "skip": false}
arm: which mexp/glm53/<arm>.sh launches the server; env: its knobs; server_args:
extra sglang flags appended to that launch (e.g. ["--vestigekv-index-rank", "256"]); client:
ruler | stream | needle | gsm8k | replay (saved RULER prompts, args.samples/tasks/length); probe: run needle.py against the server before
the client. The queue file is re-read before every job, so lines can be added,
removed or reordered while a job runs; ids already in queue_state.jsonl (done or
failed) are not run again -- delete their state line to rerun. A server is kept
between consecutive jobs with the same arm and env.
"""

import json
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
PY = os.environ.get("PYTHON", os.path.expanduser("~/.conda/envs/sglang-dev/bin/python"))
# --line <name> (default glm53): jobs, arm scripts and results live under mexp/<name>/
# and results/<name>/; the line's model is what its clients are told to talk to.
LINE = sys.argv[sys.argv.index("--line") + 1] if "--line" in sys.argv else "glm53"
MODELS = {"glm53": "nvidia/GLM-5.3-Flash-NVFP4", "kimi": "moonshotai/Kimi-Linear-48B-A3B-Instruct"}
MODEL = MODELS[LINE]
HERE = os.path.join(ROOT, "mexp", LINE)
QUEUE = os.path.join(HERE, "queue.jsonl")
RESULTS = os.path.join(ROOT, "results", LINE)
STATE = os.path.join(RESULTS, "queue_state.jsonl")
SERVER_LOG = os.path.join(RESULTS, "server_{arm}_{job}.log")  # one log per job: a crash must stay readable


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip() and not line.lstrip().startswith("#")]


def append_state(rec):
    with open(STATE, "a") as f:
        f.write(json.dumps(rec) + "\n")


def gpu_used_mib():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    ).stdout
    return sum(int(x) for x in out.split())


def client_pids():
    out = subprocess.run(["ps", "-eo", "pid,comm,args"], capture_output=True, text=True).stdout
    return [int(l.split()[0]) for l in out.splitlines()[1:]
            if l.split()[1].startswith("python")
            and any(k in l for k in ("run_ruler.py", "sglang.benchmark.serving", "needle.py", "replay_prompts.py"))]


def server_pids():
    out = subprocess.run(["ps", "-eo", "pid,comm,args"], capture_output=True, text=True).stdout
    return [int(l.split()[0]) for l in out.splitlines()[1:]
            if l.split()[1].startswith("python") and "sglang.launch_server" in l]


class Server:
    def __init__(self):
        self.sig = None
        self.proc = None
        self.port = "30000"

    def stop(self):
        for pid in server_pids():
            subprocess.run(["kill", "-TERM", str(pid)])
        for _ in range(90):
            if gpu_used_mib() < 2000 and not server_pids():
                break
            time.sleep(2)
        self.sig = self.proc = None

    def ensure(self, arm, env, job_id, server_args=()):
        sig = (arm, json.dumps(env, sort_keys=True), tuple(server_args))
        if self.sig == sig and self.proc is not None and self.proc.poll() is None:
            return True
        self.stop()
        logpath = SERVER_LOG.format(arm=arm, job=job_id)
        os.makedirs(RESULTS, exist_ok=True)
        full_env = dict(os.environ, **{k: str(v) for k, v in env.items()})
        self.port = str(env.get("PORT", "30000"))
        with open(logpath, "w") as f:
            self.proc = subprocess.Popen(
                ["bash", os.path.join(HERE, f"{arm}.sh"), *server_args],
                stdout=f, stderr=subprocess.STDOUT, env=full_env, cwd=ROOT,
            )
        log(f"server {arm} env={env} args={list(server_args)} launching -> {logpath}")
        for _ in range(360):
            time.sleep(5)
            text = open(logpath, errors="replace").read()
            if "fired up" in text:
                cfg = [l for l in text.splitlines() if "max_total_num_tokens" in l and "TP1]" not in l]
                vk = [l for l in text.splitlines() if "VestigeKV:" in l and "TP1]" not in l]
                log(f"server {arm} ready: {cfg[:1]} {vk[:1]}")
                self.sig = sig
                return True
            if self.proc.poll() is not None:
                break
        err = [l for l in open(logpath, errors="replace").read().splitlines()
               if ("Error" in l or "error:" in l) and "TP1]" not in l][-3:]
        log(f"server {arm} FAILED: {err}")
        self.stop()
        return False


def run_client(job, port):
    client, args = job["client"], job.get("args", {})
    arm = job["arm"]
    env = dict(os.environ, OPENAI_API_KEY="dummy", PYTHONPATH=os.path.join(ROOT, "engine", "python"))
    env.pop("HF_HUB_OFFLINE", None)  # RULER pulls its corpora from the Hub
    env["CUDA_VISIBLE_DEVICES"] = ""  # clients are HTTP only; no CUDA context next to the server
    os.makedirs(RESULTS, exist_ok=True)
    if client == "ruler":
        cmd = [PY, os.path.join(ROOT, "mexp", "glm53", "run_ruler.py"), "--arm", arm, "--port", port,
               "--model", MODEL, "--n", str(args.get("n", 10)), "--out", os.path.join(RESULTS, "ruler")]
        if "lengths" in args:
            cmd += ["--lengths", args["lengths"]]
        if "tasks" in args:
            cmd += ["--tasks", args["tasks"]]
        if "seed" in args:
            cmd += ["--seed", str(args["seed"])]
        if job.get("server_args") or "tasks" in args or "lengths" in args:
            cmd += ["--tag", job["id"]]  # a sweep job must not overwrite the arm's full run
        out = os.path.join(RESULTS, f"ruler_{arm}_{job['id']}.log")
    elif client == "stream":
        # README metric 1: bs=1, 4k prefill, continuous decode; per-token latency curve.
        n_out = int(args.get("output_len", 126976))
        # a stats-on stream syncs every step: keep its file apart from the timed one
        tag = "_stats" if str(job.get("env", {}).get("SGLANG_DEBUG_VESTIGEKV_STATS", "0")) == "1" else ""
        if job.get("server_args"):
            tag += "_" + job["id"]  # a flag sweep must not overwrite the arm's default run
        n_in = int(args.get("input_len", 4096))
        n_req = int(args.get("num_prompts", 1))
        # Concurrency defaults to 1: every job written before this arg existed is
        # a latency curve and must keep measuring one request at a time. A
        # throughput sweep sets it to the batch it means, and the server's
        # captured graph has to be at least that wide (GRAPH_BS == MAX_REQS) or
        # the batch it reports is not the batch it ran.
        conc = int(args.get("concurrency", 1))
        shape = f"{n_in // 1024}k-{n_out}" + (f"-x{n_req}" if n_req > 1 else "")
        if conc > 1:
            shape += f"-c{conc}"
        out_jsonl = os.path.join(RESULTS, f"latency_stream_{shape}_{arm}{tag}.jsonl")
        cmd = [PY, "-m", "sglang.benchmark.serving", "--backend", "sglang", "--model", MODEL,
               # num_prompts > 1 makes this a controlled comparison against a
               # short-answer benchmark: same context, same generated length,
               # enough requests to accumulate steps. Default 1 = the latency curve.
               "--port", port, "--num-prompts", str(args.get("num_prompts", 1)), "--dataset-name", "random",
               "--random-input-len", str(args.get("input_len", 4096)), "--random-output-len", str(n_out),
               "--random-range-ratio", "1", "--max-concurrency", str(conc), "--warmup-requests", "0",
               "--output-details", "--output-file", out_jsonl]
        out = os.path.join(RESULTS, f"stream_{arm}_{job['id']}.log")
    elif client == "continue":
        # Continue a real novel: the natural-text column of the 2x2 against the
        # random-token stream, holding context and request count fixed.
        cmd = [PY, os.path.join(ROOT, "mexp", "kimi", "continue_text.py"),
               "--port", port, "--model", MODEL,
               "--input-len", str(args.get("input_len", 65536)),
               "--output-len", str(args.get("output_len", 14)),
               "--num-prompts", str(args.get("num_prompts", 130))]
        if "sub_domains" in args:
            cmd += ["--sub-domains", args["sub_domains"]]
        out = os.path.join(RESULTS, f"continue_{arm}_{job['id']}.log")
    elif client == "needle":
        cmd = [PY, os.path.join(ROOT, "mexp", "glm53", "needle.py"), str(args.get("reps", 330)), port]
        out = os.path.join(RESULTS, f"needle_{arm}_{job['id']}.log")
    elif client == "replay":
        cmd = [PY, os.path.join(ROOT, "mexp", "glm53", "replay_prompts.py"), "--port", port,
               "--samples", os.path.join(ROOT, args["samples"]), "--tasks", args.get("tasks", "niah_single_2,ruler_qa_squad,ruler_cwe"),
               "--length", str(args.get("length", 65536)), "--n", str(args.get("n", 1)),
               "--max-tokens", str(args.get("max_tokens", 64))]
        if args.get("ignore_eos"):
            cmd.append("--ignore-eos")
        out = os.path.join(RESULTS, f"replay_{arm}_{job['id']}.log")
    elif client == "longbench2":
        # LongBench v2, <= 120k-token subset (mexp/kimi/longbench2/): serial, greedy,
        # the four choices scored at the "Answer:" position, one request per question.
        cmd = [PY, os.path.join(ROOT, "mexp", "kimi", "run_longbench2.py"), "--arm", arm, "--port", port,
               "--model", MODEL, "--out", os.path.join(RESULTS, "longbench2")]
        if "limit" in args:
            cmd += ["--limit", str(args["limit"])]
        # max_tokens>0 switches to the generating protocol, which is the only one
        # that puts decode steps on the compressed path (see run_longbench2.py).
        if args.get("max_tokens"):
            cmd += ["--max-tokens", str(args["max_tokens"])]
        if (job.get("server_args") or "limit" in args or args.get("max_tokens")
                or str(job.get("env", {}).get("SGLANG_DEBUG_VESTIGEKV_STATS", "0")) == "1"):
            cmd += ["--tag", job["id"]]  # a variant must not overwrite the arm's plain run
        out = os.path.join(RESULTS, f"longbench2_{arm}_{job['id']}.log")
    elif client == "profile":
        # Kernel-level decode profiles at the given contexts (mexp/kimi/profile_stream.py).
        cmd = [PY, os.path.join(ROOT, "mexp", "kimi", "profile_stream.py"), "--port", port,
               "--out", os.path.join(RESULTS, "profile", job["id"]),
               "--ctxs", str(args.get("ctxs", "131072,262144")), "--steps", str(args.get("steps", 200)),
               # TODO(fan-wenjie): the client ignores this; drop both once the
               # long-running runner has been restarted onto this file.
               "--server-log-glob", os.path.join(RESULTS, f"server_{arm}_*.log")]
        out = os.path.join(RESULTS, f"profile_{arm}_{job['id']}.log")
    elif client == "gsm8k":
        cmd = [PY, "-m", "sglang.test.few_shot_gsm8k", "--num-shots", str(args.get("shots", 64)),
               "--num-questions", str(args.get("n", 1209)),
               "--data-path", os.path.join(ROOT, "mexp", "quality", "gsm8k_platinum.jsonl"),
               "--parallel", "1", "--port", port]
        out = os.path.join(RESULTS, f"gsm8k_{arm}_{job['id']}.log")
    else:
        raise ValueError(f"unknown client {client!r}")
    log(f"client {client} for {job['id']}: {' '.join(cmd)} -> {out}")
    with open(out, "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=ROOT).returncode
    tail = open(out, errors="replace").read().replace("\r", "\n").strip().splitlines()[-6:]
    return rc, out, tail


def main():
    import signal

    os.makedirs(RESULTS, exist_ok=True)
    server = Server()
    # SIGTERM (a restart) must run the finally below: Python's default handler
    # exits without it and leaves the server and the client running as orphans.
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        while True:
            done = {r["id"] for r in read_jsonl(STATE) if r.get("status") in ("done", "failed")}
            jobs = [j for j in read_jsonl(QUEUE) if not j.get("skip") and j["id"] not in done]
            if not jobs:
                log("queue empty; exiting")
                break
            job = jobs[0]
            t0 = time.time()
            append_state({"id": job["id"], "status": "running", "start": time.strftime("%F %T")})
            if not server.ensure(job["arm"], job.get("env", {}), job["id"], job.get("server_args", ())):
                append_state({"id": job["id"], "status": "failed", "note": "server did not start",
                              "end": time.strftime("%F %T")})
                continue
            try:
                if job.get("probe", False):
                    rc, out, tail = run_client({**job, "client": "needle", "id": job["id"] + "-probe"}, server.port)
                    log(f"probe: {tail[-1:] if tail else rc}")
                rc, out, tail = run_client(job, server.port)
            except Exception as e:
                # A malformed job is that job's failure, not the queue's: a
                # missing `args` key used to raise out of the loop and take
                # every remaining job down with the runner.
                log(f"{job['id']} failed to launch: {type(e).__name__}: {e}")
                append_state({"id": job["id"], "status": "failed",
                              "note": f"client not launched: {type(e).__name__}: {e}",
                              "wall_s": round(time.time() - t0), "end": time.strftime("%F %T")})
                continue
            status = "done" if rc == 0 else "failed"
            append_state({"id": job["id"], "status": status, "rc": rc, "log": out,
                          "wall_s": round(time.time() - t0), "end": time.strftime("%F %T"),
                          "tail": tail[-3:]})
            log(f"{job['id']} {status} rc={rc} wall={round(time.time() - t0)}s")
    finally:
        for pid in client_pids():
            subprocess.run(["kill", "-TERM", str(pid)])
        server.stop()


if __name__ == "__main__":
    main()
