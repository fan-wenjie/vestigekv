"""Kernel-level profile of decode steps at given context lengths, on a running server.

    python mexp/kimi/profile_stream.py --port 30000 --out results/kimi/profile/<job> \
        [--ctxs 131072,262144] [--steps 200] [--server-log-glob 'results/kimi/server_vestigekv_*.log']

One request (4096 random tokens in, ignore_eos, decoded past the last context point)
is driven in the background; when the server's decode log reaches each context, the
built-in profiler is started for `steps` forward steps (POST /start_profile with
num_steps: torch.profiler with CUPTI, kernels inside CUDA-graph replays included)
and its chrome trace lands in --out as ctx<N>k-TP-<rank>.trace.json.gz. Two trees'
traces are diffed per kernel with mexp/kimi/kernel_diff.py.
"""

import argparse
import glob
import json
import os
import random
import re
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def post(port, path, body, timeout=60):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    text = urllib.request.urlopen(req, timeout=timeout).read().decode(errors="replace")
    try:
        return json.loads(text)
    except ValueError:  # /start_profile answers with plain text
        return text.strip()


def newest(pattern):
    paths = glob.glob(pattern)
    return max(paths, key=os.path.getmtime) if paths else None


def decoded_tokens(log_path):
    """The last '#full token: N' (or '#token: N') the scheduler logged."""
    n = 0
    with open(log_path, errors="replace") as f:
        for line in f:
            if "Decode batch" in line and "TP0]" in line:
                m = re.search(r"#(?:full )?token: (\d+)", line)
                if m:
                    n = int(m.group(1))
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="30000")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ctxs", default="131072,262144")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--input-len", type=int, default=4096)
    ap.add_argument("--server-log-glob", default=os.path.join(ROOT, "results", "kimi", "server_vestigekv_*.log"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ctxs = [int(x) for x in args.ctxs.split(",")]
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    rng = random.Random(args.seed)
    ids = [rng.randrange(1000, 100000) for _ in range(args.input_len)]
    max_new = max(ctxs) - args.input_len + args.steps + 2048
    result = {}

    def generate():
        t0 = time.time()
        try:
            r = post(args.port, "/generate", {"input_ids": ids, "sampling_params": {
                "max_new_tokens": max_new, "ignore_eos": True, "temperature": 0}}, timeout=6 * 3600)
            result["completion_tokens"] = r.get("meta_info", {}).get("completion_tokens")
        except Exception as e:  # the request outlives the profiles; a failure is reported below
            result["error"] = repr(e)
        result["wall_s"] = time.time() - t0

    th = threading.Thread(target=generate, daemon=True)
    th.start()
    log = None
    for ctx in ctxs:
        target = ctx - 1024  # trigger a little early; the window then straddles the context point
        while True:
            log = newest(args.server_log_glob)
            n = decoded_tokens(log) if log else 0
            if n >= target or not th.is_alive():
                break
            time.sleep(5)
        if not th.is_alive():
            print(f"generation ended before {ctx}: {result}", flush=True)
            break
        pid = f"ctx{ctx // 1024}k"
        t0 = time.time()
        r = post(args.port, "/start_profile", {"output_dir": out, "num_steps": args.steps, "profile_id": pid,
                                               "activities": ["CPU", "GPU"]}, timeout=600)
        print(f"{pid}: profile started at #token={n} ({r}); waiting for the trace", flush=True)
        trace = os.path.join(out, f"{pid}-TP-0.trace.json.gz")
        while not os.path.exists(trace) and th.is_alive() and time.time() - t0 < 1800:
            time.sleep(5)
        print(f"{pid}: {'trace written' if os.path.exists(trace) else 'NO TRACE'} after {time.time() - t0:.0f}s: {trace}", flush=True)
    th.join()
    print(f"generation finished: {result}", flush=True)
    with open(os.path.join(out, "profile_run.json"), "w") as f:
        json.dump({"ctxs": ctxs, "steps": args.steps, "input_len": args.input_len, "server_log": log, **result}, f, indent=2)
    if any(not os.path.exists(os.path.join(out, f"ctx{c // 1024}k-TP-0.trace.json.gz")) for c in ctxs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
