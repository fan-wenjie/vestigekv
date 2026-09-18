"""Per-token decode latency vs context length from the stream jobs.

    python mexp/glm53/stream_curve.py [--arms baseline,vestigekv] [--window 4096]
    python mexp/glm53/stream_curve.py --source server [--jobs stream-baseline-256k,stream-vestigekv-256k]

Client source (default): results/<line>/latency_stream_4k-<N>_<arm>.jsonl
(sglang.benchmark.serving --output-details: one request, its inter-token latencies);
per arm, the median inter-token latency in the `window` tokens ending at each
context length (4k prefill + tokens decoded so far), plus the ratio between arms.

Server source: the arm's server log (results/<line>/server_<arm>_<job>.log), whose
scheduler prints "Decode batch ... #full token: T ... gen throughput (token/s): R"
every 40 steps; per context point the median of R over the log lines within
+/-`window`/2 tokens, as ms/token, with the spread (p10-p90 of the same lines) so
a point's stability is visible. This is the paper's metric (server clock, no
client streaming path, no end-of-stream artifact); the client curve's last
window is not comparable to it.
"""

import argparse
import json
import os
import re
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def load(arm, prefill, out_len, line="glm53"):
    """Return a curve that answers median-ms-at-context, from either the full
    per-token array or the reduced record mexp/tools/reduce_streams.py writes.

    The reduced record stores the window medians in the source unit (seconds),
    computed from the full array before it was dropped, so both paths give
    identical numbers -- see that script's --verify."""
    path = os.path.join(ROOT, "results", line, f"latency_stream_{prefill // 1024}k-{out_len}_{arm}.jsonl")
    with open(path) as f:
        d = json.loads(f.readlines()[-1])
    if "itls" in d:
        itls = [x * 1000.0 for x in d["itls"][0]]  # seconds -> ms
        return {"full": itls, "inp": d["input_lens"][0],
                "mean": statistics.mean(itls), "median": statistics.median(itls),
                "n": len(itls)}
    r = d["itls_reduced"][0]
    g = r["itl_global"]
    return {"full": None, "inp": r["input_len"],
            "windows": {int(c): w for c, w in r["itl_windows"].items() if w},
            "mean": g["mean"] * 1000.0, "median": g["median"] * 1000.0, "n": g["n"]}


def median_at(c, ctx, window):
    """Median inter-token latency (ms) over the `window` tokens ending at `ctx`."""
    if c["full"] is not None:
        end = ctx - c["inp"] - 1  # itls[i] is the gap before output token i+1
        seg = c["full"][max(0, end - window):end]
        return statistics.median(seg) if seg else float("nan")
    w = c["windows"].get(ctx)
    return w["median"] * 1000.0 if w else float("nan")


def load_server(job, arm, line):
    path = os.path.join(ROOT, "results", line, f"server_{arm}_{job}.log")
    pts = []
    for l in open(path, errors="replace"):
        if "TP0]" in l and "Decode batch" in l:
            m = re.search(r"#(?:full )?token: (\d+).*gen throughput \(token/s\): ([0-9.]+)", l)
            if m and float(m.group(2)) > 0:
                pts.append((int(m.group(1)), 1000.0 / float(m.group(2))))  # tok/s -> ms/token
    return pts


def server_curve(args, arms):
    jobs = args.jobs.split(",") if args.jobs else [f"stream-{arm}-{(args.prefill + args.output_len) // 1024}k" for arm in arms]
    # keyed by job, not by arm: two jobs of the same arm (two engine trees) are
    # the common comparison here and would collapse into one column
    pairs = list(zip(arms, jobs))
    curves = {job: load_server(job, arm, args.line) for arm, job in pairs}
    labels = [job if len(set(jobs)) == len(jobs) else f"{arm}:{job}" for arm, job in pairs]
    total = args.prefill + args.output_len
    points = sorted({p for p in [8192 * 2**i for i in range(0, 6)] + [total - 4096] if p <= total})
    half = args.window // 2
    print(f"ms/token from the server's gen-throughput lines within +/-{half} tokens of each context (median [p10-p90])")
    print(f"{'context':>9s}" + "".join(f"{l[-23:]:>24s}" for l in labels) + (f"{'ratio':>8s}" if len(pairs) == 2 else ""))
    for ctx in points:
        row = f"{ctx // 1024:8d}k"
        vals = []
        for job in jobs:
            seg = sorted(v for n, v in curves[job] if abs(n - ctx) <= half)
            if not seg:
                row += f"{'-':>24s}"
                vals.append(float("nan"))
                continue
            med = statistics.median(seg)
            p10, p90 = seg[int(0.1 * (len(seg) - 1))], seg[int(0.9 * (len(seg) - 1))]
            vals.append(med)
            row += f"{med:9.3f} [{p10:6.3f}-{p90:6.3f}]"
        if len(pairs) == 2:
            row += f"{vals[0] / vals[1]:8.3f}"
        print(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="baseline,vestigekv")
    ap.add_argument("--source", choices=["client", "server"], default="client")
    ap.add_argument("--jobs", default="", help="server source: the job ids whose server logs to read, one per arm")
    ap.add_argument("--prefill", type=int, default=4096)
    ap.add_argument("--output-len", type=int, default=126976)
    ap.add_argument("--window", type=int, default=4096)
    ap.add_argument("--line", default="glm53", help="results/<line>/ holds the stream files")
    args = ap.parse_args()
    arms = args.arms.split(",")
    if args.source == "server":
        server_curve(args, arms)
        return
    curves = {arm: load(arm, args.prefill, args.output_len, args.line) for arm in arms}
    points = [8192 * 2**i for i in range(0, 5)] + [args.prefill + args.output_len]
    points = sorted({p for p in points if p <= args.prefill + args.output_len})
    print(f"median inter-token latency (ms) over the {args.window} tokens ending at each context length")
    print(f"{'context':>9s}" + "".join(f"{arm:>12s}" for arm in arms) + (f"{'ratio':>8s}" if len(arms) == 2 else ""))
    for ctx in points:
        row = f"{ctx // 1024:8d}k"
        vals = []
        for arm in arms:
            v = median_at(curves[arm], ctx, args.window)
            vals.append(v)
            row += f"{v:12.3f}"
        if len(arms) == 2:
            row += f"{vals[0] / vals[1]:8.3f}"
        print(row)
    for arm in arms:
        c = curves[arm]
        print(f"{arm}: mean {c['mean']:.3f} ms/token, median {c['median']:.3f}, tokens {c['n'] + 1}")


if __name__ == "__main__":
    main()
