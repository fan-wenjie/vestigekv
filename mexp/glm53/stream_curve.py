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
    path = os.path.join(ROOT, "results", line, f"latency_stream_{prefill // 1024}k-{out_len}_{arm}.jsonl")
    with open(path) as f:
        d = json.loads(f.readlines()[-1])
    return [x * 1000.0 for x in d["itls"][0]], d["input_lens"][0]  # seconds -> ms


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
    curves = {arm: load_server(job, arm, args.line) for arm, job in zip(arms, jobs)}
    total = args.prefill + args.output_len
    points = sorted({p for p in [8192 * 2**i for i in range(0, 6)] + [total - 4096] if p <= total})
    half = args.window // 2
    print(f"ms/token from the server's gen-throughput lines within +/-{half} tokens of each context (median [p10-p90])")
    print(f"{'context':>9s}" + "".join(f"{arm:>24s}" for arm in arms) + (f"{'ratio':>8s}" if len(arms) == 2 else ""))
    for ctx in points:
        row = f"{ctx // 1024:8d}k"
        vals = []
        for arm in arms:
            seg = sorted(v for n, v in curves[arm] if abs(n - ctx) <= half)
            if not seg:
                row += f"{'-':>24s}"
                vals.append(float("nan"))
                continue
            med = statistics.median(seg)
            p10, p90 = seg[int(0.1 * (len(seg) - 1))], seg[int(0.9 * (len(seg) - 1))]
            vals.append(med)
            row += f"{med:9.3f} [{p10:6.3f}-{p90:6.3f}]"
        if len(arms) == 2:
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
            itls, inp = curves[arm]
            end = ctx - inp - 1  # itls[i] is the gap before output token i+1
            seg = itls[max(0, end - args.window) : end]
            v = statistics.median(seg) if seg else float("nan")
            vals.append(v)
            row += f"{v:12.3f}"
        if len(arms) == 2:
            row += f"{vals[0] / vals[1]:8.3f}"
        print(row)
    for arm in arms:
        itls, _ = curves[arm]
        print(f"{arm}: mean {statistics.mean(itls):.3f} ms/token, median {statistics.median(itls):.3f}, tokens {len(itls) + 1}")


if __name__ == "__main__":
    main()
