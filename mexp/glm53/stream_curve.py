"""Per-token decode latency vs context length from the stream jobs' result files.

    python mexp/glm53/stream_curve.py [--arms baseline,vestigekv] [--window 4096]

Reads results/glm53/latency_stream_4k-126976_<arm>.jsonl (sglang.benchmark.serving
--output-details: one request, its inter-token latencies) and prints, per arm, the
median inter-token latency in the `window` tokens ending at each context length
(4k prefill + tokens decoded so far), plus the ratio between the arms.
"""

import argparse
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def load(arm, prefill, out_len, line="glm53"):
    path = os.path.join(ROOT, "results", line, f"latency_stream_{prefill // 1024}k-{out_len}_{arm}.jsonl")
    with open(path) as f:
        d = json.loads(f.readlines()[-1])
    return [x * 1000.0 for x in d["itls"][0]], d["input_lens"][0]  # seconds -> ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="baseline,vestigekv")
    ap.add_argument("--prefill", type=int, default=4096)
    ap.add_argument("--output-len", type=int, default=126976)
    ap.add_argument("--window", type=int, default=4096)
    ap.add_argument("--line", default="glm53", help="results/<line>/ holds the stream files")
    args = ap.parse_args()
    arms = args.arms.split(",")
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
