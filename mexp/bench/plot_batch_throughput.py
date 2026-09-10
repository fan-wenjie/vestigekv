"""Plot decode throughput vs batch size straight from bench_serving JSONLs.

bs=1 latency lives in plot_curve_jsonl.py (per-token ms). The batch axis is a
THROUGHPUT story: how aggregate tok/s scales with concurrency. Each --arm
takes one or more JSONLs (one bench run per bs); max_concurrency and
output_throughput are read from the official record -- zero intermediate files.

  python plot_batch_throughput.py \
    --arm dense:run_bs1.jsonl,run_bs2.jsonl,... \
    --arm vestigekv:run_bs1.jsonl,... --out fig.png
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--arm", action="append", required=True,
                help="NAME:jsonl1,jsonl2,...  (one bench run per batch size)")
ap.add_argument("--out", required=True)
a = ap.parse_args()

colors = {"dense": "black", "vestigekv": "black"}
fig, ax = plt.subplots(figsize=(7.5, 4.5))
for spec in a.arm:
    name, paths = spec.split(":", 1)
    pts = []
    for p in paths.split(","):
        rec = json.loads(open(p).read().strip().splitlines()[-1])
        pts.append((int(rec["max_concurrency"]), float(rec["output_throughput"])))
    pts.sort()
    bs = [p[0] for p in pts]
    tput = [p[1] for p in pts]
    c = colors.get(name)
    ls = "-" if name == "dense" else "--"
    ax.plot(bs, tput, marker="o", ls=ls, lw=2, label=name, color=c)
ax.set_xlabel("batch size (concurrent requests)")
ax.set_ylabel("decode throughput (tok/s)")
ax.set_xticks(sorted({b for b in bs}))
ax.legend(loc="upper left")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(a.out, dpi=130)
print(f"saved {a.out}")
