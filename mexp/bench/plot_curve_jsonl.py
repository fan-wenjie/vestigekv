"""Plot decode-ms-vs-S straight from bench_serving --output-details JSONLs.

Zero intermediate files: reads the official "itls" arrays, draws rolling
median + IQR per arm. Pass one or more --arm NAME:JSONL[:PREFILL] pairs.
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--arm", action="append", required=True,
                help="NAME:JSONL[:PREFILL]  (PREFILL default 4096)")
ap.add_argument("--out", required=True)
ap.add_argument("--window", type=int, default=2048)
ap.add_argument("--clip", type=int, default=None,
                help="drop client itls beyond this S (stream-batching artifact "
                     "region; pair with --srv for the server-side authority)")
ap.add_argument("--ylim", type=str, default=None, help="LO:HI y-axis bounds")
ap.add_argument("--srv", action="append", default=[],
                help="NAME:NODE0_LOG  server-side Decode-batch bucket rates, "
                     "overlaid as points (the per-token authority at high S)")
a = ap.parse_args()

fig, ax = plt.subplots(figsize=(8.5, 4.5))
# grayscale-safe: one line, one legend entry per arm; the client->server
# source splice is invisible by design (same statistic) and is documented
# in the caption, not the legend.
styles = {"dense": dict(color="black", ls="-"), "vestigekv": dict(color="black", ls="--")}
series = {}
for spec in a.arm:
    parts = spec.split(":")
    name, path = parts[0], parts[1]
    prefill = int(parts[2]) if len(parts) > 2 else 4096
    rec = json.loads(open(path).read().strip().splitlines()[-1])
    itls = np.array(rec["itls"][0]) * 1e3
    S = prefill + np.arange(1, len(itls) + 1)
    if a.clip:
        keep = S <= a.clip
        itls, S = itls[keep], S[keep]
    # ONE statistic globally: 4k-token buckets, MEDIAN per bucket.
    B = 4096
    nb = len(itls) // B
    med = np.array([np.median(itls[i * B:(i + 1) * B]) for i in range(nb)])
    Sm = prefill + (np.arange(nb) * B + B // 2)
    series.setdefault(name, []).append((list(Sm), list(med)))
import datetime
import re

for spec in a.srv:
    name, path = spec.split(":", 1)
    pts = []
    for line in open(path):
        m = re.match(r"\[(\S+ \S+) PP0\].*Decode batch.*#full token: (\d+)", line)
        if m:
            pts.append((datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"),
                        int(m.group(2))))
    # Same statistic as the client side: per 4k bucket, the MEDIAN of its
    # four 1k sub-interval rates (the finest rate samples a 1-second-
    # stamped log yields; at 4k the bucket spans ~35s, so quantization is
    # <3%). Drawn only beyond --clip, splicing the client curve.
    sub = {}
    for t, n in pts:
        sub.setdefault(n // 1024, []).append((t, n))
    rates = {}
    for k, b in sub.items():
        if b[-1][1] - b[0][1] > 400:
            rates[k] = (b[-1][0] - b[0][0]).total_seconds() * 1e3 / (b[-1][1] - b[0][1])
    xs, ys = [], []
    for k4 in sorted({k // 4 for k in rates}):
        rs = [rates[k] for k in (4 * k4, 4 * k4 + 1, 4 * k4 + 2, 4 * k4 + 3) if k in rates]
        if len(rs) >= 3:
            xs.append(k4 * 4096 + 2048)
            ys.append(float(np.median(rs)))
    if a.clip:
        keep = [i for i, x in enumerate(xs) if x > a.clip]
        xs = [xs[i] for i in keep]
        ys = [ys[i] for i in keep]
    series.setdefault(name, []).append((xs, ys))

for name, segs in series.items():
    X, Y = [], []
    for xs, ys in segs:
        X += list(xs)
        Y += list(ys)
    order = np.argsort(X)
    X = np.array(X)[order]
    Y = np.array(Y)[order]
    ax.plot(X, Y, lw=1.8, label=name, **styles.get(name, {}))
# x-axis in k where k = 1024 tokens (KiB-style), not 1000
import matplotlib.ticker as _mtick

ax.xaxis.set_major_formatter(
    _mtick.FuncFormatter(lambda x, _pos: f"{x / 1024:.0f}k")
)
# tick at round multiples of 64k so labels read 0, 64k, 128k, ... 512k
ax.xaxis.set_major_locator(_mtick.MultipleLocator(64 * 1024))
ax.set_xlabel("sequence length S (k tokens, k = 1024)")
ax.set_ylabel("per-token decode latency (ms/token)")
if a.ylim:
    lo, hi = a.ylim.split(":")
    ax.set_ylim(float(lo), float(hi))
ax.legend(loc="upper left")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(a.out, dpi=130)
print(f"saved {a.out}")
