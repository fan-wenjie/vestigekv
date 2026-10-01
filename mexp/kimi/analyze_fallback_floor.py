"""Is fallback a property of the certificate, or of the fetch buffer?

The certificate's answer on a step is a SET: {u : score_u > max1 - margin},
of size t = n_fired. The fallback is not triggered by that set being large in
any absolute sense -- it is triggered by t exceeding the buffer W. And softmax
decomposes exactly over any partition of the key set (the log-sum-exp merge),
so a step needing t > W rows has an exactly-sound answer that costs t rows,
not the whole archive.

So the question is quantitative: how large is t relative to n_arch when the
buffer overflows? If t/n_arch is small on overflowing steps, fallback is a
buffer artefact and raising W removes it while REDUCING traffic. If t/n_arch
is near 1, the certificate really is saying the step is dense and nothing but
a tighter bound can help.

Reads n_fired / n_arch / n_kept per layer-step from the step dumps and sweeps W.
CPU only.
"""

import argparse
import glob
import json
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROW_DENSE = 1152    # LATENT_DIM 576 x bf16: the row the attention reads
ROW_SCAN = 137      # branch scan row: sidecar 128 + rho 4 + hit 1
ID_BYTES = 4        # _fetch_stack is [n_layers, max_reqs, W] of D.INDEX_DTYPE
N_MLA = 7           # MLA layers on this model: 3, 7, 11, 15, 19, 23, 26


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--min-arch", type=int, default=4096)
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(
        ROOT, "results/kimi/stepattr", f"stepattr_{a.job}_tp*.jsonl")))
    if not paths:
        raise SystemExit(f"no dumps for {a.job}")

    steps = []          # (lid, n_fired, n_arch, n_kept)
    for p in paths:
        with open(p) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("n_arch", 0) < a.min_arch:
                    continue
                steps.append((d["lid"], int(d["n_fired"]), int(d["n_arch"]),
                              int(d.get("n_kept", 0))))
    n = len(steps)
    print(f"job={a.job}  {n} live layer-steps from {len(paths)} dumps\n")

    # how much of the archive the certificate actually asks for, on the steps
    # that overflow the current buffer
    print("admitted fraction t/n_arch on steps that overflow W=4096:")
    byl = defaultdict(list)
    for lid, t, na, _ in steps:
        if t > 4096:
            byl[lid].append(t / na)
    print(f"  {'lid':>4} {'steps':>7} {'p50':>7} {'p90':>7} {'max':>7}")
    for lid in sorted(byl):
        v = sorted(byl[lid])
        q = lambda f: v[min(len(v) - 1, int(f * len(v)))]
        print(f"  {lid:>4} {len(v):>7} {q(.5):>7.3f} {q(.9):>7.3f} {v[-1]:>7.3f}")
    allv = sorted(x for v in byl.values() for x in v)
    if allv:
        q = lambda f: allv[min(len(allv) - 1, int(f * len(allv)))]
        print(f"  {'ALL':>4} {len(allv):>7} {q(.5):>7.3f} {q(.9):>7.3f} {allv[-1]:>7.3f}")

    # traffic per decode step (summed over the 7 MLA layers) against W
    print(f"\nper-step traffic vs W   (dense reference = scan-free, "
          f"n_arch+n_kept rows at {ROW_DENSE} B)")
    print(f"  {'W':>8} {'fallback':>9} {'bytes/step':>12} {'vs dense':>9} "
          f"{'buf bs1':>8} {'buf bs32':>9}  {'of which scan':>14}")
    nsteps = len(set((lid, i) for i, (lid, _, _, _) in enumerate(steps)))
    # one "decode step" is 7 layer-steps; report per layer-step and scale
    dense = sum((na + nk) * ROW_DENSE for _, _, na, nk in steps)
    for W in (4096, 8192, 16384, 32768, 65536, 131072, 262144):
        scan = sum(na * ROW_SCAN for _, _, na, _ in steps)
        fetch = 0
        ovf = 0
        for lid, t, na, nk in steps:
            fetch += nk * ROW_DENSE
            if t > W:
                ovf += 1
                fetch += na * ROW_DENSE          # dense fallback reads it all
            else:
                fetch += t * ROW_DENSE
        tot = scan + fetch
        buf = N_MLA * W * ID_BYTES / 2**20
        print(f"  {W:>8} {100*ovf/n:>8.1f}% {tot/n/1024:>11.1f}K "
              f"{tot/dense:>8.3f}x {buf:>7.1f}M {32*buf:>8.0f}M "
              f"{100*scan/tot:>13.1f}%")

    print(f"\n  floor (W = n_arch, fallback impossible): the scan alone is "
          f"{100*sum(na*ROW_SCAN for _,_,na,_ in steps)/dense:.1f}% of dense")


if __name__ == "__main__":
    main()
