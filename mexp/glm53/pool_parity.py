"""What does recalling in groups of 4 cost at a FIXED row budget?

    python mexp/glm53/pool_parity.py [--dir results/glm53/caldump2]
                                     [--pool 4] [--budget 2048]

The parity rule is the whole point: DSA attends index_topk = 2048 rows, so a
pooled selector is compared at 2048 rows too (budget // pool groups), not at
the pooled budget. archive_pool.py measured the pooled archive LOSING at row
parity and winning only at twice the rows, and that measurement was a ranking
over the rank-64 sketch; this one is over the rows themselves, per layer, and
reports the two statistics the selection telemetry already uses:

  count  fraction of the oracle's top-budget rows the selector holds
  mass   fraction of the oracle's score it holds, floor-shifted so the ratio
         stays in [0, 1] (missing the 2000th row is not missing the 1st)

A row's score is its best over heads, which is the per-layer granularity DSA
selects at: a row any head wants is a row the layer wants.

Also reports the pooling SPREAD, max_i ||x_i - mean|| / ||mean||, for both the
latent rows and the indexer keys. That ratio is what made pooling vacuous for
the sketch (1.00 prose, 0.73 RULER): adjacent rows were uncorrelated, so a
group's mean said nothing about its members. The indexer keys are the channel
this design scores on, and DSA pools them 4:1 itself, so their spread is the
number that decides whether the grouping is viable here.
"""

import argparse
import glob
import os

import torch


def spread_ratio(x: torch.Tensor, pool: int) -> float:
    """median of max_i ||x_i - m|| / ||m|| over groups of `pool` rows."""
    g = x.shape[0] // pool
    if g == 0:
        return float("nan")
    grp = x[: g * pool].view(g, pool, -1).float()
    m = grp.mean(1)
    sp = (grp - m.unsqueeze(1)).norm(dim=2).amax(1)
    return float((sp / m.norm(dim=1).clamp_min(1e-9)).median())


def recall(rows, q, pool, budget):
    """(count, mass) of the oracle's top-`budget` rows held by a pooled
    selector that takes budget//pool groups and attends all their rows."""
    score = (rows @ q.T).amax(dim=1)          # [T], best over heads
    T = score.shape[0]
    k = min(budget, T)
    top = score.topk(k)
    g = T // pool
    gsel = max(1, k // pool)
    # score the group by its POOLED row, which is what the scan would read
    gmean = rows[: g * pool].view(g, pool, -1).mean(1)
    gscore = (gmean @ q.T).amax(dim=1)
    picked = gscore.topk(min(gsel, g)).indices
    off = torch.arange(pool, device=rows.device)
    sel = (picked[:, None] * pool + off).reshape(-1)
    hit = torch.zeros(T, dtype=torch.bool, device=rows.device)
    hit[sel[sel < T]] = True
    got = hit[top.indices]
    w = top.values - top.values[-1]
    tot = w.sum()
    mass = float((w * got).sum() / tot) if float(tot) > 0 else float(got.float().mean())
    return float(got.float().mean()), mass, int(sel.numel())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/glm53/caldump2")
    ap.add_argument("--pool", type=int, default=4)
    ap.add_argument("--budget", type=int, default=2048)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"pool={a.pool}  row budget={a.budget} (groups={a.budget // a.pool})  "
          f"device={dev}")
    print(f"{'lid':>4} {'tp':>2} {'spread_lat':>10} {'spread_idx':>10} "
          f"{'count':>7} {'mass':>7} {'cnt@2x':>7} {'mass@2x':>7}")
    agg = {}
    for f in sorted(glob.glob(os.path.join(a.dir, "*.pt"))):
        d = torch.load(f, map_location="cpu", weights_only=False)
        rows = d["rows"].to(dev).float()
        qcal = d["qcal"].to(dev).float()
        ik = d.get("index_k")
        s_lat = spread_ratio(rows, a.pool)
        s_idx = spread_ratio(ik.to(dev), a.pool) if ik is not None else float("nan")
        c = m = c2 = m2 = 0.0
        n = qcal.shape[0]
        for i in range(n):
            q = qcal[i]
            r = recall(rows, q, a.pool, a.budget)
            r2 = recall(rows, q, a.pool, a.budget * 2)
            c += r[0]; m += r[1]; c2 += r2[0]; m2 += r2[1]
        c, m, c2, m2 = c / n, m / n, c2 / n, m2 / n
        base = os.path.basename(f)
        tp = int(base.split("_tp")[1][0]); lid = int(base.split("_lid")[1].split("_")[0])
        print(f"{lid:>4} {tp:>2} {s_lat:>10.3f} {s_idx:>10.3f} "
              f"{c:>7.3f} {m:>7.3f} {c2:>7.3f} {m2:>7.3f}")
        agg.setdefault("rows", []).append((s_lat, s_idx, c, m, c2, m2))
    v = torch.tensor(agg["rows"])
    mu = v.nanmean(0)
    print(f"{'mean':>7} {mu[0]:>10.3f} {mu[1]:>10.3f} "
          f"{mu[2]:>7.3f} {mu[3]:>7.3f} {mu[4]:>7.3f} {mu[5]:>7.3f}")
    print("\ncount/mass are the pooled selector's recall of the oracle at the "
          "stated row budget; @2x is the same at twice the rows, which is where "
          "archive_pool.py found pooling winning for the sketch.")


if __name__ == "__main__":
    main()
