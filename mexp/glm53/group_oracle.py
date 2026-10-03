"""Three selectors against the oracle, at DSA's own 2048-row budget.

    python mexp/glm53/group_oracle.py [--dir results/glm53/cd_grp] [--rows 2048]

The metric is the one select_recall_telemetry already uses, so these numbers
sit beside the ones the server reports:

  count  fraction of the oracle's top-`rows` positions the selector holds
  mass   fraction of the oracle's score it holds, floor-shifted into [0, 1]
         (missing the 2000th position is not missing the 1st)

The oracle is the LATENT score -- what attention actually wants -- reduced best
over heads, which is the per-layer granularity DSA selects at. The selectors
are scored in their own spaces and then judged against that one oracle:

  idx-group   top rows//4 groups by pooled index key . indexer query, expanded
              (the design: DSA's channel, DSA's ratio, DSA's budget)
  lat-group   top rows//4 groups by pooled LATENT row . latent query, expanded
              (what grouping costs when the score is the oracle's own space --
              an upper bound on any group selector, and the control that says
              whether a loss is the POOLING or the CHANNEL)
  sigma       top `rows` positions by tier-1 sigma, query-agnostic
              (the floor: what the resident set alone would recall)

A budget cannot overflow, so none of these has a fallback rate. That is the
point of measuring recall instead.
"""

import argparse
import glob
import os

import torch


def recall(score_oracle, selected, rows):
    """(count, mass) of the oracle's top-`rows` held by `selected` positions."""
    T = score_oracle.shape[0]
    k = min(rows, T)
    top = score_oracle.topk(k)
    hit = torch.zeros(T, dtype=torch.bool, device=score_oracle.device)
    sel = selected[(selected >= 0) & (selected < T)]
    hit[sel] = True
    got = hit[top.indices]
    w = top.values - top.values[-1]
    tot = w.sum()
    mass = float((w * got).sum() / tot) if float(tot) > 0 else float(got.float().mean())
    return float(got.float().mean()), mass


def expand(group_ids, pool, T):
    off = torch.arange(pool, device=group_ids.device)
    ids = (group_ids[:, None] * pool + off).reshape(-1)
    return ids[ids < T]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/glm53/cd_grp")
    ap.add_argument("--rows", type=int, default=2048)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    import sys
    sys.path.insert(0, "/home/user/vestigekv-wt/engine-glm521/python")
    from sglang.srt.layers.attention.vestigekv.dsa_index_view import (
        group_scores, select_groups)

    print(f"budget={a.rows} rows  device={dev}")
    print(f"{'lid':>4} {'tp':>2} {'idx-grp c/m':>14} {'lat-grp c/m':>14} "
          f"{'sigma c/m':>14} {'groups':>7}")
    agg = []
    for f in sorted(glob.glob(os.path.join(a.dir, "*.pt"))):
        d = torch.load(f, map_location="cpu", weights_only=False)
        ik = d.get("index_k")
        if ik is None or ik.get("index_q") is None:
            continue
        pool = int(ik["pool_size"])
        rows_lat = d["rows"].to(dev).float()
        qlat = d["qcal"].to(dev).float()
        gkeys = ik["deq"].to(dev)
        qidx = ik["index_q"].to(dev)
        if qidx.ndim == 3:            # [tok, heads, dim] -> last token's heads
            qidx = qidx[-1]
        T = rows_lat.shape[0]
        G = gkeys.shape[0]
        # pooled latent rows, the control
        glat = rows_lat[: G * pool].view(G, pool, -1).mean(1)
        n = qlat.shape[0]
        c = torch.zeros(3); m = torch.zeros(3)
        for i in range(n):
            q = qlat[i]
            oracle = (rows_lat @ q.T).amax(dim=1)
            s_idx = group_scores(gkeys, qidx)
            sel_i = expand(select_groups(s_idx, a.rows, pool), pool, T)
            s_lat = (glat @ q.T).amax(dim=1)
            sel_l = expand(select_groups(s_lat, a.rows, pool), pool, T)
            c0, m0 = recall(oracle, sel_i, a.rows)
            c1, m1 = recall(oracle, sel_l, a.rows)
            c[0] += c0; m[0] += m0; c[1] += c1; m[1] += m1
        c /= n; m /= n
        base = os.path.basename(f)
        tp = int(base.split("_tp")[1][0]); lid = int(base.split("_lid")[1].split("_")[0])
        print(f"{lid:>4} {tp:>2} {c[0]:>6.3f}/{m[0]:<7.3f} {c[1]:>6.3f}/{m[1]:<7.3f} "
              f"{'-':>6}/{'-':<7} {G:>7}")
        agg.append((c[0], m[0], c[1], m[1]))
    if agg:
        v = torch.tensor(agg).mean(0)
        print(f"{'mean':>7} {v[0]:>6.3f}/{v[1]:<7.3f} {v[2]:>6.3f}/{v[3]:<7.3f}")
        print("\nidx-grp is the design; lat-grp is the same grouping scored in the "
              "oracle's own space, so a gap between them is the CHANNEL and a gap "
              "from 1.0 in lat-grp is the POOLING.")


if __name__ == "__main__":
    main()
