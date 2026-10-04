"""kept, bytes and recall against rho -- recomputed from the dumps' sigma.

    python mexp/glm53/kept_rho_sweep.py [--dirs cd_g2,cd_w2,cd_w3]

kept = sinks + round(rho*closed) + (seq - closed) (eviction.kept_rows). At the
served shape the tail is 0 (the 32k prompt is an exact multiple of CLOSE_BLOCK)
so rho*closed IS kept, which makes rho the only large lever on the byte claim.
This re-derives the kept set from each dump's own sigma record at several rho,
and reports what tier 1 then holds of the true top-2048 and what the arm reads.

Recall here is tier 1 ALONE, which is the quantity rho moves directly; the
branch rule's tier 2 then fires on the groups the indexer ranks above the
kept groups' q0.75, so a smaller kept set also moves the threshold. That
coupling is why this is a screen, not a verdict: it says which rho are worth
a served run, not what the arm would score.
"""
import argparse, glob, sys
import torch

sys.path.insert(0, "/home/user/vestigekv-wt/engine-glm521/python")
from sglang.srt.layers.attention.vestigekv import defaults as D  # noqa: E402

ROW_B, IDX_B, DSA_ROWS = 1024, 33, 2048


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", default="cd_g2,cd_w2,cd_w3")
    ap.add_argument("--rhos", default="0.03125,0.015625,0.0078125,0.00390625")
    a = ap.parse_args()
    torch.set_num_threads(8)
    rhos = [float(x) for x in a.rhos.split(",")]
    agg = {r: [] for r in rhos}
    for dname in a.dirs.split(","):
        for f in sorted(glob.glob(f"results/glm53/{dname}/*.pt")):
            d = torch.load(f, map_location="cpu", weights_only=False)
            sigma = d["sigma"].float()
            T = int(d["seq_len"])
            closed = (T // D.CLOSE_BLOCK) * D.CLOSE_BLOCK
            if closed == 0:
                continue
            rows = d["rows"].float()
            for rho in rhos:
                m = max(1, round(rho * closed))
                n_sink = min(D.SINKS, closed)
                ranked = sigma[:closed].clone()
                ranked[:n_sink] = float("-inf")
                m = min(m, closed - n_sink)
                picked = ranked.topk(m).indices
                pos = torch.cat([torch.arange(n_sink), picked,
                                 torch.arange(closed, T)])
                kept_n = pos.numel()
                hits = []
                for i in range(d["qcal"].shape[0]):
                    q = d["qcal"][i].float()
                    oracle = (rows @ q.T).amax(1)
                    top = set(oracle.topk(min(DSA_ROWS, T)).indices.tolist())
                    hits.append(len(top & set(pos.tolist())) / max(len(top), 1))
                agg[rho].append((kept_n, T, sum(hits) / len(hits)))
    print(f"{'rho':>10}{'kept':>9}{'tail':>7}{'tier1 recall':>14}{'B/token':>10}{'vs DSA':>9}")
    for rho in rhos:
        v = agg[rho]
        if not v:
            continue
        kept = sum(x[0] for x in v) / len(v)
        T = sum(x[1] for x in v) / len(v)
        rec = sum(x[2] for x in v) / len(v)
        tail = T - (int(T) // D.CLOSE_BLOCK) * D.CLOSE_BLOCK
        ours = 35.25 + kept * ROW_B / T          # tier 1 only, no fetch
        dsa = IDX_B + DSA_ROWS * ROW_B / T
        print(f"{rho:>10.6f}{kept:>9.0f}{tail:>7.0f}{rec:>14.3f}{ours:>10.1f}{ours/dsa:>9.3f}")
    print(f"\n(mean seq {T:.0f}; B/token counts tier 1 only, so the arm's real")
    print(" number adds the fetched rows -- ~472 at the served rho)")


if __name__ == "__main__":
    main()
