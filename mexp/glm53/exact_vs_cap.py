"""On overflow steps: exact top-512 of the fired set, against the coarse-level cap.

The cap is CONSERVATIVE -- it keeps `above[cap] <= WF`, so it may keep fewer than the
buffer holds, short by one level's population. An exact 512-entry top-k over the fired
set would use the whole budget. The question is what that is worth, because the exact
version costs a refinement pass the cap does not need.
"""
import glob, sys, collections
import torch
sys.path.insert(0, "mexp/glm53")
from dsa_selection_stats import score, sigma_at

import os
WF, KEPT, Q, BLOCK, KAPPA = 512, 32, 0.75, 256, 4
NC = int(os.environ.get("NC", 24))
acc = collections.defaultdict(list)
files = sorted(glob.glob("results/glm53/cd3_64k/*.pt"))[:60]
for fp in files:
    d = torch.load(fp, map_location="cpu", weights_only=False)
    gk = d["index_k"]["deq"].float(); pool = int(d["index_k"]["pool_size"])
    G = gk.shape[0]; nb = G // BLOCK
    if nb == 0 or G <= 512: continue
    C = nb * BLOCK
    rows = d["rows"].float(); Tn = rows.shape[0]; sc_ = float(d["scale"])
    kept = sigma_at(gk, BLOCK, KAPPA)[:C].topk(min(KEPT, C)).indices
    tail = torch.arange(C * pool, Tn)
    qs, gates, qc = d["qidx"], d["qgate"], d["qcal"].float()
    if not isinstance(qs, (list, tuple)): continue
    for i in range(min(4, len(qs), qc.shape[0])):
        s = score(qs[i], gates[i], gk)
        ks = s[kept]; lo, hi = float(ks.min()), float(ks.max())
        w = max(hi - lo, 1e-6); thr = float(ks.quantile(Q))
        cand = (s[:C] > thr); cand[kept] = False
        fired = cand.nonzero().flatten()
        if fired.numel() <= WF:      # the cap does not engage
            continue
        full = torch.softmax(qc[i] @ rows.T * sc_, dim=-1)
        def mass(g):
            r = (g[:, None] * pool + torch.arange(pool)[None, :]).reshape(-1)
            r = r[r < Tn]
            r = torch.cat([torch.arange(min(4, Tn)), r, tail]).unique()
            return float(full[:, r].sum(-1).mean()), r.numel()
        # the coarse cap, as implemented: whole codes above `cap`
        upper = max(float(s[fired].max()), lo + 2 * w)
        lvl = (((s[fired] - lo) / max(upper - lo, 1e-6) * NC).to(torch.int32)).clamp(0, NC - 1)
        pop = torch.bincount(lvl, minlength=NC)
        above = pop.sum() - pop.cumsum(0)
        ok = (above <= WF).nonzero().flatten()
        cap = int(ok.min()) if ok.numel() else NC - 1
        cap = min(cap, NC - 2)
        keep_cap = torch.cat([kept, fired[lvl > cap]])
        # an exact top-512 of the fired set
        keep_exact = torch.cat([kept, fired[s[fired].topk(WF).indices]])
        mc, nc_ = mass(keep_cap); me, ne_ = mass(keep_exact)
        acc["cap_mass"].append(mc); acc["exact_mass"].append(me)
        acc["cap_groups"].append(float((lvl > cap).sum())); acc["exact_groups"].append(float(WF))
        acc["cap_rows"].append(float(nc_)); acc["exact_rows"].append(float(ne_))
        acc["fired"].append(float(fired.numel()))
n = len(acc["cap_mass"])
print(f"{len(files)} dumps, {n} OVERFLOW steps (fired > {WF})\n")
if n:
    for k in ("fired","cap_groups","exact_groups","cap_rows","exact_rows"):
        print(f"   {k:14s} {sum(acc[k])/n:9.1f}")
    cm, em = sum(acc['cap_mass'])/n, sum(acc['exact_mass'])/n
    print(f"\n   attention mass: coarse cap {cm:.4f}   exact top-512 {em:.4f}"
          f"   gap {em-cm:+.4f} ({100*(em-cm)/cm:+.2f}%)")
