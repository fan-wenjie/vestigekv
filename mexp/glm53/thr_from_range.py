"""Can a range-interpolated threshold replace the exact quantile, and delete a kernel?

tier 2's threshold is the q0.75 quantile of the kept GROUPS' scores, and computing it
needs a 256-bin histogram -- one of five kernels in the branch pipeline, 32.6 us/step at
124k, which the ~2x NCCL amplifier makes ~65 us of step time. But _branch_kept already
computes the kept scores' min and max, so `lo + alpha*(hi-lo)` is free.

The question is whether some alpha reproduces the quantile's (fired count, attended
mass). If it does, the histogram kernel goes away; if the two disagree, the shape of the
kept score distribution is load-bearing and the histogram stays.
"""
import glob, sys, collections
import torch
sys.path.insert(0, "mexp/glm53")
from dsa_selection_stats import score, sigma_at

WF, NC, KEPT, Q, BLOCK, KAPPA = 512, 24, 32, 0.75, 256, 4
ALPHAS = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
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
        full = torch.softmax(qc[i] @ rows.T * sc_, dim=-1)
        def apply(thr):
            cand = (s[:C] > thr); cand[kept] = False
            fired = cand.nonzero().flatten()
            if fired.numel() > WF:                     # the cap, as implemented
                upper = max(float(s[fired].max()), lo + 2*(hi-lo))
                lvl = (((s[fired]-lo)/max(upper-lo,1e-6)*NC).to(torch.int32)).clamp(0,NC-1)
                pop = torch.bincount(lvl, minlength=NC)
                above = pop.sum() - pop.cumsum(0)
                ok = (above <= WF).nonzero().flatten()
                cap = min(int(ok.min()) if ok.numel() else NC-1, NC-2)
                fired = fired[lvl > cap]
            g = torch.cat([kept, fired])
            r = (g[:, None]*pool + torch.arange(pool)[None, :]).reshape(-1)
            r = r[r < Tn]
            r = torch.cat([torch.arange(min(4, Tn)), r, tail]).unique()
            return float(full[:, r].sum(-1).mean()), fired.numel(), r.numel()
        m_q, n_q, r_q = apply(float(ks.quantile(Q)))
        acc["q_mass"].append(m_q); acc["q_fired"].append(float(n_q)); acc["q_rows"].append(float(r_q))
        for a in ALPHAS:
            m, n_, r_ = apply(lo + a*(hi-lo))
            acc[f"m{a}"].append(m); acc[f"n{a}"].append(float(n_)); acc[f"r{a}"].append(float(r_))
n = len(acc["q_mass"])
print(f"{len(files)} dumps, {n} honest queries, kept {KEPT} groups\n")
print(f"   {'threshold':>22s} {'mass':>8s} {'fired':>8s} {'rows':>8s}")
print(f"   {'q0.75 quantile (now)':>22s} {sum(acc['q_mass'])/n:8.4f} "
      f"{sum(acc['q_fired'])/n:8.1f} {sum(acc['q_rows'])/n:8.1f}")
for a in ALPHAS:
    print(f"   {'lo + %.2f*(hi-lo)'%a:>22s} {sum(acc[f'm{a}'])/n:8.4f} "
          f"{sum(acc[f'n{a}'])/n:8.1f} {sum(acc[f'r{a}'])/n:8.1f}")
