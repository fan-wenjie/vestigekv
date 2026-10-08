"""Which data property could tighten the sigma bound -- and is it present?

The Cauchy-Schwarz slack measured at ~1/sqrt(d) is what makes the SOUND sigma
screen vacuous. Tightening it needs structure, so check the three candidates:

1. are the index-k keys normalised?  If ||k_g|| is near-constant the norm carries
   no information by construction, which would explain its below-random precision.
2. is q anisotropic?  A low-dimensional query subspace V gives
   |<R_t,q>| <= ||P_V R_t|| ||q|| with ||P_V R_t|| << sigma_t. Measured as the
   participation ratio of the q second-moment spectrum: eff_rank = (sum l)^2/sum l^2.
3. is the block already mean-centred along the sequence axis?  If so the DC mode
   is empty and kappa=1 is vacuous by construction.

The tension to settle: isotropy is what makes E[(Q_perp l)_t^2] = sigma_t^2/d exact
AND what makes the bound loose by sqrt(d). Both cannot be had.
"""

import glob
import sys

import torch

sys.path.insert(0, "mexp/glm53")
torch.set_num_threads(8)
B, KAPPA = 256, 16

files = []
for dn in ("cd_w3", "cd_w2", "cd_g2"):
    files += sorted(glob.glob(f"results/glm53/{dn}/*.pt"))

print("== 1. are the index-k keys normalised? (CV of ||k_g|| per dump)")
print(f"   {'lid':>4s} {'mean ||k||':>11s} {'CV':>8s} {'p01/p99 ratio':>14s}")
rows = {}
for fp in files:
    d = torch.load(fp, map_location="cpu", weights_only=False)
    n = d["index_k"]["deq"].float().norm(dim=-1)
    q01, q99 = n.quantile(0.01), n.quantile(0.99)
    rows.setdefault(int(d["lid"]), []).append(
        (float(n.mean()), float(n.std() / n.mean()), float(q01 / q99)))
for lid in sorted(rows):
    v = torch.tensor(rows[lid])
    print(f"   {lid:4d} {float(v[:,0].mean()):11.4f} {float(v[:,1].mean()):8.4f} "
          f"{float(v[:,2].mean()):14.4f}")

print("\n== 2. is q anisotropic in the 128-dim index space?")
print(f"   {'lid':>4s} {'eff_rank/128':>13s} {'top-1 share':>12s} {'top-16 share':>13s}")
acc = {}
for fp in files:
    d = torch.load(fp, map_location="cpu", weights_only=False)
    qs = d["qidx"]
    if not isinstance(qs, (list, tuple)):
        qs = [qs]
    Q = torch.cat([q.float() for q in qs], 0)  # [nq*heads, 128]
    M = Q.T @ Q / Q.shape[0]
    l = torch.linalg.eigvalsh(M.double()).clamp(min=0).flip(0)
    eff = float(l.sum() ** 2 / (l * l).sum())
    acc.setdefault(int(d["lid"]), []).append(
        (eff / l.numel(), float(l[0] / l.sum()), float(l[:16].sum() / l.sum())))
for lid in sorted(acc):
    v = torch.tensor(acc[lid])
    print(f"   {lid:4d} {float(v[:,0].mean()):13.4f} {float(v[:,1].mean()):12.4f} "
          f"{float(v[:,2].mean()):13.4f}")

print("\n== 3. how much block energy is in the DC mode (is kappa=1 vacuous)?")
print(f"   {'lid':>4s} {'DC share':>10s} {'bins 1-15 share':>16s} {'bins >=16 share':>16s}")
acc2 = {}
for fp in files:
    d = torch.load(fp, map_location="cpu", weights_only=False)
    gk = d["index_k"]["deq"].double()
    nb = gk.shape[0] // B
    if nb == 0:
        continue
    blk = gk[: nb * B].reshape(nb, B, -1)
    f = torch.fft.rfft(blk, dim=1)
    e = (f.abs() ** 2).sum(-1)  # [nb, bins]
    tot = e.sum(1)
    acc2.setdefault(int(d["lid"]), []).append(
        (float((e[:, 0] / tot).mean()), float((e[:, 1:KAPPA].sum(1) / tot).mean()),
         float((e[:, KAPPA:].sum(1) / tot).mean())))
for lid in sorted(acc2):
    v = torch.tensor(acc2[lid])
    print(f"   {lid:4d} {float(v[:,0].mean()):10.4f} {float(v[:,1].mean()):16.4f} "
          f"{float(v[:,2].mean()):16.4f}")
