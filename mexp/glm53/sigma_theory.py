"""Verify the four claims about sigma as a query-free screen, on real dumps.

1. leverage      (CC^T)_tt == (2kappa-1)/B exactly, for the basis the kernel builds
2. commutation   Q_perp(K q) == (Q_perp K) q, so the high-frequency part of the
                 logit sequence IS R q and |.| <= sigma_t ||q||
3. isotropy      E_q[(Q_perp l)_t^2] == sigma_t^2 / d under q ~ N(0, I/d)
4. does it bite  the screen needs the LOW-frequency part of the score, which is
                 query-dependent. So measure how much of the indexer score's energy
                 is low-frequency: that is what decides whether a cheap coarse
                 representative can supply the trend the bound is taken around.
"""

import glob
import sys

import torch

sys.path.insert(0, "mexp/glm53")
from dsa_selection_stats import score

sys.path.insert(0, "/home/user/vestigekv-wt/engine-glm521/python")
from sglang.srt.layers.attention.vestigekv.sigma_fused import _basis

B, KAPPA, DIRS = 256, 16, "cd_w3,cd_w2,cd_g2"
torch.set_num_threads(8)


def lowpass(x, kappa):
    """Project along dim 0 onto the first kappa rFFT bins."""
    f = torch.fft.rfft(x.double(), dim=0)
    f[kappa:] = 0
    return torch.fft.irfft(f, n=x.shape[0], dim=0)


print("== 1. leverage of the kernel's own basis")
for kap in (1, 2, 4, 16):
    C = _basis(B, kap, "cpu").double()[:, : 2 * kap - 1]
    lev = (C * C).sum(1)
    print(f"   kappa={kap:2d}  (CC^T)_tt  min {lev.min():.8f}  max {lev.max():.8f}"
          f"   (2k-1)/B = {(2*kap-1)/B:.8f}   spike retains {1-(2*kap-1)/B:.4f}")

f = sorted(glob.glob("results/glm53/cd_w3/*.pt"))[0]
d = torch.load(f, map_location="cpu", weights_only=False)
gk = d["index_k"]["deq"].double()
K = gk[:B]
C = _basis(B, KAPPA, "cpu").double()[:, : 2 * KAPPA - 1]
Qp = torch.eye(B, dtype=torch.double) - C @ C.T
R = Qp @ K
sigma = R.norm(dim=-1)

print("\n== 2. commutation, and the bound, on a real block")
q = d["qidx"][0].double() if not isinstance(d["qidx"], (list, tuple)) else d["qidx"][0].double()
qh = q[0]
lhs, rhs = Qp @ (K @ qh), R @ qh
print(f"   ||Q_perp(Kq) - (Q_perp K)q||_inf = {float((lhs-rhs).abs().max()):.3e}")
print(f"   bound holds for every row:        {bool((lhs.abs() <= sigma*qh.norm()+1e-9).all())}")
print(f"   median |high-freq logit| / (sigma*||q||) = "
      f"{float((lhs.abs()/(sigma*qh.norm())).median()):.4f}   (1.0 = tight)")
print(f"   max over rows                           = "
      f"{float((lhs.abs()/(sigma*qh.norm())).max()):.4f}")

print("\n== 3. isotropic identity, 4096 draws")
g = torch.Generator().manual_seed(0)
dd = K.shape[1]
Z = torch.randn(dd, 4096, generator=g, dtype=torch.double) / dd**0.5
emp = ((R @ Z) ** 2).mean(1)
pred = sigma**2 / dd
print(f"   max relative error |emp-pred|/pred = {float(((emp-pred).abs()/pred).max()):.4f}")

print("\n== 4. where the indexer score's energy sits (does the screen bite?)")
print(f"   {'lid':>4s} {'low-freq share of s':>20s} {'of |s-mean|':>13s} "
      f"{'corr(sigma, |s-trend|)':>23s}")
files = []
for dname in DIRS.split(","):
    files += sorted(glob.glob(f"results/glm53/{dname}/*.pt"))
rows = {}
for fp in files:
    dd_ = torch.load(fp, map_location="cpu", weights_only=False)
    g2 = dd_["index_k"]["deq"].double()
    G = g2.shape[0]
    nb = G // B
    if nb == 0:
        continue
    Cn = nb * B
    qs, gates = dd_["qidx"], dd_["qgate"]
    if not isinstance(qs, (list, tuple)):
        qs, gates = [qs], [gates]
    sg = R.new_zeros(0)
    sigs = []
    for qq, gg in list(zip(qs, gates))[:4]:
        s = score(qq, gg, g2).double()[:Cn]
        sb = s.reshape(nb, B)
        lo = lowpass(sb.T, KAPPA).T
        hi = sb - lo
        share = float((lo**2).sum() / (sb**2).sum())
        share_c = float((lo - sb.mean())[None].pow(2).sum() / (sb - sb.mean()).pow(2).sum())
        # sigma for these blocks
        sigb = torch.stack([(torch.eye(B, dtype=torch.double) - C @ C.T) @ g2[i*B:(i+1)*B]
                            for i in range(nb)]).norm(dim=-1)
        a, b = sigb.flatten(), hi.abs().flatten()
        a = a - a.mean(); b = b - b.mean()
        corr = float((a*b).sum() / (a.norm()*b.norm()))
        rows.setdefault(int(dd_["lid"]), []).append((share, share_c, corr))
for lid in sorted(rows):
    v = torch.tensor(rows[lid])
    print(f"   {lid:4d} {float(v[:,0].mean()):20.4f} {float(v[:,1].mean()):13.4f} "
          f"{float(v[:,2].mean()):23.4f}")
