"""Can the close CADENCE be short while sigma's WINDOW stays long?

CLOSE_BLOCK sets both today, and that is why lowering it breaks sigma: the
window IS the block, kappa is fixed at 16, so the low-pass cutoff is
window/kappa and moves with it (256 tokens at B=4096, 16 at B=256), while the
31-dim basis grows from 3% of the window to 48%.

They are independent in principle. This measures the decoupled form offline on
the dumps' own pooled index-k: fit the low-pass subspace on the TRAILING W
groups, then take the residual of the newest `cad` groups against it. Cadence
and window become separate numbers, and the question is whether tier 1's
selection is as good as the block-aligned W it came from.

Judged the way the rho sweep was: recall of the true top-2048 by the kept set
(sinks + top-m by sigma + the un-closed tail), at the served rho.
"""
import argparse, glob, sys
import torch

sys.path.insert(0, "/home/user/vestigekv-wt/engine-glm521/python")
from sglang.srt.layers.attention.vestigekv import defaults as D  # noqa: E402

POOL = 4


def basis(W, kappa, dev):
    """Real Fourier basis over a W-point window, 2*kappa-1 columns."""
    t = torch.arange(W, device=dev, dtype=torch.float32)
    cols = [torch.ones(W, device=dev)]
    for k in range(1, kappa):
        cols.append(torch.cos(2 * torch.pi * k * t / W))
        cols.append(torch.sin(2 * torch.pi * k * t / W))
    C = torch.stack(cols, 1)
    return C / C.norm(dim=0, keepdim=True)


def sigma_blocked(K, W, kappa):
    """Today's form: non-overlapping windows of W groups."""
    G, Dm = K.shape
    out = torch.zeros(G, device=K.device)
    C = basis(W, kappa, K.device)
    for a in range(0, (G // W) * W, W):
        R = K[a : a + W]
        out[a : a + W] = (R - C @ (C.T @ R)).norm(dim=1)
    return out


def sigma_trailing(K, W, cad, kappa):
    """Decoupled: fit on the trailing W groups, score the newest `cad`."""
    G, Dm = K.shape
    out = torch.zeros(G, device=K.device)
    C = basis(W, kappa, K.device)
    for b in range(0, (G // cad) * cad, cad):
        a = max(0, b + cad - W)
        R = K[a : b + cad]
        if R.shape[0] < W:          # early blocks: fit on what exists
            Cs = basis(R.shape[0], kappa, K.device)
            res = (R - Cs @ (Cs.T @ R)).norm(dim=1)
        else:
            res = (R - C @ (C.T @ R)).norm(dim=1)
        out[b : b + cad] = res[-cad:]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", default="cd_g2,cd_w2,cd_w3")
    ap.add_argument("--rho", type=float, default=1 / 32)
    a = ap.parse_args()
    torch.set_num_threads(8)
    kappa = D.LOWPASS_KAPPA
    # (label, window groups, cadence groups) -- window*POOL and cad*POOL in tokens
    schemes = [
        ("blocked  W=1024 (B 4096, today)", 1024, 1024),
        ("blocked  W=256  (B 1024)", 256, 256),
        ("blocked  W=64   (B 256, broken)", 64, 64),
        ("trailing W=1024 cad=64  (B 256 cadence)", 1024, 64),
        ("trailing W=1024 cad=256 (B 1024 cadence)", 1024, 256),
    ]
    agg = {s[0]: [] for s in schemes}
    for dname in a.dirs.split(","):
        for f in sorted(glob.glob(f"results/glm53/{dname}/*.pt")):
            d = torch.load(f, map_location="cpu", weights_only=False)
            K = d["index_k"]["deq"].float()
            rows = d["rows"].float()
            T = int(d["seq_len"])
            G = K.shape[0]
            for lab, W, cad in schemes:
                if G < W:
                    continue
                sg = (sigma_blocked(K, W, kappa) if W == cad
                      else sigma_trailing(K, W, cad, kappa))
                # per-group sigma -> per-position, as _index_sigma does
                pos_sigma = sg.repeat_interleave(POOL)[:T]
                closed = (T // (cad * POOL)) * (cad * POOL)
                if closed <= D.SINKS:
                    continue
                m = max(1, round(a.rho * closed))
                r = pos_sigma[:closed].clone()
                r[: D.SINKS] = float("-inf")
                m = min(m, closed - D.SINKS)
                kept = torch.cat([torch.arange(D.SINKS), r.topk(m).indices,
                                  torch.arange(closed, T)])
                hits = []
                for i in range(d["qcal"].shape[0]):
                    q = d["qcal"][i].float()
                    oracle = (rows @ q.T).amax(1)
                    top = set(oracle.topk(min(2048, T)).indices.tolist())
                    hits.append(len(top & set(kept.tolist())) / max(len(top), 1))
                agg[lab].append((kept.numel(), sum(hits) / len(hits)))
    print(f"rho={a.rho:g}, kappa={kappa}, pool={POOL}; recall of the true top-2048 by tier 1")
    print(f"{'scheme':<42}{'kept':>8}{'tier1 recall':>14}")
    for lab, _, _ in schemes:
        v = agg[lab]
        if not v:
            print(f"{lab:<42}{'--':>8}{'--':>14}")
            continue
        print(f"{lab:<42}{sum(x[0] for x in v)/len(v):>8.0f}"
              f"{sum(x[1] for x in v)/len(v):>14.3f}")


if __name__ == "__main__":
    main()
