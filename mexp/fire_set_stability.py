"""Fire-set stability: quantized-storage scan vs fp32 reference.

Quantize-then-calibrate: each arm builds its OWN tier from identical latent
rows, so the quantized arm's conformal zp is recalibrated on quantized scores.
Measured: (a) fired-set agreement across arms, (b) coverage of the exact-fp32
top-8 rows per query (the recall the certificate is supposed to guarantee).
Run once per PYTHONPATH (fp32 main tree / quantized worktree); compare JSON.
"""
import json
import sys

import torch

torch.manual_seed(20260907)
dev = "cuda"

from sglang.srt.layers.attention.vestigekv import defaults as D
from sglang.srt.layers.attention.vestigekv.eviction import blockwise_sigma, select_kept
from sglang.srt.layers.attention.vestigekv.recall_tier import RecallTier

S, H = 262144, 32
with torch.inference_mode():
    base = torch.randn(64, D.LATENT_DIM, device=dev)
    mix = torch.randn(S, 64, device=dev).softmax(-1)
    rows = (mix @ base + 0.3 * torch.randn(S, D.LATENT_DIM, device=dev)).contiguous()

    sigma = blockwise_sigma(rows[:, D.KV_LORA_RANK :])
    keep = select_kept(sigma, D.RHO, S)
    ncal = 64
    q_pos = torch.arange(S - ncal, S, device=dev)
    q_cal = (
        rows[q_pos].unsqueeze(1).expand(ncal, H, D.LATENT_DIM) * 0.02
        + torch.randn(ncal, H, D.LATENT_DIM, device=dev) * 0.05
    ).contiguous()

    tier = RecallTier()
    stats = tier.build(rows, torch.arange(S, device=dev), keep, q_cal, q_pos)

    Q = 256
    qidx = torch.randint(0, S, (Q,), device=dev)
    q = (
        rows[qidx].unsqueeze(1) * 0.02
        + torch.randn(Q, H, D.LATENT_DIM, device=dev) * 0.05
    )

    arch_rows = rows[tier.arch].float()
    out = {
        "zp": round(float(tier.zp), 6),
        "csk_dtype": str(tier.csk.dtype),
        "side_dtype": str(tier.side.dtype),
        "arch": int(tier.arch.numel()),
        "fire": [],
    }
    for i in range(Q):
        fetched = tier.query(q[i])
        fset = sorted(fetched.tolist())
        sc_true = (q[i].float() @ arch_rows.T) * tier.scale
        top_true = set(
            tier.arch[sc_true.topk(8, -1).indices.flatten().unique()].tolist()
        )
        cov = len(set(fset) & top_true) / max(1, len(top_true))
        out["fire"].append({"n": len(fset), "cov": round(cov, 4), "rows": fset[:512]})
json.dump(out, open(sys.argv[1], "w"))
print("zp", out["zp"], "csk", out["csk_dtype"], "arch", out["arch"])
