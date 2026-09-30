#!/usr/bin/env python3
"""Can the query's RMSNorm give a weight-only sketch basis and a sound bound?

    python mexp/kimi/analyze_rmsnorm_basis.py [--layers 3,15,26]

The query projection's input is RMSNorm'd. On this model that is NOT a
query-specific norm -- q_lora_rank is None, so there is no q_a_proj/q_a_layernorm
pair at all -- it is the block's own input_layernorm over hidden_size=2304. The
consequence is the same either way:

    u = h / rms(h),   ||u|| = sqrt(2304) = 48   EXACTLY

and the absorbed content query is a fixed linear image of u,

    q_c[head] = (u * g) @ W_q_nope[head] @ W_kc[head] =: u @ A[head]

with A known at load time. Two things follow that the current design does not
have.

1. A SOUND a priori bound. The certificate today is a conformal quantile at
   0.90 -- zp/sqrt(448) = 0.378 of Cauchy-Schwarz -- so it is an estimate, and
   leak_cert >= leak_true fails on 6.8% of steps. With ||u|| fixed,

       ||q_res|| = ||u A P|| <= 48 * sigma_max(A P),   P = I - V'V

   is a worst-case bound needing no calibration at all.

2. A basis that needs no calibration sample. Minimising E||q_res||^2 over rank-r
   projectors gives V = the top-r RIGHT SINGULAR SUBSPACE OF A when u is
   isotropic, which is computable from the weights once at load. That matters
   because the present basis is fitted on the prompt's last queries, which are
   out of distribution for answer steps (0.1% against 22.85% miss) -- the single
   defect behind every dead detector in this work. A weight-derived basis has no
   calibration distribution to be shifted from, and it also deletes the per-build
   cusolver eigendecomposition (~4 ms/layer, ~20 ms/request on the token path).

The catch, and what this script measures: u is NOT isotropic. Rank-64 was
measured to capture 91-99% of the QUERY covariance, which is strongly
anisotropic, so the weight-only subspace optimises for the wrong metric. The
question is how much tightness that costs, and the answer is the singular
spectrum of A: if the top 64 of 512 directions already carry most of ||A||_F^2,
the weight-only basis is close to free.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import torch
from safetensors import safe_open

ROOT = os.path.expanduser(
    "~/.cache/huggingface/hub/models--moonshotai--Kimi-Linear-48B-A3B-Instruct/"
    "snapshots/*/"
)


def load_layer(lid, keys_wanted):
    base = glob.glob(ROOT)[0]
    idx = json.load(open(os.path.join(base, "model.safetensors.index.json")))
    wm = idx["weight_map"]
    out = {}
    for k in keys_wanted:
        name = f"model.layers.{lid}.{k}"
        if name not in wm:
            continue
        with safe_open(os.path.join(base, wm[name]), framework="pt") as f:
            out[k] = f.get_tensor(name)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="3,15,26")
    ap.add_argument("--rank", type=int, default=64)
    a = ap.parse_args()

    base = glob.glob(ROOT)[0]
    cfg = json.load(open(os.path.join(base, "config.json")))
    H = cfg["num_attention_heads"]
    dn, dr = cfg["qk_nope_head_dim"], cfg["qk_rope_head_dim"]
    kv = cfg["kv_lora_rank"]
    hid = cfg["hidden_size"]
    print(f"hidden={hid} heads={H} nope={dn} rope={dr} kv_lora={kv} "
          f"||u||=sqrt({hid})={hid ** 0.5:.3f}")

    for lid in [int(x) for x in a.layers.split(",")]:
        w = load_layer(lid, [
            "input_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.kv_b_proj.weight",
        ])
        if "self_attn.q_proj.weight" not in w:
            print(f"\n=== layer {lid}: no q_proj (not an MLA layer)")
            continue
        g = w["input_layernorm.weight"].float()             # [hid]
        Wq = w["self_attn.q_proj.weight"].float()           # [H*(dn+dr), hid]
        Wkv = w["self_attn.kv_b_proj.weight"].float()       # [H*(dn+v), kv]
        Wq = Wq.view(H, dn + dr, hid)
        Wkc = Wkv.view(H, -1, kv)[:, :dn, :]                # [H, dn, kv]

        print(f"\n=== layer {lid}")
        tot_top, tot_all, smax = 0.0, 0.0, 0.0
        for h in range(H):
            # A = diag(g) @ Wq_nope^T @ Wkc  ->  [hid, kv]
            A = (Wq[h, :dn, :] * g[None, :]).T @ Wkc[h]
            s = torch.linalg.svdvals(A)
            tot_top += float((s[: a.rank] ** 2).sum())
            tot_all += float((s ** 2).sum())
            smax = max(smax, float(s[0]))
        frac = tot_top / tot_all
        print(f"  top-{a.rank} of {kv} singular directions carry "
              f"{100 * frac:.2f}% of ||A||_F^2  (mean over {H} heads)")
        print(f"  sigma_max(A) = {smax:.4f}  ->  ||q_c|| <= "
              f"{hid ** 0.5 * smax:.2f} a priori, with no calibration")
        print(f"  residual share left to the certificate: {100 * (1 - frac):.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
