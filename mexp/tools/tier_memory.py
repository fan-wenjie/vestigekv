#!/usr/bin/env python3
"""How many bytes does one (layer, request) of VestigeKV recall state hold?

    python mexp/tools/tier_memory.py --tree ~/vestigekv-wt/engine-pre-memopt
    python mexp/tools/tier_memory.py --tree ~/vestigekv-wt/engine-memopt

Builds a tier over an N-row closed prefix at GLM-5.3-Flash geometry (rope-less:
512-dim latent, no in-row sidecar), runs one decode-time close (extend +
membership refresh), points a pool-mode in-graph pack at it the way the
backend does (update, then drop the tier's materialised selections), and
reports the allocator's steady footprint for the request-lifetime state and
the peak reached during build. The pack's own capacity tables are persistent
(sized by the pool, not the request) and are subtracted out.

Numbers are per (layer, request); the server holds one per MLA layer per live
request. Run against two trees to price a change; run alone to read the
inventory.
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True, help="engine checkout to measure")
    ap.add_argument("--rows", type=int, default=65536, help="closed prefix length")
    ap.add_argument("--pool", type=int, default=78464, help="KV pool rows (MEM_FRAC 0.95)")
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(os.path.expanduser(a.tree), "python"))

    import torch
    from types import SimpleNamespace

    from sglang.srt.layers.attention.vestigekv import defaults as D
    from sglang.srt.layers.attention.vestigekv.batched_step import BatchedScanPack
    from sglang.srt.layers.attention.vestigekv.geometry import Geometry
    from sglang.srt.layers.attention.vestigekv.recall_tier import RecallTier

    geom = Geometry.from_hf_config(
        SimpleNamespace(
            kv_lora_rank=512, qk_rope_head_dim=0, qk_nope_head_dim=256, index_head_dim=128
        )
    )
    H, R, N, POOL = 32, 64, a.rows, a.pool
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(7)
    kbuf = torch.randn(POOL, geom.latent_dim, device=dev, generator=g, dtype=torch.bfloat16)
    perm = torch.randperm(POOL, device=dev, generator=g)
    slots = perm[:N]
    nkeep = max(1, int(D.RHO * N))
    keep = torch.zeros(N, dtype=torch.bool, device=dev)
    keep[torch.randperm(N, device=dev, generator=g)[:nkeep]] = True
    qcal = torch.randn(32, H, geom.latent_dim, device=dev, generator=g)
    qpos = torch.randint(0, N, (32,), device=dev, generator=g)

    # Pack at the backend's capacity: one pair, nkm = rho*ctx + 3 blocks, arena
    # bounded by the pool. Allocated BEFORE the baseline reading so it counts as
    # persistent, exactly as the server allocates it at graph capture.
    max_ctx = 135168
    nkm = int(D.RHO * max_ctx) + 3 * D.CLOSE_BLOCK
    qbuf = torch.zeros(1, 2, H, geom.latent_dim, device=dev)
    fetch = torch.zeros(1, 2, 2048, dtype=torch.int64, device=dev)
    flen = torch.zeros(1, 2, dtype=torch.int64, device=dev)
    pack = BatchedScanPack.at_capacity(
        1, nkm, 1, R, H, qbuf, fetch, flen,
        torch.zeros_like(flen, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32, device=dev), 1,
        arena=POOL + D.CLOSE_BLOCK,
        pool_bases=[kbuf.data_ptr()], pool_row=kbuf.shape[1], pool_rows=POOL,
        side_from_pool=True, csk_from_tier=True, geom=geom,
    )
    # Process-wide one-time workspaces (cuBLAS, cusolver) are taken by the
    # first GEMM / eigh in the process; warm them here so they are not billed
    # to the first request's build.
    w = torch.randn(512, 512, device=dev)
    torch.linalg.eigh(w.T @ w)
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.memory._record_memory_history(max_entries=200000)
    before = set()
    for seg in torch.cuda.memory._snapshot()["segments"]:
        addr = seg["address"]
        for b in seg["blocks"]:
            if b["state"] == "active_allocated":
                before.add((addr, b["size"]))
            addr += b["size"]

    t = RecallTier(r=R, geom=geom)
    t.build(kbuf, slots, keep, qcal, qpos)
    torch.cuda.synchronize()
    peak_build = torch.cuda.max_memory_allocated() - base
    after_build = torch.cuda.memory_allocated() - base

    # the backend's pack sync: update, then release what the pack now reads through
    pairs = [(0, 0)]
    assert pack.fits(pairs, [t])
    pack.update(pairs, [t])
    for name in ("drop_side", "drop_operands", "drop_kept_rows", "drop_arch"):
        fn = getattr(t, name, None)
        if fn is not None:
            fn()
    torch.cuda.synchronize()
    steady = torch.cuda.memory_allocated() - base

    # one decode-time close: extend the closed prefix by a block, re-decide membership
    torch.cuda.reset_peak_memory_stats()
    more = perm[N : N + D.CLOSE_BLOCK]
    t.extend_closed(kbuf[more], more)
    keep2 = torch.zeros(N + D.CLOSE_BLOCK, dtype=torch.bool, device=dev)
    keep2[torch.randperm(N + D.CLOSE_BLOCK, device=dev, generator=g)[: nkeep + 128]] = True
    t.refresh_membership(keep2, kbuf)
    assert pack.fits(pairs, [t])
    pack.update(pairs, [t])
    for name in ("drop_side", "drop_operands", "drop_kept_rows", "drop_arch"):
        fn = getattr(t, name, None)
        if fn is not None:
            fn()
    torch.cuda.synchronize()
    peak_close = torch.cuda.max_memory_allocated() - base
    steady_close = torch.cuda.memory_allocated() - base

    rows = N
    print(f"tree {a.tree}  rows {N}  pool {POOL}")
    print(f"  steady after build+pack sync : {steady/2**20:8.2f} MiB  ({steady/rows:6.1f} B/row)")
    print(f"  peak during build            : {peak_build/2**20:8.2f} MiB  (+{(peak_build-after_build)/2**20:.2f} MiB transient)")
    print(f"  peak during close+pack sync  : {peak_close/2**20:8.2f} MiB  (+{(peak_close-steady_close)/2**20:.2f} MiB transient)")
    print(f"  steady after close           : {steady_close/2**20:8.2f} MiB  ({steady_close/(rows+D.CLOSE_BLOCK):6.1f} B/row)")
    # Every ACTIVE allocator block that did not exist at the baseline, with the
    # frame in the tier/pack code that made it: plain tensors are not gc-
    # tracked, so the allocator's own snapshot is the only complete inventory.
    snap = torch.cuda.memory._snapshot()
    blocks = []
    for seg in snap["segments"]:
        addr = seg["address"]
        for b in seg["blocks"]:
            if b["state"] == "active_allocated" and (addr, b["size"]) not in before:
                frames = [
                    f'{os.path.basename(f["filename"])}:{f["line"]}'
                    for f in b.get("frames", [])
                    if "vestigekv" in f["filename"]
                ]
                blocks.append((b["size"], " < ".join(frames[:3]) or "?"))
            addr += b["size"]
    blocks.sort(reverse=True)
    total = 0
    for size, where in blocks:
        total += size
        if size >= 2**18:
            print(f"    {size/2**20:7.2f} MiB  {where}")
    print(f"  active blocks new since baseline: {total/2**20:.2f} MiB across {len(blocks)}")
    return 0




if __name__ == "__main__":
    sys.exit(main())
