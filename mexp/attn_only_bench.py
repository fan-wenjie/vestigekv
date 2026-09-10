"""Attention-only pseudo-continuous decode: the method's term in isolation.

Derived from sglang's own decode-attention benchmark
(benchmark/kernels/decoding_attention_triton/triton_flashinfer_cudnn.py):
same decode_attention_fwd kernel, same torch.utils.benchmark.Timer timing
convention. The vestigekv per-step pipeline and the pseudo-continuous close
semantics have no sglang counterpart -- that part is experiment-side by rule.

Strips B entirely (no MoE/KDA/network): both arms drive the SAME sglang MLA
decode kernel over a synthetic latent pool, differing only in the index set --
dense attends [0..S), vestigekv runs its real per-step pipeline (fused
prologue + scan + compaction -> kept+fired CSR) with decode-time closes every
CLOSE_BLOCK, exactly the serving semantics. Microsecond-precision slope and
crossover, immune to launch/network state. bs>1 exposes the per-request
(unbatchable) nature of attention directly.

  python attn_only_bench.py [--layers 7] [--bs 1] [--points 16k,64k,128k,256k]
"""
import argparse
import torch
import torch.utils.benchmark as benchmark

import sglang.srt.layers.attention.vestigekv.defaults as D
from sglang.kernels.ops.attention.decode_attention import decode_attention_fwd
from sglang.srt.layers.attention.vestigekv.batched_step import BatchedScanPack
from sglang.srt.layers.attention.vestigekv.eviction import blockwise_sigma, select_kept

H, LAT, KV = 32, 576, 512
SPLITS = 128  # matches sglang _mla_decode_kv_splits_cap at long ctx (SM-count scaled)


def _attn(q, kbuf, indptr, indices, logits, lse, o, nsplits):
    decode_attention_fwd(
        q, kbuf.view(-1, 1, LAT), kbuf.view(-1, 1, LAT)[:, :, :KV], o,
        indptr, indices, logits, lse, nsplits, SPLITS,
        sm_scale=D.ATTN_SCALE, k_scale=1.0, v_scale=1.0, has_mla=True,
    )


def bench_point(S, layers, bs, dev="cuda"):
    torch.manual_seed(S)
    kbuf = torch.randn(bs * (S + 8192), LAT, device=dev, dtype=torch.bfloat16)
    q = torch.randn(bs, H, LAT, device=dev, dtype=torch.bfloat16)
    o = torch.empty(bs, H, KV, device=dev, dtype=torch.bfloat16)
    logits = torch.empty(bs, H, SPLITS, KV, device=dev, dtype=torch.float32)
    lse = torch.empty(bs, H, SPLITS + 1, device=dev, dtype=torch.float32)
    nsplits = torch.full((bs,), SPLITS, device=dev, dtype=torch.int32)

    # dense arm: contiguous [r*(S+8192), ...+S)
    d_indptr = torch.arange(0, (bs + 1) * S, S, device=dev, dtype=torch.int32)
    d_indices = torch.cat(
        [torch.arange(r * (S + 8192), r * (S + 8192) + S, device=dev, dtype=torch.int32)
         for r in range(bs)]
    )

    # vk arm state per (layer, request): tier + kept table (built once at "prefill")
    class T:  # minimal tier facade the pack consumes
        pass

    pairs, tiers = [], []
    kept_all = []
    for li in range(layers):
        for r in range(bs):
            base = r * (S + 8192)
            rows = kbuf[base : base + S]
            sig = blockwise_sigma(rows[:, KV:].float())
            keep = select_kept(sig, D.RHO, (S // D.CLOSE_BLOCK) * D.CLOSE_BLOCK)
            keep = torch.cat([keep, torch.ones(S - keep.numel(), dtype=torch.bool, device=dev)])
            t = T()
            t.kept_rows = rows[keep].to(torch.bfloat16)
            t.V = torch.linalg.qr(torch.randn(KV, D.INDEX_RANK, device=dev))[0].T.contiguous()
            arch_idx = (~keep).nonzero().flatten()
            af = rows[arch_idx].float()
            csk = af[:, :KV] @ t.V.T
            t.side = rows[arch_idx][:, KV:].to(torch.bfloat16)
            t.csk = csk.half()
            t.rho = (af[:, :KV] - csk @ t.V).norm(dim=-1)
            t.arch = (arch_idx + base).to(torch.int64)
            t.thr_g = 1.0
            t.zp = 4.0
            t.scale = D.ATTN_SCALE
            t.r = D.INDEX_RANK
            t.version = 0
            pairs.append((li, r))
            tiers.append(t)
            if li == 0:
                kept_all.append((keep.nonzero().flatten() + base).to(torch.int32))
    W = 4096
    qbuf = torch.randn(layers, bs, H, LAT, device=dev)
    fetch = torch.zeros(layers, bs, W, dtype=torch.int64, device=dev)
    flen = torch.zeros(layers, bs, dtype=torch.int64, device=dev)
    pack = BatchedScanPack(pairs, tiers, qbuf, fetch, flen, H)
    nk = torch.tensor([k.numel() for k in kept_all], device=dev)
    v_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=dev)
    v_indptr[1:] = (nk + W).cumsum(0).to(torch.int32)
    v_indices = torch.zeros(int(v_indptr[-1]), dtype=torch.int32, device=dev)
    off = 0
    for r in range(bs):
        n = int(nk[r])
        v_indices[off : off + n] = kept_all[r]
        off += n + W

    def vk_step():
        pack.run()  # prologue+scan+compact for ALL layer-pairs (per-step cost)
        for _li in range(layers):  # attention per layer over kept(+W slots)
            _attn(q, kbuf, v_indptr, v_indices, logits, lse, o, nsplits)

    def dense_step():
        for _li in range(layers):
            _attn(q, kbuf, d_indptr, d_indices, logits, lse, o, nsplits)

    def t(fn, repeats=30):
        # sglang benchmark convention (triton_flashinfer_cudnn.benchmark_forward)
        timer = benchmark.Timer(stmt="fn()", globals={"fn": fn})
        m = timer.timeit(repeats)
        return m.mean * 1000

    return t(dense_step), t(vk_step)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=7)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--points", default="16384,65536,131072,262144")
    a = ap.parse_args()
    print(f"attention-only pseudo-decode  layers={a.layers} bs={a.bs}")
    print(f"{'S':>8} {'dense ms':>9} {'vk ms':>9} {'speedup':>8}")
    for S in [int(x) for x in a.points.split(",")]:
        d, v = bench_point(S, a.layers, a.bs)
        print(f"{S:>8} {d:9.3f} {v:9.3f} {d/v:7.2f}x")
