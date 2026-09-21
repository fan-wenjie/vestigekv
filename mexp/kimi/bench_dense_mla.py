"""Correctness and speed of the page-table dense MLA decode against the path it replaces.

    python mexp/kimi/bench_dense_mla.py [--seq 262144]

A fenced lane attends its whole row set. The deployed path copies that row set
into the CSR (seq int64, written by the pack and read back by attention) and the
stock kernel reads it from there; `vestigekv.dense_mla` reads the page table
directly. This checks the second against a torch reference and times both
halves that differ:

  materialize  the copy the deployed path makes (page table -> int64 CSR slice)
  dense_mla    the replacement kernel, which does the attention and no copy

The attention itself is common to both, so the saving is the materialize row;
the dense_mla row says what the whole fenced step costs when the copy is gone.
Run it when no timed job is measuring -- it saturates the GPU.
"""

import argparse
import os
import sys
import time

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.environ.get("ENGINE", os.path.join(ROOT, "engine")), "python"))

from sglang.srt.layers.attention.vestigekv.dense_mla import (  # noqa: E402
    ADDR_AFFINE,
    ADDR_PAGE_TABLE,
    dense_mla_decode,
)


def timed(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000.0  # us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=262144)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--pool", type=int, default=400000)
    ap.add_argument("--row", type=int, default=576)
    ap.add_argument("--nsplit", default="16,32,64,128")
    ap.add_argument("--block-n", default="32,64,128")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(0)
    pool = torch.randn(args.pool, args.row, dtype=torch.bfloat16, device=dev, generator=g)
    # a fresh long request owns a contiguous run of pool rows, as the allocator gives it
    page = torch.arange(args.seq, dtype=torch.int32, device=dev)
    q = torch.randn(args.heads, args.row, dtype=torch.float32, device=dev, generator=g)
    kbase = torch.tensor([pool.data_ptr()], dtype=torch.int64, device=dev)
    loc = torch.tensor([args.seq - 1], dtype=torch.int64, device=dev)
    seq = torch.tensor([args.seq], dtype=torch.int64, device=dev)
    fenced = torch.tensor([1 << 30], dtype=torch.int32, device=dev)  # a fired count above any capacity
    scale = 1.0 / (args.row**0.5)
    payload = args.seq * args.row * 2

    rows = pool[: args.seq]
    ref_s = (q.to(torch.bfloat16).float() @ rows.float().T) * scale
    ref = torch.softmax(ref_s, -1) @ rows[:, : args.row - 64].float()

    print(f"seq={args.seq} heads={args.heads} row={args.row}: {payload / 2**20:.0f} MiB of rows")
    best = {}
    for name, addr in (("page-table", ADDR_PAGE_TABLE), ("affine", ADDR_AFFINE)):
        for ns in (int(x) for x in args.nsplit.split(",")):
            for bn in (int(x) for x in args.block_n.split(",")):
                run = lambda: dense_mla_decode(
                    q, page, kbase, loc, seq, fenced, scale=scale, capacity=4096,
                    addr=addr, nsplit=ns, block_n=bn,
                )
                err = (run() - ref).abs().max().item() / ref.abs().max().item()
                us = timed(run, args.iters)
                gbs = payload / (us * 1e-6) / 1e9
                print(f"  {name:10s} nsplit={ns:4d} block_n={bn:4d}  {us:8.1f} us  {gbs:7.0f} GB/s  rel err {err:.2e}")
                if name not in best or us < best[name][0]:
                    best[name] = (us, ns, bn)
    # the exit an unfenced lane pays: the launch happens either way in a graph
    unfenced = torch.zeros(1, dtype=torch.int32, device=dev)
    us = timed(lambda: dense_mla_decode(q, page, kbase, loc, seq, unfenced, scale=scale,
                                        capacity=4096, addr=ADDR_PAGE_TABLE,
                                        nsplit=best["page-table"][1], block_n=best["page-table"][2]), args.iters)
    print(f"  not fenced (early exit, the cost every step pays)  {us:8.1f} us")
    csr = torch.empty(args.seq, dtype=torch.int64, device=dev)
    def materialize():
        csr.copy_(page.to(torch.int64))
    us = timed(materialize, args.iters)
    print(f"  materialize (the copy this replaces)      {us:8.1f} us  "
          f"{args.seq * 8 / (us * 1e-6) / 1e9:7.0f} GB/s written")
    for name, (us, ns, bn) in best.items():
        print(f"best {name:10s} {us:8.1f} us at nsplit={ns} block_n={bn}")
    if len(best) == 2:
        print(f"affine / page-table at each best: {best['affine'][0] / best['page-table'][0]:.3f}x")


if __name__ == "__main__":
    main()
