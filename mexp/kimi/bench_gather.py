"""Gather bandwidth of the attended tier: contiguous rows against scattered ones.

    python mexp/kimi/bench_gather.py [--rows 8192] [--pool 262144] [--iters 50]

The decode profile at 256k shows the attention reading its ~8k kept rows at about
88 GB/s while the same kernel reading the whole 262144-row set reaches about
1.4 TB/s. Two explanations fit: the kept rows are scattered (one row every 32, so
every 1152-byte row opens its own DRAM page), or the read is split into too many
pieces to saturate anything (the split count is sized from the DENSE length).

This isolates the first. One Triton kernel reads `rows` rows of 576 bf16 through an
index list and reduces them, so the only difference between the arms is where the
rows sit; the grid sweep says how much of any gap is parallelism rather than
locality. An arena that keeps the attended tier contiguous is worth building only
if the scattered arm stays far below the contiguous one at its best grid.
"""

import argparse

import torch
import triton
import triton.language as tl

ROW = 576  # MLA latent row: 512 content + 64 sidecar, bf16


@triton.jit
def _gather_rows_kernel(idx_ptr, pool_ptr, out_ptr, n_rows, ROW: tl.constexpr, BLOCK: tl.constexpr):
    g = tl.program_id(0)
    G = tl.num_programs(0)
    cols = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in range(g, n_rows, G):
        row = tl.load(idx_ptr + i).to(tl.int64)
        acc += tl.load(pool_ptr + row * ROW + cols, mask=cols < ROW, other=0.0).to(tl.float32)
    tl.store(out_ptr + g * BLOCK + cols, acc, mask=cols < BLOCK)


def time_gather(idx, pool, grid, iters):
    out = torch.zeros(grid, 1024, dtype=torch.float32, device="cuda")
    run = lambda: _gather_rows_kernel[(grid,)](
        idx, pool, out, idx.numel(), ROW=ROW, BLOCK=1024, num_warps=8
    )
    for _ in range(5):
        run()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        run()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000.0  # us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=8192, help="attended rows (256k context at rho=1/32)")
    ap.add_argument("--pool", type=int, default=262144, help="pool rows the attended ones sit in")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--grids", default="8,32,64,128,256,512,1024")
    args = ap.parse_args()
    pool = torch.randn(args.pool, ROW, dtype=torch.bfloat16, device="cuda").flatten()
    stride = max(args.pool // args.rows, 1)
    arms = {
        "contiguous": torch.arange(args.rows, dtype=torch.int32, device="cuda"),
        f"scattered (every {stride})": torch.arange(0, stride * args.rows, stride, dtype=torch.int32, device="cuda")
        % args.pool,
        "random": torch.randperm(args.pool, device="cuda")[: args.rows].to(torch.int32),
    }
    payload = args.rows * ROW * 2
    print(f"{args.rows} rows x {ROW} bf16 = {payload / 2**20:.1f} MiB out of a {args.pool}-row pool, "
          f"{args.iters} iters; GB/s (us)")
    print(f"{'grid':>6s}" + "".join(f"{name:>26s}" for name in arms))
    best = dict.fromkeys(arms, 0.0)
    for grid in (int(g) for g in args.grids.split(",")):
        row = f"{grid:6d}"
        for name, idx in arms.items():
            us = time_gather(idx, pool, grid, args.iters)
            gbs = payload / (us * 1e-6) / 1e9
            best[name] = max(best[name], gbs)
            row += f"{gbs:18.0f} ({us:5.0f})"
        print(row)
    print("best: " + "  ".join(f"{k} {v:.0f} GB/s" for k, v in best.items()))
    ratio = best["contiguous"] / max(best[f"scattered (every {stride})"], 1e-9)
    print(f"contiguous / scattered at each arm's best grid: {ratio:.2f}x")


if __name__ == "__main__":
    main()
