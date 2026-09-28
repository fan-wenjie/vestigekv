"""Where a tier build's time actually goes, per op, served arm against branch.

    python mexp/kimi/build_profile.py [--rows 15424] [--ncal 64]

VKSTATS reports build=<ms> x<n> and nothing finer, so the 7.5 ms a build costs
at the serving shape has been attributed by guessing twice already -- once to
the eigendecomposition (it was 1.0 ms of it) and once to the operand kernel
(about 1% of it). This runs build() outside the server on the serving shape
under torch.profiler and prints the ops, so the next deletion is chosen from a
table instead.

Both arms are built on identical inputs; the branch arm differs only in the
class, which is the whole point of putting its deletions in their own files.
"""

import argparse

import torch


def _inputs(rows, ncal, dev):
    from sglang.srt.layers.attention.vestigekv import defaults as D

    g = torch.Generator(device=dev).manual_seed(0)
    pool = torch.randn(rows + 4096, D.LATENT_DIM, device=dev,
                       dtype=torch.bfloat16, generator=g)
    row_slots = torch.arange(rows, device=dev, dtype=torch.int32)
    # tier-1 keeps a small fraction; the archive is the rest
    keep = torch.zeros(rows, dtype=torch.bool, device=dev)
    keep[torch.randperm(rows, device=dev, generator=g)[: rows // 32]] = True
    q_cal = torch.randn(ncal, 32, D.LATENT_DIM, device=dev, generator=g)
    q_pos = torch.arange(rows - ncal, rows, device=dev, dtype=torch.long)
    return pool, row_slots, keep, q_cal, q_pos


def _build(cls, args, index_rank):
    from sglang.srt.layers.attention.vestigekv import defaults as D

    pool, row_slots, keep, q_cal, q_pos = args
    tier = cls(r=index_rank, threshold="max", gauss_target=D.CERT_GAUSSIAN_TARGET
               if hasattr(D, "CERT_GAUSSIAN_TARGET") else 0.0)
    return tier.build(pool, row_slots, keep, q_cal, q_pos, conservative=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=15424)
    ap.add_argument("--ncal", type=int, default=64)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--top", type=int, default=14)
    a = ap.parse_args()

    from torch.profiler import ProfilerActivity, profile

    from sglang.srt.layers.attention.vestigekv.branch_tier import BranchRecallTier
    from sglang.srt.layers.attention.vestigekv.recall_tier import RecallTier

    dev = torch.device("cuda")
    args = _inputs(a.rows, a.ncal, dev)

    for name, cls in (("served", RecallTier), ("branch", BranchRecallTier)):
        _build(cls, args, a.rank)  # warm: compile, cusolver init, allocator
        torch.cuda.synchronize()
        e = torch.cuda.Event(True), torch.cuda.Event(True)
        e[0].record()
        for _ in range(a.reps):
            _build(cls, args, a.rank)
        e[1].record()
        torch.cuda.synchronize()
        wall = e[0].elapsed_time(e[1]) / a.reps

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
            _build(cls, args, a.rank)
            torch.cuda.synchronize()
        print(f"\n=== {name}: {wall:.2f} ms per build "
              f"({a.rows} rows, n_cal={a.ncal}, rank={a.rank})")
        print(p.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=a.top,
            max_name_column_width=44))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
