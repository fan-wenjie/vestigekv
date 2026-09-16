# Performance audit: origin/vestigekv vs the current branch (Kimi Linear, 2x RTX PRO 6000)

Question: do the 443 commits between `origin/vestigekv` (281abac, the paper's
backend) and the current branch cost per-step decode time? Method: both trees
serve the same protocol on this box (4k prefill, continuous decode, bs=1,
CTX=270336, CUDA graph on, stats off), the current tree from `engine/` and the
reference from a detached worktree at `~/vestigekv-wt/engine-origin` via
`ENGINE=`. Two instruments:

- latency (ground truth): `stream-vestigekv-256k-origin` against the already
  recorded `stream-vestigekv-256k`, read with
  `mexp/glm53/stream_curve.py --source server` (the paper's metric).
- kernels (attribution): `profile-vestigekv-origin*` / `profile-vestigekv-current*`
  run the server's own torch profiler for 200 forward steps at 128k and 256k
  (`mexp/kimi/profile_stream.py`), diffed by kernel with
  `mexp/kimi/kernel_diff.py`.

## Result at 128k (2026-09-16, 200 profiled steps per tree)

Per-step wall time from the kernel span of the trace: origin 4383.9 us,
current 4409.3 us, **+25.4 us/step (+0.58%)**. GPU kernel time per step:
5299.2 vs 5324.5 us (kernels overlap, so this exceeds the step).

| kernel | origin us/step | current us/step | diff |
|---|---|---|---|
| ncclDevKernel_AllReduce_Sum_bf16_RING_LL | 552.8 | 583.4 | +30.6 |
| _pack_csr_gather_kernel | 11.0 | 19.8 | +8.8 |
| _fwd_grouped_kernel_stage1 | 81.7 | 86.5 | +4.9 |
| _fwd_kernel_stage2 | 178.5 | 174.2 | -4.3 |
| _scan_batched_kernel | 119.5 | 106.5 | -13.0 |

Everything else is within 0.2 us/step. Reading:

- `_pack_csr_gather_kernel` nearly doubles. This is the one attributable
  regression: the CSR index buffers went back to the base decode kernel's
  int64 (commit "CSR dtype", the fix for the 5.1M-row pool overflow), so the
  pack writes 8-byte row ids. It is 0.2% of the step and it buys correct
  addressing above 3.7M rows -- not a candidate for reversal.
- The NCCL all-reduce is a synchronization sink, not work: its duration
  absorbs rank skew, so +30 us there is only meaningful if it reproduces at
  256k with the same sign.
- The scan and stage-2 kernels are *faster* on the current tree (kernel tiles
  and the rank flag), which offsets most of the above.

At 128k the branch therefore costs ~0.6% of a decode step, and the 1.28x
(paper, P1 v2 protocol) vs 1.19x (this box, this protocol) gap against dense
is not explained by it. The 256k pair and the timed origin stream decide
whether the gap grows with context or is a protocol difference.

## The 256k finding: the overflow fence, and the rank skew it creates

Collective time is not work: an all-reduce absorbs the wait for the slower rank.
`mexp/kimi/rank_skew.py` splits the two apart per rank (us per step, 200 steps):

| trace | TP0 nccl | TP0 compute | TP1 nccl | TP1 compute | skew | step |
|---|---|---|---|---|---|---|
| origin @128k | 582.8 | 4716.5 | 461.6 | 4724.7 | 8.2 | 4383.9 |
| current @128k | 613.4 | 4711.0 | 483.9 | 4700.9 | 10.1 | 4409.3 |
| current @256k | 915.4 | 4850.5 | 452.1 | 5316.4 | 466.0 | 4845.4 |

**At 128k the branch costs nothing in compute**: the slower rank's compute is
4724.7 us on origin and 4711.0 us on the current tree, i.e. the current tree is
13.7 us/step *faster* in real work. The +25 us of step time is collective
jitter, and the `_pack_csr_gather_kernel` regression (int64 CSR) is paid back by
the faster scan and stage-2 kernels.

**At 256k the two ranks diverge by 466 us/step**, and the whole divergence is
two kernels (TP1 minus TP0, us/step):

| kernel | TP0 | TP1 | diff |
|---|---|---|---|
| _pack_csr_gather_kernel | 13.5 | 264.2 | +250.7 |
| _fwd_grouped_kernel_stage1 | 106.5 | 301.4 | +194.9 |

Both are the signature of the **dense-fallback fence**: when a scan fires more
rows than the recall capacity, the pack replaces that lane's kept-plus-fetched
row set with the request's *entire* row set, so the gather and the attention
run over 256k rows instead of the ~8k kept ones. A pack 20x heavier on one rank
means most of TP1's MLA layers fence while TP0's do not -- the ranks hold
different heads, so they fire different row sets and overflow independently.
TP0 then waits for TP1 inside the all-reduce (915 vs 452 us).

The fence does not exist on `origin/vestigekv`: there an overflow is **truncated**
to the first `FETCH_WIDTH_UNCAPPED`=4096 fired rows and counted in the histogram.
It arrived with commit 256f320 (capacity, flags, int32 tables), default on. So
this is a genuine cost the branch added at long context, it is bounded by how
often the certificate over-fires, and it is the same defect the quality side
sees as a high fallback rate. `--disable-vestigekv-recall-overflow-fallback`
restores the origin behaviour and is measured as its own arm below.

## Production protocol (radix cache on)

The paper's serving numbers were taken with the prefix cache on; this line has
been running `--disable-radix-cache` (quality determinism). `RADIX=on` in
`mexp/kimi/common.sh` selects the production protocol, and the `prod-stream-*`
jobs repeat the 4k->256k timed stream under it for dense, the current tree,
`origin/vestigekv` and the current tree with the fence disabled -- which
separates "the branch costs something" from "the harness protocol differs".

## Pending
- `stream-vestigekv-256k-origin`: origin's timed 4k->256k stream, the clean A/B
  against `stream-vestigekv-256k` at 64k/128k/256k.
- `profile-vestigekv-origin-256k` and the 256k window of
  `profile-vestigekv-current`: the same kernel diff at 256k.
- If the 256k latency gap exceeds the window spread (~0.1 ms), bisect with the
  same profile job at 256f320 (fence, int32 tables, eager), 28a133d (rank flag,
  kernel tiles) and 15f9ae1 (int64 CSR).
