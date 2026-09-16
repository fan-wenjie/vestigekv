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

## The latency A/B: the fence is the whole gap, and the paper's 1.28x reproduces

Same box, same protocol (bs=1, 4k prefill, continuous decode, CUDA graph on,
radix off), server-side metric, ms/token as the median of the scheduler's
gen-throughput lines within +/-2k tokens of each context, p10-p90 in brackets:

| context | dense | origin/vestigekv | current branch | origin speedup | current speedup |
|---|---|---|---|---|---|
| 64k | 4.354 | 4.089 | 4.202 | 1.065 | 1.036 |
| 128k | 4.778 | 4.175 | 4.204 | 1.145 | 1.137 |
| 252k | 5.550 | 4.343 [4.337-4.358] | 4.664 [4.523-4.744] | **1.278** | 1.190 |
| 256k | 5.564 | 4.360 [4.345-4.368] | 4.701 [4.531-4.778] | **1.276** | 1.184 |

Three things follow.

- **The paper's 1.28x at 256k reproduces exactly on this box** once the paper's
  backend is what serves it: 1.276 against a dense arm that itself reproduces
  (5.564 ms/token here, 179.1 tok/s in the paper's own units). The protocol was
  never the explanation.
- **The branch's whole cost is at long context**: within noise to 128k (the
  sign even flips), then 7.3% slower at 252k and 7.8% at 256k. 341 us/step at
  256k, which is the fence cost the profile attributes to the pack (~250 us)
  and the dense attention it forces (~195 us), partly overlapped.
- **The spread confirms the mechanism**: origin's window is [4.345-4.368],
  4.5x tighter than the current tree's [4.531-4.778]. A per-step cost that
  switches on and off is what an intermittent fence looks like; a uniformly
  slower kernel would shift the median without widening the window.

## Production protocol (radix cache on)

The paper's serving numbers were taken with the prefix cache on; this line has
been running `--disable-radix-cache` (quality determinism). `RADIX=on` in
`mexp/kimi/common.sh` selects the production protocol, and the `prod-stream-*`
jobs repeat the 4k->256k timed stream under it for dense, the current tree,
`origin/vestigekv` and the current tree with the fence disabled -- which
separates "the branch costs something" from "the harness protocol differs".

## A trap in the profile harness (fixed)

The first `profile-vestigekv-origin-256k` run profiled its request's *prefill*,
not a 256k decode window: the client watched the newest server log for the
decode progress, the runner reuses a server between same-signature jobs, and
that log therefore belonged to the previous job and still carried its finished
256k stream. The client read 262129 tokens fifteen seconds in and opened the
window immediately. The trace is discarded (its scan kernel reads 1.1 us and
prefill kernels appear in it); the client now streams its own request and
counts the tokens, so no server log is consulted at all.

## Decision on numerics (2026-09-16, owner's call)

Mathematical equivalence is sufficient; a change may regroup the floating-point
accumulation and stop the output matching the dense arm bit for bit. What stays
fixed is the row set: which rows enter a step's softmax, which the registered
tests pin bitwise (fetch sets, keep masks, sigma and basis) and which is what
the paper's parity claims are about. No registered test asserts attention
output numerics, so this unblocks three optimizations that were otherwise
closed: the KV split count sized from the attended rows (done,
--enable-vestigekv-attended-splits), a contiguous arena for the kept tier that
may also reorder it, and an archive ordered into residual-norm bands so the
scan can skip whole bands under a sound bound.

## The grid-strided pack recovers the regression, and locality is not the issue

Timed 4k->256k streams, same box and protocol, server-side ms/token:

| context | dense | origin | current branch | perf branch |
|---|---|---|---|---|
| 128k | 4.778 | 4.166 | 4.204 | 4.179 |
| 252k | 5.550 | 4.343 [4.337-4.358] | 4.664 [4.523-4.744] | 4.391 [4.358-4.424] |
| 256k | 5.564 | 4.360 [4.345-4.368] | 4.701 [4.531-4.778] | 4.395 [4.368-4.437] |

The grid-strided gather alone takes the branch from 7.8% behind the paper's
backend to 0.8% behind it at 256k (speedup over dense 1.184x -> 1.266x against
origin's 1.276x), and the p10-p90 window narrows from 0.247 to 0.069 ms, which
is the intermittent fence going away. 306 us of the 341 us gap was one
serialized copy. The rebuild trigger and the attended splits are therefore
optimizations past parity, not repairs.

**A contiguous arena for the attended tier is not worth building.**
`mexp/kimi/bench_gather.py`, 8192 rows of 576 bf16 out of a 262144-row pool:

| grid | contiguous | scattered (every 32) | random |
|---|---|---|---|
| 8 | 25 GB/s | 25 | 25 |
| 128 | 751 | 750 | 750 |
| 1024 | 2255 | 2232 | 2232 |

At every grid the three index distributions are within 1%, and the best
scattered arm reaches 2232 GB/s against the contiguous arm's 2255. Row-granular
scatter costs nothing at this row width (1152 B is 18 cache lines, and the
pointer load amortizes 1:288); what the sweep does show is that the same read
moves from 25 to 2255 GB/s on parallelism alone. So the attended tier's ~88
GB/s in the decode kernel is a parallelism artifact, not a locality one, and
the lever is the split count (--enable-vestigekv-attended-splits), not the
layout. The arena idea is dropped.

## Queued behind every measurement: recall as search over a static set

Exchangeability plus the frozen ranking make the archive a static point set and
each step's trigger a maximum-inner-product query against it, so the archive can
be clustered and a cluster skipped whole under the cluster form of the same
Cauchy-Schwarz certificate (centroid and radius for the sidecar and sketch
terms, the cluster's largest residual for the certificate term), with the
conformal calibration carried from rows to clusters. This attacks the largest
VestigeKV kernel: the scan is 210 us/step at 256k, grows linearly with the
archive and is bandwidth-bound, so only reading fewer rows makes it faster.
The paper states it as future work (appendix, "Recall as search over a static
set"); implementation starts only after every queued measurement has run, and
lands only if the measured gain justifies changing the algorithm this close to
the deadline (2026-09-26).

## Pending
- `stream-vestigekv-256k-origin`: origin's timed 4k->256k stream, the clean A/B
  against `stream-vestigekv-256k` at 64k/128k/256k.
- `profile-vestigekv-origin-256k` and the 256k window of
  `profile-vestigekv-current`: the same kernel diff at 256k.
- If the 256k latency gap exceeds the window spread (~0.1 ms), bisect with the
  same profile job at 256f320 (fence, int32 tables, eager), 28a133d (rank flag,
  kernel tiles) and 15f9ae1 (int64 CSR).
