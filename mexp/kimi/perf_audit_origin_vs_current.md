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

## Negative: sizing the KV splits from the attended rows (2026-09-16)

`--enable-vestigekv-attended-splits` clamps what the decode kernel's split
count is sized from, 262144 rows at 256k down to the attended bound near
12288. Timed 256k streams, server-side ms/token, against the same tree
without it:

| context | attended splits | perf branch | ratio |
|---|---|---|---|
| 128k | 4.146 [4.115-4.238] | 4.179 [4.138-4.212] | 0.992 |
| 252k | 4.478 [4.431-4.548] | 4.391 [4.358-4.424] | 1.020 |
| 256k | 4.517 [4.470-4.559] | 4.395 [4.368-4.437] | 1.028 |

The gains at and below 128k sit inside overlapping windows; the 2.8%
regression at 256k does not. Per kernel at 256k, `_fwd_grouped_kernel_stage1`
rises 107.6 -> 132.8 us and `_fwd_kernel_stage2` does not move.

So the model behind the flag was wrong in both halves: fewer splits starve the
read of parallelism rather than feeding each split more work, and the combine
stage was never dominated by the number of partials. The base's sizing is
already near optimal. The flag stays off and is not adopted.

With this and the gather microbenchmark, both explanations offered for the
attended tier's ~88 GB/s are closed: not locality (contiguous, every-32 and
random index lists are within 1% at every grid) and not over-splitting. What
remains is stage1 itself, which is not a pure read -- it carries a 576-wide
multi-head dot per row -- so it is occupancy or arithmetic bound, and it is
upstream code outside this backend.

## Negative: a sound ball bound cannot prune the scan (2026-09-16)

`mexp/kimi/bucket_offline.py` measures, on dumped calibration snapshots, what
fraction of the archive survives the cluster form of the certificate. Three
snapshots, 59518 archived rows each, k-means in sketch space:

| clusters | rows still read (mean) | median | rows the certificate fires |
|---|---|---|---|
| 256 | 90.9% | 100.0% | 1754 of 59518 |
| 1024 | 87.9% | 98.6% | 1739 of 59518 |

Quantile schemes on one or two coordinates are worse (99-100% at B up to
16384): they cut slabs, not balls, and the unconstrained coordinates keep the
radius. The obstacle is dimensional, not implementational: the bound's slack is
the sketch query norm times the ball radius, and a radius falls as B^(-1/r), so
at r=64 raising B from 256 to 16384 shrinks it by about 4%. The soundness
assertion passes everywhere, so the bound is right and simply too loose.

The algorithm is therefore not changed. The paper's future-work paragraph now
states the reframing and this obstacle together, and the body clause pointing
at it is removed. What remains open is an approximate, conformally calibrated
pruning test, which would move the fired set and so would have to be measured
as a quality change rather than a speed one -- out of scope before the
deadline.

## Superseded: queued behind every measurement

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

## Where the fallback actually fires: two regimes, and a rank asymmetry

VKSTATS is per rank, and reading one rank understated the streaming rate by
20x (corrected in pre-registration 4). Both regimes, both ranks:

| workload | steps | index builds | fetch p50 | attended frac | fallback TP0 | fallback TP1 |
|---|---|---|---|---|---|---|
| stream 4k->256k | 258050 | 14 | 0 | 0.049 | 0.0027 | 0.0548 |
| RULER 13 tasks @64k | 1850 | 889 | 1023 | 0.108 | 0.3596 | 0.3242 |

Per RULER task over 4k-64k (fb-<task>, TP0): single_1 0.096, multikey_1 0.146,
single_2 0.154, single_3 0.167, multikey_2 0.173, multikey_3 0.245.

Three things separate the regimes, and **matched against each other at the
same context, the first one explains almost none of it.**

- **Context at the moment of decoding.** The stream's first overflow is at
  step 39650, context 43746 tokens (TP1's at 43150, 47246 tokens); below that
  the fired set cannot reach the 4096 capacity and the fence is unreachable by
  construction. Its rate then grows monotonically with context (TP0, per 32k
  window): 0.0000 to 35k, 0.0001 at 66k, 0.0014 at 97k, 0.0022 at 129k, 0.0026
  at 191k, 0.0054 at 222k, 0.0082 at 254k. That is why the stream's early
  steps are exactly zero, and it is the whole of the answer only there.
  **At matched context it is not the explanation**: in the 57k-69k windows the
  stream fires 0.00002, 0.00017 and 0.00014, against RULER's 0.360 at 64k --
  a factor of about 2100 with context held fixed.
- **Calibration amortized against calibration repaid.** The stream fits 14
  indexes in 258050 steps, two per layer. RULER fits 889 in 1850 steps, which
  is exactly one per request per layer, and 1850/130 = 14.2 steps per request:
  the index is calibrated at about the point the answer ends, so most of a
  RULER answer is served by the provisional identity-basis index with z at
  Z_MAX, which is conservative and over-fires.
- **What the query wants.** A needle is placed where tier-1's salience signal
  does not keep it -- that is what makes it a needle -- so the recall tier has
  to reach into the archive on every step. Free-running continuation wants
  recent and globally salient rows, which tier-1 already keeps. fetch p50 1023
  against 0, and attended_frac 0.108 against 0.049 at a quarter of the
  context, are the same statement twice.

So the 2100x at matched context is the other two, and the per-task spread
says both are present: every RULER task pays the same one-build-per-request
structure, yet they range from 0.096 (single_1) to 0.245 (multikey_3) over
4k-64k. The common floor is the calibration regime, the spread on top is the
task. `stats-stream-64kprefill` separates them directly -- 64k prefill,
continuation text, 4096 decode steps, stats on -- because it holds the context
at RULER's and amortizes the calibration the way the long stream does. Near
0.0002 puts the weight on RULER's questions; near 0.3 puts it on the prefilled
context itself.

What this costs is not hypothetical: pre-registration 3's C1 control turns the
fallback off and RULER loses 0.0565 of the 65-cell mean, concentrated on the
multi-key tasks. On those tasks a third of the scans cannot certify, and part
of the quality is the fallback's rather than the criterion's.

**The consequence for the paper.** The speedup is measured in the streaming
regime and the RULER quality in the short-answer retrieval regime, and the two
regimes have fallback rates two orders of magnitude apart. The two numbers do
not hold simultaneously, and the text must not let a reader assume they do.

## Verdict: the cost is arming the fence, and removing it recovers all of it

Production protocol (radix on, one running request), server-side ms/token from
the scheduler's gen-throughput lines, median over a +/-2048-token window:

| context | dense | origin | current default | perf branch | fence stub |
|---|---|---|---|---|---|
| 64k  | 4.360 | 4.094 | 4.101 | 4.106 | 4.088 |
| 128k | 4.783 | 4.181 | 4.296 | 4.325 | 4.155 |
| 256k | 5.576 | 4.346 | 4.771 | 4.879 | 4.328 |
| speedup at 256k | 1.000 | 1.283 | 1.169 | 1.143 | **1.288** |

`prod-stream-vestigekv-256k-fencestub` runs the `vestigekv-fused-fallback`
branch with `SGLANG_DEBUG_VESTIGEKV_FENCE_STUB=1`, which compiles the fence
body out of the CSR gather and changes nothing else. It reaches 1.288x where
origin measures 1.283x, so the entire 12% regression is the fence's *arming*
work inside the pack kernel -- the branch itself, and the 443 commits in it,
cost nothing. This agrees with the stats run, where the fence fires on 0.27%
of scans and recaptures nothing.

What the stub does and does not remove, read off the kernel: `FENCE_BODY=0`
skips only the copy of the fenced lane's row set out of the page table, and the
lane then runs the unfenced kept-plus-fetch copy instead. The prep kernel is
untouched, so `indptr` still gives that lane `seq` rows and stage 1 still
attends `seq` of them. The stub therefore removes the ~250 us copy and keeps
the ~195 us attention, which is exactly the split the tier-decode router is
aiming at, and 4.33 is the right ceiling for it.

One caveat on reading that ceiling as achievable. A stubbed lane attends `seq`
rows whose ids are whatever the previous step left in the buffer -- often zeros
-- so its reads may collapse onto a few cached rows, where a correct
implementation walks a real 256k-row page table. The stub can therefore be
optimistic by however much that locality is worth, and the honest statement is
that it bounds the copy's cost, not that a correct implementation must reach
its total. `td-profile-256k` separates the two by kernel.

The stub is a debug switch, not an implementation: its output is wrong on the
steps that fence. What it establishes is the ceiling, 4.33 ms/token at 256k, for a
correct design that keeps the overflow path but stops paying for it on every
step. That design is the tier-decode router (engine
`vestigekv/tier_decode.py` + `vestigekv/decode_fork.py`): it reads a lane's
rows from the kept table and the fetch buffer, or from the page table when the
lane is fenced, so the CSR that the pack kernel builds -- and the arming work
with it -- has no consumer left.

## The tier-decode router measured: a third of the way, adopted, gap still open

Production protocol, 256k, server-side ms/token (`td-stream-256k`, wall 1422s):

| arm | 64k | 128k | 256k | speedup at 256k |
|---|---|---|---|---|
| dense | 4.360 | 4.783 | 5.576 | 1.000 |
| origin/vestigekv | 4.094 | 4.181 | 4.346 | 1.283 |
| current default | 4.101 | 4.296 | 4.771 | 1.169 |
| **tier-decode** | 4.105 | 4.232 | **4.643** | **1.201** |
| fence stub | 4.088 | 4.155 | 4.328 | 1.288 |

It removes the per-step copy and it is faster than the default, but it
recovers only 0.128 of the 0.443 ms gap, 29%, and stays 0.297 ms behind
origin.

**Adoption criterion, owner's ruling (2026-09-16 23:43): better than the
implementation it replaces is enough to ship, and the gap is worked on
separately.** By that rule it qualifies -- 4.643 against the default's 4.771,
1.201x against 1.169x -- and the earlier reading against a 4.33 gate is
superseded. Shipping still waits on the correctness pair (`td-ruler-64k`
against `td-ruler-64k-off`, same tree, one flag apart), because the router is
supposed to move no row and a difference there is a bug, not a tradeoff.

Two readings, and the arms that separate them.

- **The window widened.** td-stream-256k spreads [4.495-4.805] where the
  default spreads [4.751-4.786]. A cost that switches on and off got bigger,
  not smaller, which is what moving a fenced lane's row reading out of a bulk
  copy and into stage 1's inner loop looks like: one coalesced copy became a
  dependent load per block, the indirect addressing the advisor named.
- **The copy's saving is real but partly eaten.** The pack's fenced branch was
  ~250 us/step and 0.128 ms of it survives to the total, so roughly half is
  paid back inside stage 1.

`td-stream-256k-nodense` runs the same build with
`--disable-vestigekv-recall-overflow-fallback`, which truncates an overflowed
scan exactly as origin does. With the overflow work gone from both sides it
must be **no worse than origin's 4.346**; if it is worse, that difference is
the forked stage 1's own overhead and has to be closed before the fenced
branch is worth tuning. `td-profile-256k` then attributes whatever remains by
kernel.

### The nodense arm answers it: the machinery is free

`td-stream-256k-nodense` measures **4.348** at 256k against origin's **4.346**,
inside both p10-p90 windows ([4.332-4.360] and [4.330-4.363]), and 4.168
against 4.181 at 128k. The forked stage 1, the prep-only pack and the router
together cost nothing. There is no bottleneck to close before tuning the
fenced branch, because there is no gap.

That makes the decomposition exact:

| arm | 256k | what it contains |
|---|---|---|
| origin | 4.346 | base; an overflow truncates and loses rows |
| nodense | 4.348 | base; same truncation, tier-decode machinery on |
| tier-decode | 4.643 | base + attending the full row set inside stage 1 |
| current default | 4.771 | base + copying the CSR + attending the full row set |

- Handling an overflow costs **0.423 ms/step** in the CSR design and **0.295
  ms/step** in the tier design. The router made it 30% cheaper and that is the
  whole of its win.
- The residual 0.295 ms is not an implementation defect. It is the feature:
  attending every row of an overflowing lane, which pre-registration 3's C1
  values at 0.057 of the 65-cell RULER mean. Origin does not pay it because it
  drops those rows.

So closing the remaining gap to origin is not a kernel problem. It is either
firing fewer rows (calibration -- origin fires fewer because it solves z
against the kept maximum rather than the archived row's true score) or making
the fenced lane's attention cheaper.

### Two kinds of "do not run this kernel", and which one is legal

A captured decode graph replays a fixed node list, so a launch cannot be
skipped at replay, and deciding from a GPU-computed value would need a
device-to-host read -- a sync on the token path, and not legal during capture
at all. Anything data-driven therefore **launches every step and exits early**,
which is what the code already does: a lane's fenced flag is read inside the
kernel because "a captured graph cannot skip a launch, so an unfenced lane
exits on one scalar load", and `PACK_GRID_CAP` sizes its grid for the hardware
so that "programs past a lane's tile count exit on one scalar load".

`gather=False` is the other kind: a startup flag read on the host at capture
time, so the node is simply never recorded. No read-back, no sync, and
strictly cheaper than an early return, which still pays a dispatch -- at 256k
the gather's grid is `(L, bs, PACK_GRID_CAP)` = 224 blocks. The router's choice
of tier against CSR is the same kind of decision. So: not launching where the
decision is configuration, early return where it is data.

One consequence that the measurement bears out. The gather is worth ~250 us on
TP1 and 13.5 us on TP0, and the step is gated by the slower rank, so removing
it should show about 250 us. It showed 128. The missing ~120 us is inside
stage 1, which is where the fenced lane's row reading moved.

### Atomics, checked statically before the data

Three `tl.atomic_*` calls exist in the VestigeKV kernels, and only one of them
is new against origin:

| call | file | in origin? | how often it fires |
|---|---|---|---|
| `atomic_add(hist_ptr + bin_, 1)` | sigma_fused.py:133 | yes | tier-1 block close, not the token path |
| `atomic_max(amax_ptr, ...)` | operand_fused.py:74 | yes | index build, 14 times in 258050 steps |
| `atomic_add(ovf_count_ptr + li, 1)` | fused_prologue.py:622 | **no** | once per overflowing (layer, lane), inside `if pid == 0` |

The new one counts overflows for `--vestigekv-rebuild-overflow-fraction`. It
survives the nodense arm, because the compaction raises the flag whatever the
fallback setting says, so it fires at the measured overflow rate: about 0.055
per (step, layer) on TP1, roughly 0.4 single-word adds per step from one
program each. That is not a 0.3 ms/step cost, and the host-side reads of that
counter are both gated -- `_overflow_total` only on a stats dump, and
`_rearm_overflowing_indices` only when the rebuild fraction is above zero,
which the default leaves at 0. So atomics are on record as checked, not as the
explanation; if the nodense arm is short of origin, the profile decides.

The standing prediction, for the record before the data: nodense does strictly
*less* work than origin (no gather launch at all against origin's full pack),
so a shortfall cannot be work volume. It would have to be the forked stage 1's
row read -- two segments with a select per block (kept then fetched) against
origin's one contiguous CSR load.

## Pending
- `td-replay-64k` ran and its six answers are sane, five of six hit, but it is
  **not** a comparison: the only other replay on record used
  `--enable-vestigekv-prefill-calibration`, so the two differ by that flag and
  not by the router. The hit pattern matches, including the same qa_squad miss.
  `td-ruler-64k` is the real correctness test, and it has a control (the
  default arm's 650 answers) plus a tool that compares them question by
  question (`mexp/kimi/ruler_diff.py`).
- `td-stream-256k`: the timed 256k stream with `--enable-vestigekv-tier-decode`
  (wired, tested, registered). The gate is 4.33 ms/token with the overflow path
  intact. `td-needle`, `td-replay-64k` and `td-ruler-64k` are its correctness
  arms; the row sets do not move, so a RULER difference is a bug.
- If it is adopted, four paper places stop being true and must be rewritten
  from the new design, not edited. Body: "The stock MLA-decode kernel is
  untouched" (stage 1 is a fork of it, differing only in where a row id comes
  from); the kernel count, where the two-kernel CSR pack becomes the prep
  kernel alone, so seven fused kernels become six; and the int64-CSR sentence
  near the deployment notes, which describes a width the graph path no longer
  builds (the tiers stay int32 and the fork widens on the load). Appendix: the
  paragraph on the same int64 overflow, which stays true of the eager path and
  has to say which path it is about.
- `stream-vestigekv-256k-origin`: origin's timed 4k->256k stream, the clean A/B
  against `stream-vestigekv-256k` at 64k/128k/256k.
- `profile-vestigekv-origin-256k` and the 256k window of
  `profile-vestigekv-current`: the same kernel diff at 256k.
- If the 256k latency gap exceeds the window spread (~0.1 ms), bisect with the
  same profile job at 256f320 (fence, int32 tables, eager), 28a133d (rank flag,
  kernel tiles) and 15f9ae1 (int64 CSR).
