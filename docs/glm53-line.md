# The GLM-5.3-Flash line: what is measured, and what is not

Nothing in this file is in the paper, and nothing here goes into the paper until
the GLM line's experiments complete. The paper is the Kimi Linear line. This is
the separate record.

The reason for the separation is not tidiness. The GLM port changes one thing
that reaches into the paper's cost accounting -- where the tier-1 salience
channel lives -- and a half-stated version of that would weaken an argument the
paper currently makes correctly for the model it is about.

## The geometry, and the one thing that differs

| | Kimi Linear | GLM-5.3-Flash-NVFP4 |
|---|---|---|
| `qk_rope_head_dim` | 64 | **0** (rope-less) |
| `side_dim` (scored by attention) | 64 | 0 |
| `sigma_dim` (tier-1 ranks by) | 64, in the latent row | **128**, the DSA indexer key |
| `sigma_in_row` | True | False |
| MLA layers | 7 of 27 (hybrid) | **11 of 45** (hybrid: `full_attn_layers` 3,7,...,43; 34 KDA) |
| pool row `c_kv` | 1152 B | 1024 B |

`geometry.py` keeps `side_dim` and `sigma_dim` distinct for exactly this reason.
On Kimi they are the same 64-dim row tail, which is why one word covered both.
A rope-less MLA has no un-roped branch at all, so the salience channel has to
come from somewhere else: the DSA indexer's key, which is never rotated when
rope is absent.

## Memory: why this does not become a per-token allocation

The paper's answer to the reviewer's arithmetic challenge (the scan reads
~260 B/row, so is the auxiliary storage 260 B/row?) is that the 128 B sidecar is
**read in place** from the row -- traffic, not allocation. That answer is
specific to an in-row geometry and does not transfer to a rope-less one.

What makes it hold here instead is that a salience key is read exactly once,
when its `CLOSE_BLOCK` closes and sigma is computed. After that the sigma is
what matters and the key is dead. So the keys are held in a per-request ring of
the open positions (`CLOSE_BLOCK` + one prefill step), not in a table aligned
with the KV pool:

| salience storage | bytes | % of a 1.66 GiB pool | 1+alpha |
|---|---|---|---|
| pool-aligned table | 219 MiB | 12.89% | 1.28 |
| key ring, max_reqs=1, chunk 512 | 13 MiB | 0.75% | **1.16** |
| key ring, max_reqs=4, chunk 1024 | 35 MiB | 2.09% | 1.18 |
| Kimi Linear, for reference | 0 | 0% | 1.14 |

(An earlier version of this table counted 45 MLA layers; GLM-5.3-Flash has 11,
the other 34 being KDA. The ratios are per token per MLA layer and unchanged;
every absolute byte count above is the corrected 11-layer figure, and the pool
is the 157888-token pool at `MEM_FRAC=0.96` on this box, 157888 x 1024 B x 11.)

The ring is sized by request slots, not by tokens, so it does not grow with
context. The residual 0.02 against Kimi is the shorter row (1024 against
1152 B), not the channel.

**These are analytic, carried over the paper's measured staging remainder
(0.027). The paper's 1.14 is measured. A GLM number cannot enter any table
until it is measured the same way.**

## What is wired, and what was found unwired

`glm53-v0520-box` was assembled as v0.5.20 + the VestigeKV line + three box
compat commits. The VestigeKV line is deliberately model-agnostic -- see
`a5891adb5e`, "The GLM port has its own branch" -- so the assembly took the
backend half of the GLM port (geometry, the side pool, `salience.py`) and not
the model half. Nothing failed: the side pool was allocated and stayed zero,
sigma was constant across rows, and the global top-m degenerated to the first m
row ids. Tier 1 was index order wearing salience's name, in a run that served
tokens and reported a plausible latency.

Redone on the current tree rather than cherry-picked, because the backend has
since been rebased and carries the DCP work:

- the model hook (`deepseek_v2.py` builds `SalienceKey` when the indexer's
  own top-k is off and the backend is `vestigekv_mla`; `glm5_next.py` maps
  `indexer.wk` / `indexer.k_norm` onto it at load)
- the key ring, with a position stamp: a position the ring does not hold scores
  `+inf` and stays kept, so a prefix-cache hit or a reused slot can over-keep
  but never under-recall
- an incremental sigma record (`_advance_sigma`), pinned by a bit-exactness
  test: accumulated over uneven chunk boundaries it must equal the one-shot
  blockwise sigma of the same prefix, so the kept set does not depend on how
  the prefill happened to be chunked

A `logger.info` now says when the salience module is **not** built and why. An
arm that silently does not run reads exactly like an arm that ran and changed
nothing; that is how the all-zero channel survived.

## Performance: the prefill gap is attributed, and it is not VestigeKV's code

Smoke, 32k prefill / 4096 decode, seed 0, both arms on the same server config:

| | mean ITL | TTFT |
|---|---|---|
| baseline (DSA) | 11.303 ms | 7026 ms |
| VestigeKV, dense prefill (single backend) | 11.753 ms | 12609 ms |

**Decode is at parity and that is the expected result**, not a regression: 32k
sits just past the ~24k crossover, and the Kimi line's own ratio at 32k is
1.008. The 3-4% here is within that picture.

**The 5.6 s prefill gap is dense MLA prefill attention against DSA's sparse
prefill attention.** Established by a differential nsys profile of the same
32k prefill on both arms (`cuda_gpu_kern_sum`, both TP ranks summed):

| kernel | VestigeKV | baseline | delta |
|---|---|---|---|
| `_fwd_kernel` (dense MLA prefill) | 9741 ms / 1370 launches | 0 | +9741 |
| `_sparse_mla_fwd_split_dim_kernel` (DSA prefill) | 0 | 5119 ms / 5716 launches | -5119 |
| net GPU time | 16829 ms | 12136 ms | +4692 |

Every other kernel -- GEMMs, all-reduce, MoE, the 47972 `FillFunctor` calls
that looked suspicious in isolation -- is shared to within a few percent. The
two arms simply run different attention implementations at prefill: the
vestigekv arm was launched with DSA off (`index_topk: null`) over a dense
Triton MLA base, so its prefill had no index to be sparse with.

### Two attributions before this one were wrong, and one mechanism

1. **The per-chunk whole-prefix sigma transform.** Refuted before it was
   implemented and implemented anyway: sigma over 32768 rows measures 0.292 ms
   and the whole prefill's sigma is ~17 ms, three orders below 5.8 s. The
   resulting incremental record is kept for its own reason -- the kept set no
   longer depends on chunking -- but it moved TTFT by 72 ms against a 902 ms
   run-to-run noise floor.
2. **`_maybe_prefill_build`.** Read out: paced to doublings, four builds at
   32k, on a side stream.

And the "redundancy factor" mechanism offered for the chunk-count scaling was
a coincidence. Holding the prompt at 32k and moving only `CHUNK`:

| chunk | chunks | TTFT | overhead |
|---|---|---|---|
| 512 | 64 | 12864 ms | 5838 ms |
| 2048 | 16 | 8554 ms | 1527 ms |

The overhead ratio 3.82 matched (n+1)/2 = 3.82 to two decimals, and that is
not a mechanism: chunked prefill's total attention FLOPs are S^2 (n+1)/(2n),
0.508 S^2 at n=64 against 0.531 S^2 at n=16 -- the same work. What changes
with the chunk is the dense kernel's efficiency per launch (7.1 ms each at 512
query rows, tile-starved), not the amount of work. A matching ratio was taken
as proof; it should have been taken as a coincidence to explain.

The noise floor matters when reading any of this: two baseline runs differing
only in client seed gave TTFT 7026 and 7928 ms (12%), while their ITL differed
by 0.04%. TTFT claims below ~1 s are not claims.

Attribution moved to the profiler after the second failed guess, per the
standing rule, and the profiler answered in one differential run.

### The fix: a split pair, DSA at prefill, VestigeKV at decode

The engine already supports `--prefill-attention-backend` and
`--decode-attention-backend`; a differing pair becomes one `HybridAttnBackend`
that routes by forward mode, each side wrapped in its own
`HybridLinearAttnBackend` (the KDA state lives in the runner's mamba pool, so
the two sides share it). What the split needed:

- the registry to admit DSA when it is the prefill side of exactly this pair
- the composite to hand every prefill to the decode side afterwards
  (`observe_prefill_extend`), so VestigeKV still builds its kept table, sigma
  record and prefill-time index off the shared KV pool
- the DSA indexer to file its un-rotated key as the salience channel -- it
  computes `k_norm(wk(x))` anyway, so no second module and no second GEMM
- `vestigekv_backend_of` to unwrap the composite to its decode side, so the
  prefill-time calibration queries keep reaching VestigeKV
- the pool at **page 64**, not page 1. DSA's model override forces 64 on CUDA
  (its KPool path requires `page_size == 64`), and the first split launch died
  on VestigeKV's own page-1 refusal. An earlier "DSA tolerates `--page-size 1`"
  check had passed on readiness and a needle while the resolver had silently
  set 64 -- a check of the wrong thing, recorded here so it is not repeated.
  VestigeKV's tables hold token slots and both the fork and upstream's stage 1
  derive page and offset from a slot, so a paged pool changes only the strides
  the fork is launched with; the router now carries the base's page size and
  the affine fenced capture (contiguous-slot assumption) is refused at page > 1.

Known cost left in place: on decode the model still computes DSA's top-k
(`_dsa_forward_uses_topk` sees no `use_mha` on the composite), which the
VestigeKV decode path then ignores. Wasted, not wrong; to be measured and then
gated.

### Split result: the speed is exactly as predicted, and correctness is not

Smoke, 32k prefill / 4096 decode, seed 0, pool at page 64:

| | mean ITL | p99 | TTFT |
|---|---|---|---|
| baseline (DSA) | 11.303 ms | 11.570 | 7026 ms |
| VestigeKV, dense prefill | 11.753 ms | 12.581 | 12609 ms |
| **VestigeKV, split pair** | **11.235 ms** | **11.508** | **7051 ms** |

Prefill overhead against the baseline went from +5582 ms to **+24 ms**
(noise floor ~900), and decode is 1.006x -- the Kimi line's own 1.008 at 32k.
Both pre-registered speed criteria pass.

**The needle probe missed** (`NEEDLE_MISS`), where every earlier run -- the
DSA baseline and the dense-prefill VestigeKV -- answered `NEEDLE_OK`. The
model's answer was coherent and confident and wrong: it quoted a code "at the
beginning" that is not the planted one. That is not corrupted memory (which
reads as garbage) and not the page size (the pool is slot-major and the
fork's page arithmetic reduces to the page-1 address; the decode speed says
the kernel reads what it should). This blocks the split regardless of speed.

Two mechanisms are on the table and they separate cleanly.

1. **Tier 2 is dead under the split -- established.** The prefill-time
   calibration queries reach VestigeKV through `_vestigekv_prefill_queries`,
   which is called from the **MHA** prefill path only
   (`forward_mha.py:267`). DSA prefill dispatches the MLA-absorbed method, so
   no calibration queries are written, no calibrated index is built, and the
   recall tier has nothing to recall with. The fix is a second call site on
   the absorb path, right after `q_b_proj_forward` produces the un-absorbed
   `[tokens, H, qk_head_dim]` query that the writer already expects.
2. **Tier 1 is real for the first time -- to be tested.** The probe's needle
   sits at the head of a ~10k prompt, inside the first closed block. Until
   today sigma was all zeros, so the global top-m degenerated to the first m
   rows by index, which kept the head of the prompt by accident: the probe
   passed for the wrong reason. With a real salience channel the needle's
   rows can be evicted honestly, and then only recall can bring them back.

The single-backend shape (`DSA_PREFILL=0`: MHA prefill, calibration on, the
same real channel, page 1) isolates (2) from (1). **It passed** -- `NEEDLE_OK`,
the planted code recalled from the first closed block with a real salience
channel in place. So (2) is refuted: tier 1's honest eviction is recoverable
by recall when recall has an index. The split misses for (1) alone: no
calibration queries, no calibrated index, no recall.

Fix: a second call site for `_vestigekv_prefill_queries` on the absorb
prefill path, right after `q_b_proj_forward` produces the un-absorbed
`[tokens, H, qk_head_dim]` query. Rope-less only -- at that point the pe slice
is not yet rotated, and on a rope-less model there is nothing to rotate, so
that q is exactly what the writer expects; a roped model keeps the MHA-path
hook. The needle probe is the gate for the re-run: the split is not accepted
on speed alone.

**With the tap in place the split still misses.** Re-smoke at page 64 with
the calibration tap: ITL 11.251 ms (1.005x), p99 11.581, TTFT 7062 ms
(+36 ms) -- the speed holds -- and `NEEDLE_MISS` again. So (1) was real but
not sufficient. Seams then checked and cleared by reading: both backends take
the runner's `req_to_token_pool` / `token_to_kv_pool` (shared objects); the
decode `q` reaching the backend is the standard bf16 absorbed query under
`use_dsa`; DSA's `forward_extend` writes the KV rows (`save_kv_cache=True`)
before the composite's observe hook runs; `base.forward_metadata` is read
only on decode and graph paths, never by the prefill bookkeeping.

Two suspects remain, and they are being separated the same way:

- **page 64 by itself** -- the single-backend shape at page 64 (one variable
  changed against the run that passed) **passed**: `NEEDLE_OK` with the
  resolver confirmed at `page_size: 64`. Page 64 is cleared by experiment,
  not just by the stride arithmetic; the miss belongs to the split path.
- **hooks that never fire under the split** -- now established, and with a
  single cause. One-time INFO lines on the three hooks
  (`observe_prefill_extend`, `write_prefill_queries`, `write_salience`) were
  added; the next split run printed **none of them**. Reading the builder:
  `_build_resolved_backend` composes `HybridAttnBackend(prefill, decode)` from
  the *unwrapped* full backends and applies the model-level wrapper once,
  outside, so the live object is
  `HybridLinear(full=HybridAttn(prefill=DSA, decode=VestigeKV), linear=KDA)` --
  the reverse of the nesting the first version assumed. Two consequences:
  `vestigekv_backend_of` unwrapped in the wrong fixed order, met the wrong
  layer first and returned None, so both prefill-time taps (calibration
  queries, salience keys) became silent no-ops; and `HybridLinear.forward_extend`
  calls `full_attn_backend.forward_extend` directly, bypassing the composite's
  `forward`, which is where the observe hook had been placed. Fix: unwrap to
  a fixed point in either order, and hook `HybridAttn.forward_extend` too.

  **With that fix the hooks fire** -- the next run logged `salience: first
  keys filed` on both ranks and `observe: first prefill observed` -- and the
  first real prefill then **crashed the server**: `RuntimeError: Triton Error
  [CUDA]: out of memory`, raised inside DSA's own prefill kernel
  `_sparse_mla_fwd_split_dim_kernel` during its first-launch autotune, with
  0.84 GiB (TP0) / 0.41 GiB (TP1) free and the engine's own hint in the log
  ("device-loaded after serving started ... pre-load it during engine init to
  avoid CUDA OOM"). Post-init headroom was 2.81 GB, more than the DSA
  baseline's 2.32 GB; the ~2 GB consumed before the kernel loaded is transient
  prefill working memory -- DSA's at `CHUNK=512` (already the reason 2048 and
  4096 OOM on this box) plus, now that the hooks are live, VestigeKV's prefill
  bookkeeping and the autotune's own scratch. The earlier split runs survived
  only because their hooks were dead. This is a resource fault, not an
  algorithmic one; the lever is `MEM_FRAC` (0.96 -> 0.94 frees ~1.9 GB), with
  a paired re-run at a common fraction owed before any number is compared.

  `MEM_FRAC` is not the lever it looked like. The configurator refused 0.94
  outright: "Loaded weights leave no GPU memory for the KV cache under
  --mem-fraction-static=0.94 ... minimum viable = 0.939". The weights alone are
  0.939 of the card, so the static fraction trades KV pool against working
  memory inside a 6% sliver: 0.96 gives the 157888-token pool and ~4% for
  prefill working memory; 0.95 halves the pool (still above the 36.9k tokens a
  32k/4k smoke needs, not the 131k the 128k line needs) for ~1 GB more
  headroom. The durable fix is to have DSA's prefill kernel autotuned before
  serving -- the startup warmup is an 8-token generation, far below the
  512-token chunk shape the kernel is first launched with -- or a smaller
  `CHUNK`.

  The missing `calibration` line is by design: `prefill_calibration=False`
  (`--enable-vestigekv-prefill-calibration` is opt-in and neither line passes
  it), so `write_prefill_queries` returns before doing anything and the recall
  index is calibrated at the first decode step instead. The absorb-path tap is
  correct and inert until that flag is on.

### Verdict: the split pair passes the needle, hooks live

`MEM_FRAC=0.95` (pool 78464 tokens, 3.74 GB headroom after init), page 64,
DSA prefill, VestigeKV decode, real salience channel, no prefill calibration
(index calibrated at first decode): **`NEEDLE_OK`**, with `salience: first
keys filed` and `observe: first prefill observed` logged on both ranks and
DSA's prefill kernel autotuned at 0.82 GiB free without OOM. The model's
reasoning quotes the planted code. This is the first end-to-end run of the
algorithm in the split shape, and the pass is for the right reasons: sigma is
real, the record grows by blocks, recall has an index. The runner now
pre-warms a chunk-sized prefill after readiness so that autotune never lands
inside a measured request again.

Owed next, in order: the 32k pair at a common `MEM_FRAC` (both arms, same
fraction -- numbers are not compared across a knob), then the RULER retry,
then the 128k pair, which needs the full pool and therefore 0.96 plus the
pre-warm.

### The 32k pair at a common fraction, and a correction

Both arms at `MEM_FRAC=0.95`, page 64, seed 0, pre-warm on, both `NEEDLE_OK`:

| | mean ITL | p99 | TTFT |
|---|---|---|---|
| baseline (DSA) | 11.293 ms | 11.530 | 7070 ms |
| VestigeKV, split pair | 11.788 ms | 12.352 | 7699 ms |

Decode **0.958x**, prefill **+629 ms** (inside the ~900 ms noise floor, but
not nothing). These are the first split-pair numbers measured with every hook
live, and they supersede the 1.005x / 1.006x figures reported earlier for the
split: those runs had dead hooks -- no kept table, no sigma record, no recall
-- and measured a decode side that did no VestigeKV work at all. A 4% decode
cost at 32k against an already-sparse DSA baseline is the honest starting
point; the Kimi line's crossover against dense is ~24k and its 32k ratio 1.008,
and DSA is a stronger baseline than dense.

### RULER retry: CUDA OOM in the MoE at 65k, not in VestigeKV

`rulerv0520b-vestigekv` (RULER config: `MEM_FRAC=0.955`, 4 slots, CTX
73728) died 9 minutes in, at the first 64k cell: `MemoryError: CUDA out of
memory. Tried to allocate 142.00 MiB ... 129.06 MiB is free`, raised in
`modelopt_quant.py` -- the NVFP4 MoE -- during a 634-token prefill chunk. Not
attention, not VestigeKV code. Post-init headroom was 3.15 GB on the split
against 2.73 GB on the baseline (same 100736-token pool), so the persistent
structures are not the delta; what is, is VestigeKV's runtime transient at
65k on the decode side (the recall index build gathers the prefix in fp32 and
runs an SVD; ~150 MB at 65k before workspace) on a box the baseline already
runs with ~2.7 GB to spare. 0.955 is the minimum static fraction that fits a
64k RULER prompt (0.95 gives ~61k tokens), so the fraction cannot give.

DSA's prefill kernel is autotuned per `key=["topk","H","USE_FP8_DOT",
"SEQ_BUCKET"]`, and `SEQ_BUCKET` is binary on the launch's *query rows*
(`seq >= 32768`), which chunked prefill at CHUNK <= 1024 never reaches; the
CHUNK+128 pre-warm therefore already covers the only reachable key. The
"device-loaded after serving started" lines are benign load messages (826 of
them across the baseline's hour), not repeated autotunes. Nothing about the
kernel needs to change; the OOM is headroom alone.

Remedy, for both arms so they stay paired: the RULER client is serial, so
`MAX_REQS`, `MAMBA_SLOTS` and `GRAPH_BS` go 4 -> 1 (three KDA cache slots and
a 4-lane captured scan grid freed), and the baseline RULER is re-run under the
same knobs;
`rulerv0520-baseline`'s 4-slot record is kept under `ruler/superseded/`.

### RULER, one slot, both arms: parity

`rulerv0520s1-baseline` (57 min) and `rulerv0520s1-vestigekv` (60 min), 13
tasks x {4k, 8k, 16k, 32k, 64k}, n=10 per cell, seed 0, both under the one-slot
knobs, `MEM_FRAC=0.955`, CTX 73728, split pair on the vestigekv arm
(`mexp/glm53/compare_ruler.py`, records
`results/glm53/ruler/results_{baseline,vestigekv}_n10_4096-...-65536.json`):

| | 4k | 8k | 16k | 32k | 64k | mean |
|---|---|---|---|---|---|---|
| DSA baseline | 0.93 | 0.94 | 0.95 | 0.94 | 0.95 | 0.942 |
| VestigeKV (split) | 0.97 | 0.94 | 0.94 | 0.97 | 0.93 | 0.950 |

65 cells: VestigeKV below the baseline in 7, above in 8; one sample is 0.10
at n=10, and every difference is within two samples. The eight NIAH tasks
and `ruler_vt` are 1.00 on both arms at every length except
`niah_multivalue`, where the split reads 0.95 at 32k and 0.90 at 64k against
the baseline's 1.00 -- the one cell family where recall of several planted
values from one query looks costlier than the baseline's top-2048; the qa
and fwe tasks move both ways by one or two samples. The 4-slot baseline
record and a stale 65k-only vestigekv partial (2026-09-21) are under
`ruler/superseded/`; the comparison reads only the paired one-slot records.

Records: `superseded/...split_needlemiss` (no tap) and
`superseded/...split_cal_needlemiss` (tap); neither is a result.

The pre-split smoke record is under `superseded/` as `...denseprefill`; the
split smoke is `diag-smoke32k4k-vestigekv`, whose record now carries the
NEEDLE_MISS and must not be read as a result.

## Memory, second pass: what the request-lifetime state costs, measured

`mexp/tools/tier_memory.py --tree <engine>` builds one (layer, request) of
recall state at GLM geometry over a 65536-row closed prefix, points a
pool-mode in-graph pack at it the way the server does, runs one decode-time
close, and reads the allocator (process-wide cuBLAS/cusolver workspaces are
taken first so they are not billed to the request). Per (layer, request):

| | before (9153bb6e4d) | after (exact reductions) |
|---|---|---|
| steady after build + pack sync | 8.87 MiB (141.9 B/row) | 8.88 MiB |
| steady after one close (69632 rows) | 10.01 MiB (150.8 B/row) | 9.49 MiB (142.9 B/row) |
| build peak above steady | +97.7 MiB | +97.4 MiB |
| close + pack sync peak above steady | +27.9 MiB | +28.4 MiB |

The steady state is `_csk_all` fp16 [closed, 64] (8.50 MiB, 128 B/row) plus
`_rho_all` fp32, `_pos_all` (int64 -> int32) and `_arch_idx` int32 at 4 B/row
each; `arch` was a fourth 4 B/row table and is now a selection over
`_pos_all` that the pack copies into its arena. Nothing else survives the
pack sync: side and kept rows are read out of the pool, csk through the
cache. So the per-token tax of the algorithm is 128 B of fp16 sketch +
12 B of index per MLA layer, x 11 layers = 1.5 KiB/token at 65k, against
DSA's own index cache at 132 B/token/layer (fp8 128 + scale 4) -- the same
order, one channel wider.

What the second pass removed besides the 8 B/row: the pack's `fits()` and
`update()` gathered the [nk, 576] bf16 kept rows and the archive ids of every
pair at every epoch to read two lengths (a data movement, not a resident
cost -- it does not show in the steady column); the fused build wrote and
discarded a [closed, 64] bf16 sidecar; the kept table
(`[layers, slots+1, cap]` int32) is sized for the compressed arm,
`cap = min(max_ctx, max(activation, rho*max_ctx + 3*CLOSE_BLOCK) + CLOSE_BLOCK)`
= 20608 rows instead of 135168 on this line (11.6 -> 1.7 MiB at one slot,
29 -> 4.3 at four); a configured FULL-arm flag path keeps max_ctx, and every
host-side writer refuses to serve short. All bit-identical against the
previous tree (`compare_operator_trees.py --this engine-pre-memopt`, with a
new `recall_tier` case covering build, close, refresh and query).

What it did not touch, and what is left: `csk` stays fp16 -- the fire
decision compares scores near a threshold and the 11-bit mantissa is what
was argued for; fp8+ue8m0 (the DSA index-cache packing, same precision as
the baseline's selector position) would take the row to 88 B but is a
precision change, not an exact one, and is excluded for now. The build peak
(+97 MiB above steady at 65k, the same on both trees) is the largest
remaining figure and is transient; it is the next thing to attribute with
the same tool.

## The decode kernel, reverse-engineered: what the split pair actually attended

Read back from the served process's Triton cache (`~/.cache/sglang/triton`)
and the source it was compiled from, 2026-09-22. The forked stage-1 kernel
(`_vk_fwd_grouped_kernel_stage1`, 135 variants for sm_120, 4 warps, 2
stages) does carry two mutually exclusive arms selected per lane at run time
by `fetch_ovf[slot] != 0`: the recall arm reads `kept_buf` then `fetch_buf`,
the fenced arm read `req_to_token[slot, :seq]` -- the whole page table. So an
overflowing lane on this line was served **dense**, not by DSA's top-2048,
and the earlier statement that "the overflow fence falls back to DSA" was
wrong. Dense attention on a DSA-trained model is the worse direction: the
substrate ablation (`ruler-dense-mla-short`, `results_dense_mla_n10_4096-8192.json`)
scored 0.788 against DSA's 0.832 over the same eight cells, with `ruler_fwe`
at 8k reading 0.23 against 0.83.

One level up, the larger defect. Under the split pair the model's indexer
asks the active backend for its metadata; at decode that is VestigeKV, whose
base (the Triton MLA backend) answers None, and the indexer returns at once:
no top-k, no index-cache write, no salience key for any decoded token. On
the prefill side the only salience tap sat in `_forward_cuda_skip_logits`,
taken while `max_kv_len <= index_topk`; from the third 1024-token chunk on
the full path ran and filed nothing. The ring scores an unstamped position
`+inf`, and tier-1 is a global top-m by sigma: with almost every row at
`+inf`, the kept set was an arbitrary m of the unscored rows and the rows
that had real scores -- the first 2048 of each request -- were archived.
Every GLM vestigekv-arm quality record up to this point (needle passes, the
RULER parity above) was therefore recall rescuing an index whose tier-1 was
not working, and is superseded. The note in an earlier draft that "DSA's
top-k is still computed at decode, wasted" was also wrong.

Fix (engine `glm53-dsa-decode`): the decode backend resolves its DSA sibling
through the runner's binding before graph capture (a graph captured without
the sibling's metadata records no indexer launch), drives the sibling's
decode metadata from its own metadata hooks, answers the indexer with it, and
the indexer's full path files the salience key. The fenced arm attends the
indexer's selection with DSA's own row rule, `compute_dsa_seqlens` (whole
index pools clamped to `index_topk`, plus the `seq % index_kpool` tail;
`index_kpool = 4` here) over the compact `-1`-padded `[bs, 2051]` selection,
gathered by row id and masked per entry the way `triton_sparse_mla_decode`
does. `VK_TOPK = 0` compiles to the previous kernel, so the Kimi line is
untouched (pinned by `compare_operator_trees.py`). The vestigekv arm now
pays the indexer every decode step, which is the baseline's own cost; this
line's claim is quality and index memory, not decode speed.

Gate before any vestigekv-arm record is taken again: needle probe, the
32k/4k smoke against the baseline record, the two evidence lines in the
server log (`DSA sibling drives decode metadata`, `first DECODE keys
filed`), then an nsys decode trace of both arms for the bottleneck reading.
Gate result (2026-09-22 12:17, engine `63d20ddbc4`, `dsafb-smoke32k4k-vestigekv`
against `diag-smoke32k4k-baseline`, both `MEM_FRAC=0.95`, page 64, seed 0):
`NEEDLE_OK`; both ranks log `DSA sibling drives decode metadata; fenced
lanes attend the indexer's top-2048 rows` and `salience: first DECODE keys
filed`; no error. ITL **12.01 ms against 11.29** (0.94x), TTFT **7980
against 7070 ms** (+910 ms). The earlier "1.005x, +36 ms" split figure was
measured with the hooks dead and is withdrawn. The decode-side price is the
indexer every step (the baseline's own cost, now paid by both arms) plus the
ring writes; the prefill side pays the full-path salience tap on every
512-token chunk. An nsys trace of both arms (4k prompt, 2048 decoded) put
the steady-state decode graph at a median 11.91 ms (vestigekv) against
11.23 ms (baseline) on the device, +0.68 ms, which is the whole ITL gap; the
node-level breakdown of that 0.67 ms follows from the graph-node trace
scheduled before the Kimi sweep.

## Run constraints

Fixed in `mexp/glm53/common.sh`: CUDA graph on, radix cache off,
`--max-running-requests 4`, `--max-mamba-cache-size 4`,
`--chunked-prefill-size 1024` (2048 and 4096 OOM the DSA prefill: weights are
88 GB/GPU and ~4.5 GB is left for the pool plus prefill working memory),
`--mem-fraction-static 0.955`, `--context-length 73728`, `--random-seed 0`,
`--language-model-only`, `--sampling-backend pytorch`.

`--vestigekv-recall-capacity 2048`, not the 4096 default: this checkpoint's DSA
attends `index_topk = 2048` rows and the overflow fence on this line falls back
to DSA, so a 4096-row recall budget against a 2048-row baseline would be a
difference in budget rather than in method.

Every stream job declares `seed: 0`. They did not until 2026-09-22, and every
GLM stream record before that ran at the benchmark's default 42; the affected
records are under `results/glm53/superseded/`.
