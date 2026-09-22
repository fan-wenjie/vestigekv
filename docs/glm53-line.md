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

## The decode ceiling at bs=1, and what the gap can be

Before closing the 0.68 ms, the bound. Per decode step, per TP rank, on
one RTX PRO 6000 Blackwell (512-bit GDDR7 at 28 Gbps, 1.79 TB/s nominal).

**Weights the step must read (active parameters, from the checkpoint's
config).** 42 MoE layers x (8 routed + 1 shared experts x 3 x 4096 x 2048)
= 9.5 B; 3 dense layers x 3 x 4096 x 12288 = 0.45 B; 11 MLA layers x ~124 M
(q_a/q_b, kv_a/kv_b absorbed, o_proj, the indexer's wq_b/wk) = 1.4 B; 34
KDA layers, of the order of 2.5 B; the 4096 x 154880 head, 0.63 B. About
14.5 B active parameters at NVFP4 (0.5625 B/param with the e4m3 block
scales), ~8.2 GB, ~4.1 GB per rank plus the bf16 head's half: **~4.7 GB per
rank per step, 2.6 ms at 1.79 TB/s.**

**Attention traffic is a rounding error next to that.** DSA per MLA layer
at S = 36k: the indexer scores the 4:1-pooled index cache, S/4 x 132 B =
1.2 MB, and attends 2048 x 1024 B = 2.1 MB; 11 layers, 36 MB, 20 us. The
VestigeKV arm as served: kept (rho S ~ 1.1k) + fetched (<= 2048) rows at
1 KB plus, when the gate opens, the archive sketch at S x 132 B = 4.7 MB;
11 layers, 33-88 MB, 18-49 us; and with the indexer running every step,
DSA's 1.2 MB per layer on top. Neither arm's attention is bandwidth-bound
at bs=1.

**So the step is latency-bound.** The measured baseline graph is 11.23 ms
against a 2.6 ms byte floor: 45 layers x roughly 25 graph nodes is ~1100
nodes per step at 2-4 us of fixed cost each (3-4 ms), the FP4 grouped MoE
GEMMs at M=1 run far below the bandwidth roof, and the 90 TP all-reduces
per step wait on the peer. In that regime an attention path costs what its
*nodes* cost, not what its bytes cost: DSA's decode attention is 11 x
(~5 indexer kernels + 2 attention kernels) at ~4 us plus their compute,
about 0.5-0.6 ms of the 11.2 -- five percent of the step.

**The ceiling for this arm, then.** Its attention-side floor is 11 x (2
attention kernels over ~2-3k rows, ~25 us) plus the per-step shared work
(prologue, scan, compact, pack: ~5 launches, ~25 us) plus the salience
write and the index-cache write per layer (~2 us each): roughly 0.35-0.4
ms per step, against DSA's 0.5-0.6. The best this line can show is
therefore **parity to ~2-3% faster than DSA per step**, and only if the
indexer's scoring is not paid on steps that do not overflow. There is no
1.5x here: DSA is already sparse, and the step is 95% not attention. The
Kimi line is where the speedup lives; the GLM line's claims are quality at
parity and the index's memory.

**The gap to close is ~1 ms/step**, from the measured +0.68 to the ceiling's
-0.2 to -0.3: the machinery costs ~3.5x its floor today. What the
node-level trace has to attribute, in the order the arithmetic suggests:
(1) node count -- every kernel VestigeKV adds per layer per step (qbuf copy,
two or three ring writes, the fork's two stages, the pack's two launches,
the prologue/scan/compact family) is 2-4 us of fixed cost whatever its
bytes; the target is <= 3 extra nodes per layer; (2) the fork against
DSA's own decode kernel on the same ~2k rows; (3) the indexer's scoring
and top-k on every step (the lean/top-k graph variant); (4) the gate-open
rate and the fetch_len distribution from VKSTATS, which set the scan's and
the attention's actual byte counts.

## Closing the decode gap: what was found, what was fixed, what remains

Engine `glm53-dsa-decode` at 3f2ab34150, fast-forwarded into the served
`glm53-v0520-box` on 2026-09-22 15:15 UTC. Every GLM vestigekv-arm job from
`dsaopt-smoke32k4k-vestigekv` on runs on this tree; the README block on that
job says which earlier records it retires.

**The numbers.** Decode ITL at bs=1, one request, 2 x RTX PRO 6000, TP=2,
the arm scripts' one-slot config (`mexp/glm53/*.sh`, MEM_FRAC 0.95).

| prompt / decode | DSA baseline | VestigeKV before | VestigeKV after | gap |
|---|---|---|---|---|
| 4k / 2048, no profiler | 11.26 ms | 11.88 ms (+5.5%) | 11.49 ms | +0.23 ms (+2.0%) |
| 32k / 1024, under nsys node trace | 11.51 ms | -- | 11.80 ms | +0.29 ms (+2.5%) |
| 32k / 4096, no profiler (the smoke gates, queue records) | 11.29 ms | 12.01 ms (+6.4%) | 11.54 ms | +0.25 ms (+2.2%) |

The smoke rows are diag-smoke32k4k-baseline, dsafb-smoke32k4k-vestigekv
and dsaopt-smoke32k4k-vestigekv (needle passed on both vestigekv trees).
"Before" is the served tree at 63f2d6f994 (the indexer-at-decode fix, the
first tree whose fenced lanes attend the selection). The 32k pair was only
measured under the profiler (both arms under it, so the comparison holds;
the absolute numbers carry the trace overhead).

**What the node-level trace attributed, and what each fix bought** (all at
4k unless said; per decode step, per rank).

1. *Salience-ring writes*, 0.24-0.30 ms: act_quant + three index_copy_ +
   address arithmetic per MLA layer, seven launches of 2-5 us each. One
   fused Triton launch (`vestigekv/ring_write.py`, DSA's own ue8m0
   quantisation copied line for line, byte-identical ring) -- 15 us/step now.
2. *The decode kernel*, +0.125 ms: upstream's MLA-decode fork (long
   contiguous CSR schedule) ran 20 us per layer against 9 us for DSA's
   split-K sparse decode on the same ~2k rows. The DSA kernel is now forked
   instead (`vestigekv/dsa_decode_fork.py`): the row id comes from the kept
   table, the fetch buffer or the indexer's selection; the lane's count is a
   runtime value; an empty split writes its partials. Bit-identical to
   DSA's kernel on the same rows (test), 6.7 vs 6.9 us per layer.
3. *Two torch.cat and an index_put per layer*, ~0.13 ms: the model handed a
   non-DSA backend the concatenated q and k and the base wrote KV through
   index_put. `vestigekv_dsa` now sits in `FORWARD_ABSORB_CORE_ATTENTION_
   BACKENDS`: q and k arrive as latent + rope parts, the captured step
   writes the row with the pool's two-tensor kernel (DSA's own decode write)
   and the fork reads the parts in place. Eager steps still concatenate.
4. *Graph-pool memory*: the fork borrowed DSA's split-K partial buffers,
   which reserve for a batch of 128 at the first call's split count; with
   the tiers' split count (kept capacity + fetch width over 64 = 96) that
   was 0.4 GB, and the sparse-prefill autotune then ran out of memory on the
   first request. Sized to the call now: capture 0.63 GB, 3.30 GB free
   after capture, the same as the tree before. A second memory finding
   came out of the first release: DSA's Triton sparse prefill kernel
   autotunes 27 configs at its first launch, and the one-warp configs'
   launches each reserve 0.6-0.8 GB of local memory that the driver keeps
   for the life of the context. In the smoke config (0.95, 135k) that
   sweep survived; in the RULER one-slot config (0.955) it ran out of
   memory at the first prefill of every vestigekv-arm job. Engine
   8b83b939a9 runs the sweep at backend init (4 GB free) and prunes the
   one-warp configs (never selected, 7-9 ms against 0.8): 5 late kernel
   loads at 0.8-0.9 GiB free instead of 188 at 0.1-0.35.
5. *The scan family*, 0.09 ms at 4k: the batched scan took 36 us/step at 4k
   and at 32k alike -- the serial 16-block loop of each program over a
   1024-row bucket, 22 programs live at 4k. 256-row buckets
   (`defaults.SCAN_BUCKET`, the compaction order is unchanged so no fetched
   row moves): 26.6 us at 32k. The prologue merge (30 us, one program per
   pair) did not improve at 8 warps in the served graph (35.8 us) although
   a synthetic timing said 4x; it stays at 4.
6. *The lean graph* (a second captured decode graph that files the key and
   skips the indexer's scoring and top-k, chosen when the previous step's
   probe saw no overflow): built, measured, and **left opt-in and off**
   (`SGLANG_ENABLE_VESTIGEKV_LEAN_GRAPH`). VKSTATS at 32k showed the
   overflow tally by layer as [3124, 77, 78, 77, 1032, 103, 77, ...] of 4150
   steps on rank 0 and [1246, 72, 72, 2161, 1631, ...] on rank 1: two or
   three of the eleven MLA layers fire past the fetch width on most steps,
   so a per-step choice fired on 20-23% of steps and bought ~0.08 ms; and a
   lane that overflows on a lean step attends its page table instead of the
   selection, which is not what the quality records attend. Off, every step
   attends exactly what the single-graph tree attends.

**Where the remaining +0.23 ms is.** GPU time that is VestigeKV's and DSA
does not pay: the kept-row sweep of the prologue (20 us at 4k, 57 us at
32k: the certificate scores every kept row against the step's query), the
merge (31), the scan (27-37), the ring write (15-17), the fork's extra
split partials (4), pack and compaction (~10); about 0.12-0.16 ms. The rest
is what the nodes cost to launch and the rank skew the all-reduces absorb.
The ceiling computed above (-0.45 ms, i.e. faster than DSA) required the
indexer's scoring to be skipped on quiet steps; with two layers never quiet
on this model that term is not available, so the reachable floor is
parity plus the sweep and scan -- roughly +0.1 ms -- and the arm is now
within 0.1 ms of it.

**What the tree change means for the records.** The fork, the ring write
and the split path attend the same rows and file the same bytes as the tree
before (tests: bit-identical kernels, byte-identical ring); the attention
output differs from the MLA-decode fork at bf16 rounding, as any kernel
change does, and the runner's tree rule is what decides provenance. The
baseline arm's code path is untouched by these commits, so its records
stand. The RULER vestigekv record of 09:31 predates even the indexer fix
(its server log has no 'DSA sibling drives'); it is retired and re-runs on
this tree with the rest of the vestigekv arm (order in the README block).

## Quality on the optimised tree (engine 8b83b939a9, 2026-09-22)

Every vestigekv-arm quality job re-ran on the released tree after the
autotune fix (README, `dsaopt-smoke32k4k-vestigekv`); the baseline records
are the ones already on file (its path is untouched). Seed 0, one slot.

| RULER mean over 13 tasks, n=10 per cell | 4k | 8k | 16k | 32k | 64k | all |
|---|---|---|---|---|---|---|
| DSA | 0.929 | 0.944 | 0.950 | 0.940 | 0.948 | 0.942 |
| VestigeKV | 0.980 | 0.957 | 0.947 | 0.958 | 0.948 | 0.958 |

All eight NIAH tasks, vt and cwe (but cwe at 64k: 0.94 / 0.97) are 1.00 for
both arms at every length; fwe, qa_squad and qa_hotpot carry the spread,
which at ten samples per cell is the resolution of the test. This replaces
the 09:31 record, which had no indexer at decode.

| | DSA | VestigeKV |
|---|---|---|
| GSM8K-Platinum (n=1209, 64-shot, greedy) | 0.950 | 0.955 |
| MAUVE, 4k x 16 | 0.993 | 0.930 (PASS) |
| MAUVE, 64k x 64 | 0.994 | 1.000 (PASS) |
| ROUGE-L gov_report / qmsum / multi_news | 36.20 / 25.15 / 26.69 | 33.27 / 24.94 / 26.05 |
| ROUGE-L mean | 29.35 | 28.09 |
| 4k -> 128k stream, mean ITL | 11.35 ms | 11.61 ms (+2.3%) |

The summarisation gap (1.3 points of ROUGE-L, gov_report carrying most of
it) is the one place the arm sits below the baseline outside noise; the
32k VKSTATS of the smoke say why it is small and where it comes from: the
fetch width caps at 2048 on 10% of layer-steps and those lanes attend
DSA's own selection; on the other 90% the arm attends the kept set plus
what the certificate fetched (about the same row count as the top-k, a
different row set), which is where the difference lives.

## No crossover against DSA, and what a redesign would have to change

The Kimi line's speedup comes from a dense baseline whose attention bytes
grow linearly with the context while VestigeKV's do not, so the curves
cross. DSA is already sparse -- a fixed top-2048 per layer and an indexer
over a 4:1-pooled index at 33 B per token -- and here the growth is on
VestigeKV's side. The 4k -> 128k stream pair (queue records
stream-*-128k), median ITL per 16k-token window of decode position:

| context (k tokens) | DSA | VestigeKV | gap |
|---|---|---|---|
| 4-20 | 11.28 ms | 11.45 ms | +0.17 (1.5%) |
| 20-36 | 11.30 | 11.50 | +0.20 (1.8%) |
| 36-52 | 11.29 | 11.54 | +0.25 (2.2%) |
| 52-68 | 11.33 | 11.58 | +0.25 (2.2%) |
| 68-84 | 11.36 | 11.63 | +0.27 (2.4%) |
| 84-100 | 11.36 | 11.72 | +0.37 (3.2%) |
| 100-116 | 11.38 | 11.84 | +0.46 (4.0%) |

A linear fit of the gap is 0.09 ms + 0.36 ms per 100k tokens: positive
slope, no crossing, and DSA's own slope is about 0.1 ms per 100k. The
terms that grow are the certificate's kept-row sweep (the kept set is
about rho S: 2018 rows at 4k, 3044 at 32k, every one scored against the
step's query), the archive scan (132 B per token, four times DSA's index),
and the attention over kept + fetched rows, against DSA's fixed 2048. None
of this is kernel tuning; the three kernels concerned are at or near their
roofline for the bytes they move.

**The order set on 2026-09-22 21:30 UTC**: the Kimi line finishes first
(tputB87, figures, paper macros); then the design changes; the GLM line
enters the paper only if a crossover against DSA appears, otherwise it
stays here. What the change has to do, from the numbers: (1) cap the kept
set at a fixed budget instead of rho S, so the sweep and the attention
stop growing; (2) pool or shrink the sketch below DSA's 33 B per token, so
the scan's slope falls under the indexer's; (3) make the fence rare (the
per-layer overflow of layers 0, 3 and 4 is the thing that blocked the lean
graph), so the indexer's scoring and top-k are skipped on most steps and
the intercept drops by the 0.3-0.45 ms measured above. With (1)-(3) the
arm would start below DSA and stay below it; without (3) it can at best
run parallel to DSA about 0.1 ms above. Every quality job re-runs on that
tree.

**Constraint set with it (2026-09-22): the scan period must match DSA's.**
The pooled scan runs on the same cadence and phase as the indexer -- every
step, with the current step's query, at the indexer's position inside
each MLA layer (after the q projection, before attention), over the same
pooled index cache the indexer reads at that step, with DSA's own pool
rule (kpool = 4, the seq mod 4 tail scored unpooled, `compute_dsa_seqlens`
counts). That retires the stale-by-one recall (the batched scan at the top
of the step over last step's qbuf), so a row is fetched by the query that
attends it, as DSA's selection is; the price is one scan launch per layer
instead of one per step, which the fused ring write already showed is
2-4 us each at this batch. The certificate over a pool is q . mean + |q| r
with r the pool's radius, one scalar per pool written when the pool
closes; without r the scan is DSA's ranking with a threshold, not a
certificate, and the docs would have to say so.

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
