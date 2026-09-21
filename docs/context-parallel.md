# Decode context parallelism for VestigeKV

**Status: design note. None of it has been run.** Every record under `results/`
was produced at `dcp_size: 1`, `attn_cp_size: 1`, and the backend has no code
path that shards a sequence. What follows is derivation from the operators as
they are implemented today, plus the payload sizes those derivations imply. A
number here is a consequence of the code, never a measurement, and the two are
marked apart throughout.

It exists because the memory ceiling the paper reports is a per-rank ceiling —
MLA's latent is one vector per token, so tensor parallelism does not shard it
and both ranks allocate the same 37.70 GB / 5020421 tokens — and context
parallelism is the obvious way out. A reader who works that out will want to
know whether the eviction statistic survives being sharded. It does, and the
reason is worth writing down because the first two answers we reached were
wrong in opposite directions.

## What the engine already provides

sglang's DCP for MLA works by sending the **same query to every rank**: the q
heads are all-gathered across the DCP group, each rank attends its own slice of
the KV with all heads, and the rank-local partials are merged. The gather is
visible at `engine/python/sglang/srt/layers/attention/aiter_backend.py:1230`
and `:1254` (`num_heads = layer.tp_q_head_num  # gathered heads =
num_local_heads * dcp`).

The merge is softmax's own associativity. With `attn_k = (1/Z_k) Σ_{i∈S_k}
exp(s_i) v_i` and `lse_k = log Z_k`,

    attn = (Z0·attn0 + Z1·attn1)/(Z0 + Z1)
         = σ(lse0 − lse1)·attn0 + σ(lse1 − lse0)·attn1
         = lerp(attn0, attn1, σ(lse1 − lse0))
    lse  = logaddexp(lse0, lse1)

`kernels/ops/attention/merge_state.py:56` computes the first form after
subtracting the max; the sigmoid form is algebraically identical and better
conditioned, since it saturates instead of dividing two underflowed
exponentials. Two details the caller owns rather than the kernel:

- the combined `lse` is required for anything beyond a two-way merge, and a
  hierarchical or three-way merge is what a DCP group wider than 2 needs;
- the neutral element for a rank with no rows is `(out = 0, lse = −inf)`
  (`cutedsl_mla_backend.py:267`). The kernel maps `+inf → −inf` on input but
  does **not** guard both sides being empty, which is `nan`. Unreachable under
  any rule here — every query has at least its own row somewhere — but it is a
  premise, not a property, and a future sharding rule has to re-establish it.

Only **7 of the 27 layers** are full-attention MLA
(`linear_attn_config.full_attn_layers = [4, 8, 12, 16, 20, 24, 27]`); the other
20 are KDA linear attention with no KV pool and nothing to merge. So the
per-step merge is paid seven times, not twenty-seven.

## What the backend is aware of today

One place, and it is not a rejection:
`vestigekv_mla_backend.py:468` bails out of the `_out_graph_metadata_lite`
perf path when `dcp_size > 1` and falls back to the full base hook. Nothing
refuses the combination at startup. **This is a defect independent of whether
DCP is ever implemented**: today `--dcp-size 2` would start, run, and produce
rows selected from a sequence each rank has only part of, with no message. A
startup refusal should land regardless of the rest of this note.

## The sharding rule: any partition of the rows works

This is the part that surprised us. Tier 1's σ is a low-pass residual along the
token axis, which reads like an operation that needs its rows contiguous and in
order — and the first conclusion here was that a token-level round-robin
would decimate the sequence, shift the effective cutoff by the shard count and
alias periodic content down into the kept band, making the kept set a function
of the parallel width.

That argument assumed each rank would run the transform on its own subsampled
stream. The operator does not force it. `sigma_fused.py` computes

    Y   = Cᵀ R = Σ_t c_t ⊗ r_t        (pass 1)
    σ_u = ‖ r_u − c_uᵀ Y ‖            (pass 2)

where `C` is the fixed `[T, 32]` real Fourier basis over the close window,
indexed by a row's **position inside the block** and built once at startup
(`_basis`). Pass 1 is a plain sum over rows, and a sum decomposes over *any*
partition of the index set: each rank multiplies its own rows by those rows'
own `c_t` — which it knows, because it knows their positions — sums locally,
and the shards are combined by one all-reduce of `Y`. Pass 2 is row-local once
`Y` is in hand.

So token-level, page-level and block-level round-robin all produce the **same**
`Y` and therefore the same σ. The only difference is summation order; the
kernel accumulates fp32-ieee, and the docstring already bounds reordering at
~1e-7 rms against a 1.2e-3 rms bf16 input quantization, three orders below.
Top-m selection is insensitive except at exact ties, where any tie-break is
licensed (Property 1).

Payloads, per closed block and per MLA layer:

| object | shape | bytes |
|---|---|---|
| `Y` | `[32, 64]` fp32 | 8192 |
| σ histogram | `[1024]` int32 | 4096 |

7 layers × 12 KB = 84 KB, exchanged **once per 4096 decode steps** — a block
closes every `CLOSE_BLOCK` tokens. Against the per-step merge traffic this
rounds to zero.

### Global top-m stays exact

Tier 1 selects a single global top-m over every block's σ, via the fused radix
histogram. Sharded: all-reduce the histogram (it is a sum, and block histograms
already sum today), so every rank derives the identical threshold bin `b*`.
Rows strictly above `b*` are kept locally with no further exchange. The deficit
`m − #{above b*}` is drawn from the boundary bin; exchanging each rank's count
in `b*` and assigning the deficit in rank order gives exactly m rows,
deterministically. It is exact rather than approximate because within one bin
the σ values share their fp32 top 11 bits, which is the tie regime Property 1
already covers.

## Tier 2 is where the per-step cost actually is

Tier 1's reduction is per block. Tier 2's are per step, and there are two:

1. **Firing.** A row fires where its certified score beats the tier-1 max for
   its head (`recall_tier.py` module docstring). Sharded, each rank holds part
   of the kept set, so the max is not local: an all-reduce max over `[heads]`
   has to precede firing.
2. **Fetch.** The tier then takes the top-j fired archived rows. The global
   top-j of a union is contained in the union of the per-rank local top-j, so
   one exchange of the rank-local j candidates yields it **exactly** — no
   approximation and no per-rank quota. Overflow (more fired than the fetch
   buffer holds, which falls back to exact attention) is a global predicate and
   needs the fired counts summed.

Both are small — `[heads]` floats and j scores with their ids — but they are
paid on every decode step of every MLA layer, unlike tier 1's per-block 12 KB.
If DCP is measured and comes out badly, this is the first place to look.

## Choosing the granularity

Any partition is correct, so the choice is made on balance and locality, not on
σ:

- **Row balance.** Token-level round-robin is exact (n/TP); block-level is
  within one 4096-row block, which is 3% at 128k.
- **Kept-set balance**, which is the one that costs memory and time. Anomalous
  rows cluster in time — a RULER needle sits at one position, and the standing
  component this tree measured at bin 71 is periodic (`results/kimi/SIDECAR_SPECTRUM.md`).
  Block-level sharding can therefore drop a whole cluster of kept rows on one
  rank; token-level round-robin spreads any temporal cluster by construction.
  This is the argument that decides it.
- **Pages.** `page_size` is 1 in every run here, so the pool is already
  token-indexed and a token-level rule splits nothing. Under a larger page a
  page-level round-robin is the equivalent rule, and the σ derivation above
  covers it unchanged.
- **The open block** can be replicated on every rank — 4096 rows at the
  measured 7509 B/token is 30 MB against a 37.70 GB pool — so the most recent
  tokens need no merge at all. That choice is orthogonal to how closed blocks
  are sharded and can be made separately.

## What this note deliberately does not conclude

**That DCP would make VestigeKV faster.** The measured decode slopes are 1.9 ns
per step per cached token for VestigeKV against 6.3 ns for dense
(`\serveSlopeVk`, `\serveSlopeDense`). Halving the rows each rank attends
removes six-point-three nanoseconds of work per token from the dense arm and
one-point-nine from ours, so the honest expectation is that context parallelism
**narrows** the gap the paper reports while relieving the memory ceiling for
both. Any measurement here should be registered in `README.md` as a paired
comparison at matched DCP width, not as a VestigeKV-only speedup.

**That the payload numbers transfer off this testbed.** These runs set
`NCCL_P2P_DISABLE=1`, so cross-rank traffic goes through host memory. At batch
1 the merge is latency-bound (seven layers, one round trip each); at large
batch it is bandwidth-bound on `[tokens × 32 heads × 512]`, about 1 MB per
layer at bs=32. The two regimes can rank differently and one batch size does
not settle it.

**That NoPE is what makes this work.** It is not. The decomposition above uses
only that pass 1 is a sum and that `C` indexes position within the block; rows
are stored already-rotated under RoPE, so the same derivation would hold there.
NoPE is what makes σ *meaningful* — under a rotation, equal content at distinct
positions is not equal as rows, which is the premise
`test_vestigekv_nope_premise.py` guards — not what makes it *shardable*. The
distinction matters in the paper: a sentence resting shardability on position
independence sends a reviewer looking for a dependency that is not there.
