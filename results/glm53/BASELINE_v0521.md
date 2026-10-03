# GLM-5.3-Flash on sglang v0.5.21 — serving baseline

Taken 2026-10-03 on the rebased engine (`glm53-v0521`, served from
`~/vestigekv-wt/engine-glm521` via `ENGINE=`), to establish what the GLM line
starts from after the rebase. Launch is `mexp/glm53/vestigekv.sh` unchanged:
the split pair, DSA prefill + `vestigekv_dsa` decode, fp8 side pool,
`--vestigekv-recall-capacity 2048`.

## Resolved configuration, read off server_args rather than the launch script

    prefill_attention_backend   dsa
    decode_attention_backend    vestigekv_dsa
    dsa_prefill/decode_backend  triton      (SM120 has no other DSA backend)
    page_size                   64          (DSA's KPool path requires it)
    quantization                modelopt_fp4
    tp_size                     2
    vestigekv_recall_capacity   2048
    vestigekv_side_pool_dtype   fp8

## Memory, which is the binding constraint on this box

    weights         88.26 GB per GPU   (320B total / 18B active, NVFP4, TP=2)
    avail after     5.77 GB
    KV cache        1.34 GB -> 112,704 tokens at CTX=32768, MAMBA_SLOTS=2
    avail after KV  4.23 GB

The checkpoint is 190.4 GiB on disk and each GPU has 95.6 GiB, so nothing here
works without TP sharding; the pool is the residual after weights, which is why
`--mem-fraction-static` and `--max-mamba-cache-size` move `max_total_num_tokens`
so sharply. Size both pools from one boot log before picking a stream length.

## Geometry, from config rather than from memory

45 text layers. 11 carry full attention (`deepseek_sparse_attention`) at
layers 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43; the other 34 are KDA linear
attention. `kv_lora_rank=512`, `qk_nope_head_dim=256`, `v_head_dim=256`,
64 heads, `max_position_embeddings=1048576`, and **`qk_rope_head_dim=0`** --
the NoPE premise VestigeKV needs is declared in the config, not inferred.

VKSTATS reports `layers=11`, which matches that list independently.

## Baseline numbers, 18,054-token needle prompt, STATS on

Both readings of the pair are quoted; the spread is the point.

    steps=500 layers=11 overflow=2343 fetch[p50=0 p90=2048 p99=2048] fallback=0.42600
    steps=500 layers=11 overflow=2262 fetch[p50=0 p90=2048 p99=2048] fallback=0.41127

    build 102.8 ms x22  /  105.3 ms x22

Two things to carry forward:

- **The cap binds at p90.** `p90 = p99 = 2048` is the recall capacity, so
  filtering inside the admitted set cannot reduce cost here either; only
  admission width can. Same structural finding as the Kimi line.
- **Fallback is 41-43% at 18k.** That is the number the GLM line has to move,
  and `p50=0` says the median step fetches nothing at all -- the cost is
  concentrated in the overflowing minority, not spread.

## One observation, n=1, not a result

With STATS off the model answered the needle in 97 completion tokens
(`74-ALPHA-2291`, correct). With STATS on, the identical prompt ran to the
512-token limit inside reasoning and emitted no answer. That is the direction
the STATS hazard already has on MRCR, now visible on GLM, but it is one draw
on a reasoning model with a long think budget. Do not quote it; re-run it
paired if it matters.

## Group-of-4 at DSA's 2048-row parity: the ceiling (2026-10-03)

The index channel is 4:1 pooled at the source (`index_kpool = 4`), so both
tiers work in groups of 4 -- not a choice, see the engine commits. The question
that decides the design is what that grouping costs.

Measured on one 18k prompt, 11 DSA layers x 2 TP ranks, scoring the groups in
the ORACLE'S OWN SPACE (pooled latent row against the latent query, best over
heads). A perfect channel, so this is the **ceiling** for any group selector;
the real indexer channel can only do worse.

    budget            count   mass
    2048 rows  (512 groups)   0.511   0.696
    4096 rows (1024 groups)   0.578   0.741

count is the fraction of the oracle's top-budget positions the selector holds,
mass the fraction of its score (floor-shifted), the two statistics
select_recall_telemetry already reports.

Three things follow.

**Grouping by 4 throws away half the oracle's positions at row parity.** The
unpooled selector in this space recalls 1.000 by construction -- it IS the
oracle -- so the whole 0.489 gap is the pooling. This lands on top of
archive_pool.py's independent ranking measurement (0.509 pooled at 2048
against 0.738 unpooled), from a different method, which is some comfort that
neither is an artefact.

**It keeps the heavy rows.** mass 0.696 against count 0.511 says the positions
it drops are mostly light ones. Missing the 2000th row is not missing the 1st.

**DSA has the same ceiling.** DSA selects 512 groups of 4 and expands to 2048
rows (`expand_pooled_groups_to_topk`), so this is not a respect in which the
design is worse than the baseline it is matched against -- it is a statement
that both are far from the oracle, and that the headroom a supplement could
win is large.

Per layer the spread is wide: layers 3 and 7 recall 0.30 to 0.35 where the rest
run 0.45 to 0.64. Any per-layer budget rule should start there rather than from
the mean.

**Not measured yet:** whether the indexer channel reaches this ceiling. The
dump now carries `index_q`, but it is the latest scoring step's query (a
prefill chunk) while `qcal` are calibration queries from other steps -- so the
two are not from the same step and a comparison across them would score one
selector against another step's oracle. Aligning them is the next change, and
until then the channel's recall is unknown rather than good or bad.

One prompt, one workload.

## Does the indexer channel reach the grouping ceiling? No (2026-10-03)

Matched steps: the dump now carries one indexer query per calibration query, in
the same order, so all three of (oracle, channel score, ceiling score) come
from one step. Before that alignment the only available query was the latest
scoring step's, and comparing it against another step's oracle is a comparison
that cannot fail visibly.

22 snapshots (11 DSA layers x 2 TP ranks), 8 calibration queries each,
budget 2048 rows = 512 groups of 4:

                                  count   mass
    indexer channel (the design)  0.221   0.309
    ceiling (oracle's own space)  0.520   0.704

So the grouping costs about half, and the CHANNEL costs more than half of what
remains: the design reaches 43% of the achievable count and 44% of the
achievable mass. Per layer the channel is worst at layer 7 (0.004 to 0.052) and
layer 11 tp0 (0.099); only layer 3 comes near its own ceiling, and that
ceiling is itself the lowest.

**Two things this does NOT establish, and they matter more than the number.**

First, whether it is worse than DSA. DSA selects on the same channel at the
same ratio and the same budget -- `group_scores` + `select_groups` over pooled
index keys IS DSA's rule -- so 0.221 is most likely a property of DSA's
indexer rather than of this design, and the design is at parity by
construction. The reason to care is the reverse of a deficit: it says the
headroom between the channel and the ceiling is large, and
select_recall_telemetry's own framing is that "a parasitic VestigeKV that only
ADDS rows to DSA's selection cannot lower quality, so the question is whether
it has anywhere to go". It has somewhere to go: 0.22 against 0.52.

Second, the head gate. This scored best-over-heads because the dump's
`index_head_w` came back None -- `precompute_head_gate` is conditional. DSA's
real rule weights the heads by its trained gate and sums. A gate is a
weighting, not a tie-break, so it can move this materially in either
direction, and the channel number should be re-taken with it before anything
is concluded about the channel itself.

**The measurement that decides the design** is therefore not this one. It is
whether tier 1's sigma-selected resident set recalls rows the indexer channel
misses, at a budget-matched union -- which is exactly what
`far_region_stats` computes (its d64/d128/d256/d512 columns are DSA unioned
with tier 1's top-Delta). That is the value proposition, the instrument
exists, and its tier-1 arm is only trustworthy as of today's group fix.

One prompt, one workload, 8 queries per snapshot.

## Tier 1 is complementary to DSA's channel, most where the channel is worst

The measurement the design hinges on: a budget-matched union, far region only,
following far_region_stats' discipline (both selectors force the tail, so a
whole-sequence number measures an agreement never in question; and tier 1 keeps
far more rows than the channel's budget, so the supplement's size must be
stated).

22 snapshots, 8 matched-step queries each, channel budget 2048 rows:

    supplement        count   mass
    channel alone     0.229   0.304
    + tier-1 top-64   0.240   0.348
    + top-128         0.248   0.362
    + top-256         0.260   0.378
    + top-512         0.285   0.406

**The marginal return is in the mass and it is steep at small Delta.** 64 extra
rows -- 3% on top of the channel's 2048 -- buy +0.044 mass. 512 rows buy
+0.102. Count moves far less (+0.011 and +0.056), which says tier 1 is adding
HEAVY rows rather than many rows, and that is the only kind worth paying for.

**It is concentrated exactly where the channel fails.** Layer 7 is the
channel's worst (count 0.058 / mass 0.062 on tp0, 0.007 / 0.043 on tp1) and
64 supplement rows take its mass to 0.293 and 0.341 -- a 4.7x and 8x move.
Layer 11 tp0 goes 0.069 to 0.194 at d64 and 0.368 at d512. Where the channel
is already decent the supplement adds little: layer 15 does not move at all
until d512. So the two selectors are complementary rather than redundant, which
is the property the whole split depends on and was not previously measured on a
correctly-read salience channel.

**Caveats, in order of how much they could move this.**

~~Tier 1's sigma is group-granular and repeated to its positions, so
`sigma.topk(64)` picks 64 positions out of tied groups rather than 16 whole
groups.~~ **Checked and withdrawn.** sigma is group-repeated, so every group's
four values are equal and topk returns ties contiguously: the position-wise
top-64 IS exactly 16 whole groups, verified by set equality and by counting the
distinct groups touched (16 of 16). A group-aligned supplement was computed
separately and matches to three decimals at every Delta, because it is the same
set. The union numbers above are not understated; they are exact.

The channel is scored best-over-heads because the dump's head gate is None;
DSA's real rule weights heads by its trained gate and sums. That moves the
`chan` column, hence every delta.

One prompt, one workload, 8 queries per snapshot. The per-layer spread here is
much wider than the mean, and layers 7 and 11 are carrying the result -- three
more workloads before this is a finding rather than a direction.

## With DSA's real gate, the small supplement's value mostly disappears

The head-gate caveat was material and it cut against the previous section.
`record_index_gate` now captures the RESOLVED gate (after q_scale and the
softmax scale, which is the one DSA's logits use), row-selected together with
the query so the two cannot drift. Same 22 snapshots, far region, 2048-row
channel:

    delta    max-over-heads     DSA gate
    chan     0.228/0.304        0.266/0.395
    +64      0.240/0.348        0.267/0.399
    +512     0.284/0.404        0.307/0.443

Two things, and the second supersedes the previous section's headline.

**The channel is better than reported.** Scoring it with DSA's trained gate
instead of best-over-heads lifts it from 0.304 to 0.395 mass, +0.091. Reducing
over heads by max where the model weights and sums was understating DSA's own
rule by more than the supplement was adding.

**The steep small-Delta return was an artefact of that.** 64 supplement rows
bought +0.044 mass against the max-reduced channel and buy **+0.004** against
the gated one. The gate and the supplement were two explanations for the same
missing mass, and the gate accounts for nearly all of it at small Delta. The
previous section's "+0.044 for 3% of budget" is withdrawn.

What survives is smaller and needs a real budget: 512 supplement rows, 25% on
top of the channel, add +0.041 count and +0.048 mass. That is still a gain and
still in the mass rather than the count, so tier 1 is still contributing heavy
rows -- but "3% buys a third of the gap" is gone, and with it the case that the
supplement is nearly free.

The methodological point is worth more than either number: I reported the
complementarity result WITH the gate caveat attached, and the caveat turned out
to carry most of the effect. A caveat that could flip the headline is not a
footnote, and this one should have been closed before the result was written
down.

Still one prompt, 8 queries per snapshot.

## Second workload: the structure replicates, the magnitudes agree

Every number above came from one synthetic-filler needle prompt. Repeated on
natural prose (40 Paul Graham essays, 15.4k tokens, needle at the midpoint),
far region, 2048-row channel, DSA gate, count/mass:

    delta   synthetic filler   natural prose
    chan    0.266/0.395        0.251/0.320
    +64     0.267/0.399        0.256/0.326
    +512    0.307/0.443        0.310/0.380

    supplement gain (mass)   +0.049            +0.060

Both halves of the corrected finding hold on a workload that shares nothing
with the first: 64 supplement rows are worth essentially nothing (+0.004 and
+0.006), and 512 rows are worth about +0.05 mass (+0.049 and +0.060, agreeing
to 0.011). The count gain also agrees (+0.041 and +0.059).

**The channel is worse on prose** (0.320 against 0.395 mass) **and the
supplement is worth slightly more there** (+0.060 against +0.049). That
direction is consistent with tier 1 being a SPECTRAL statistic -- natural text
develops spectral structure in a NoPE cache where synthetic repetition does
not -- so sigma having more to say on prose is what the premise predicts. Two
points is not evidence for a mechanism; it is a coincidence worth testing
rather than a finding.

Still n=1 per workload (8 queries per snapshot, 22 snapshots), so there is no
noise floor to quote and the agreement could be luck. What has improved is
that the claim no longer rests on a single corpus, and the corpus it was most
likely to be an artefact of -- repeated filler -- is now the one with the
SMALLER supplement gain.

One thing the prose run flagged and did not answer: the needle was not
retrieved there (completion hit the 300-token cap mid-reasoning), where the
synthetic prompt was answered every time. That is a quality observation on a
single draw with a truncated budget, not a retrieval failure, and it should be
re-run with a real budget before anything is read into it.
