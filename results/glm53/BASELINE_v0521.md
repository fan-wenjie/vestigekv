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

## Four workloads, and LongBench v2 is the most discriminating of them

far region, 2048-row channel, DSA gate, count/mass:

    workload                chan          +64           +512        gain(mass)
    synthetic needle 18k    0.266/0.395   0.267/0.399   0.307/0.443   +0.049
    natural prose 15k       0.251/0.320   0.256/0.326   0.310/0.380   +0.060
    LongBench-v2 QA 13k     0.121/0.151   0.128/0.160   0.203/0.250   +0.099
    multi-needle x4 16k     0.246/0.360   0.250/0.372   0.279/0.410   +0.050

The corrected finding holds on all four: 64 supplement rows are worth nothing
(+0.004 to +0.012), 512 rows are worth +0.049 to +0.099.

**LongBench v2 cannot distinguish SCORES and is the best discriminator of
RECALL.** Its score is blind here in the strongest sense -- 600 completion
tokens and no answer emitted -- and the project already has it putting both
VestigeKV arms ABOVE dense with no mechanism. But its channel mass is 0.151
against 0.320 to 0.395 elsewhere, less than half, and its supplement gain is
double. So DSA's indexer is weakest on a real document with a real question,
and that is exactly where tier 1 is worth most. The benchmark's text is
informative even though its score is not, and those are separate properties
that this line has been conflating.

## Per-layer supplement allocation: a null, and not a stable one

The per-layer channel strength spans 0.06 to 0.43, so a uniform Delta plainly
spends rows where they buy nothing. Greedy water-filling of the SAME total
across the 11 DSA layers, evaluated on the data it was fitted to:

    workload            uniform   allocated    delta
    LongBench-v2 QA     0.2498    0.2407      -0.0091
    synthetic needle    0.4434    0.4510      +0.0076
    natural prose       0.3795    0.3860      +0.0065

Plus or minus 0.008 against a supplement worth +0.049 to +0.099 -- that is
noise on the thing being optimised. The negative on LongBench is a real
artefact of greedy allocation over a non-concave mass(Delta) curve, which is
itself informative: the returns are not uniformly diminishing, so a greedy rule
is not even locally safe.

Worse for the idea, the allocation is workload-dependent. The heaviest layers
are {3, 31, 15} on LongBench, {3, 43, 39} on synthetic, {3, 15, 31} on prose,
and the starved sets differ too -- so there is no fixed per-layer budget to
ship. This is the same shape as the recorded finding that rank-64 capture is
uniform across layers while fire rate spans 0 to 96% with Spearman +0.04: this
line keeps finding that per-layer behaviour is a property of the data, not of
the layer.

One consistent signal survives: layer 3 takes the maximum supplement on all
three workloads. One layer out of eleven, and giving it more did not move the
total, so it is a lead rather than a lever.

**Direction closed.** A uniform per-layer Delta is what the design should use.

## Quality on the fixed tree: the served arm is 46 points below DSA

`niah_multikey_3` @ 65536, n=50, seed 0, both arms on the index-k-fixed tree:

    DSA baseline   50/50 = 1.000   and   50/50 = 1.000
    vestigekv      27/50 = 0.540   and   33/50 = 0.660

                   delta -0.460 (p = 8.7e-9)  and  -0.340 (p = 3.0e-6)

Both runs of each arm, never a mean: the vestigekv arm's two draws differ by
12 POINTS (0.540 against 0.660), which is its own replicate spread and is
consistent with the recorded 48% greedy agreement of this decode path. So the
gap is -0.34 to -0.46 rather than the -0.46 a single draw suggested. DSA is
1.000 twice.

check_builds clean (0 build failures, 0 tracebacks); the arm resolved as
vestigekv_dsa decode / dsa prefill, recall_capacity 2048, index_rank 64, STATS
off, so this is the shipped configuration and not a contaminated run.

**This refutes "GLM is the same quality as DSA".** That conclusion came from
RULER's four non-saturated tasks averaging +0.006 at sign p=1.000 -- an
aggregate over tasks that mostly saturate, which is the identical mistake the
Kimi line's group mean made in Q7. On the task that discriminates the arm loses
by 46 points.

It also re-frames the cost question. There was never a speed case worth making
(-2% median ITL, ~~the one win being p90 on long generation~~ -- that win was
withdrawn the same day; see the same-tree cost table below), and now there is a
46-point quality deficit on the same tree. The served GLM arm as it stands is
dominated.

**And it is the strongest argument for the indexer-only direction so far.** DSA
reaches 1.000 on this task with the selection VestigeKV is declining to use. An
arm that serves DSA's own selection and only ADDS rows cannot do worse than
DSA by construction, so the ceiling for the parasitic design here is 1.000 --
against 0.540 for what is served today. The recall measurements said the
channel is strong and our selection is weak; this says the same thing in the
units that matter.


## Same-tree decode cost, both shapes, and the withdrawal of the one win

Both arms on the v0.5.21 tree (`engine-glm521`), GRAPH_BS = MAX_REQS = 1, seed
0, decode-only stream jobs, STATS off. Median AND p99 quoted, per the cost rule.

| metric | 32k in / 4k out | | | 4k in / 124k out | | |
|---|---|---|---|---|---|---|
| | DSA | vestigekv | ratio | DSA | vestigekv | ratio |
| median ITL | 10.832 ms | 11.055 ms | 1.021x | 10.874 ms | 11.131 ms | 1.024x |
| p90 ITL | 10.920 ms | 11.162 ms | 1.022x | 11.030 ms | 11.338 ms | 1.028x |
| p99 ITL | 11.095 ms | 11.688 ms | 1.053x | 11.496 ms | 11.818 ms | 1.028x |
| std ITL | 0.504 ms | 0.831 ms | 1.65x | 0.283 ms | 0.401 ms | 1.42x |
| output tput | 80.626 tok/s | 78.533 tok/s | 0.974x | 91.854 tok/s | 89.682 tok/s | 0.976x |
| median TTFT | 6414.7 ms | 6689.4 ms | 1.043x | 814.2 ms | 848.1 ms | 1.042x |

Jobs: `c521z-{32k4k,4k124k}-{baseline,vestigekv}`, registered in README.md.

**The p90 ITL win at 4k/124k is withdrawn.** It was recorded as -17% against a
v0.5.20 baseline whose own numbers were std ITL 4.883 ms and p99 24.556 ms
against an 11.293 ms median: that baseline was stalling intermittently, and the
"win" was a comparison against a broken run, not against a slow one. Measured
same-tree, the same shape is +2.8%. The v0.5.21 baseline at this shape has std
ITL 0.283 ms, so the instability was the old tree's, not the shape's.

No shape wins. Every ITL percentile at both shapes puts the served vestigekv
arm 2-5% behind, throughput 2.5% behind, TTFT 4% behind -- and the ratio is
UNCHANGED by the index-k repair (v0.5.20 gave 1.019x median / 1.064x p99 at
32k/4k), which is itself informative: the repair moved what tier 1 selects, not
what it costs. Together with the 46-point quality deficit above, the served GLM
arm is dominated on both axes, and no ratio measured against the v0.5.20 tree
can be carried forward.

### The indexer-only arm is wired (default off)

`SGLANG_VESTIGEKV_INDEXER_ONLY=1` serves tier 1 alone: no tier-2 sketch built,
no per-step recall, no fetched row. Five gates, and the seam is
`_collect_calibration`, NOT the build -- gating the build leaves `st["tier"]`
None forever, so the `tier is None` branch fires every step, appending a cloned
query per layer per step without bound. Behavioural smoke (`c521io-smoke`,
STATS on, 250 steps x 11 layers) is the acceptance evidence:

    build=0.0ms x0   cap=0.0ms x0   caps[key=0 fits=0]
    fetch[p50=0 p90=0 p99=0]   scan=0.00ms/step   replay=0%
    VKCAL 0 lines   VKBUILDMS 0 lines   no traceback

It separates what selection costs from what the tier-2 chain costs. On a
geometry with no in-row sidecar it is also the whole method if tier 2 does not
pay for itself.

## The branch rule on GLM, offline: the threshold must come off a curve (2026-10-03)

Rule under test (memory `glm-branch-only`): tier 2 scores every pooled index-k
group with DSA's own index score `sum_h gate_h relu(q_h . K_g)`, fires archive
groups above a threshold set by the KEPT groups' scores, cap 512 groups (= 2048
rows = index_topk). No sketch, no build, no z. `mexp/glm53/branch_recall.py`
and `branch_curve.py` over the three calibration dumps (cd_g2 18k, cd_w2 15k,
cd_w3 13-17k; 704 queries), oracle = latent best-over-heads top-2048, mass =
floor-shifted score fraction, the metric group_oracle.py uses. Bytes = 33 B/tok
index read + 1 KB per attended row, DSA = index read + 2048 rows.

**The literal rule (fire above the best kept group) is dead**: 1-6 groups per
step, mass +0.01-0.02 over kept alone, against DSA's 0.30-0.38. The best kept
group under the indexer is the recent window 41-100% of the time
(`argmax-kept-in-recent`), and tier 1's own picks score nearly as high
(`nr-max` row). So the threshold is a quantile of the kept scores, a knob:

| thr (kept-score quantile) | cd_g2 mass / fired / bytes | cd_w2 | cd_w3 |
|---|---|---|---|
| kept only | 0.168 / 0 / 1.03 | 0.239 / 0 / 1.55 | 0.210 / 0 / 0.58 |
| q1.0 (the rule as stated) | 0.186 / 2 / 1.03 | 0.249 / 1 / 1.55 | 0.228 / 6 / 0.58 |
| q0.95 | 0.303 / 63 / 1.13 | 0.281 / 28 / 1.60 | 0.269 / 74 / 0.69 |
| **q0.9** | **0.343 / 149 / 1.26** | **0.321 / 78 / 1.67** | **0.302 / 133 / 0.78** |
| q0.75 | 0.453 / 457 / 1.73 | 0.465 / 346 / 2.10 | 0.384 / 364 / 1.15 |
| budget (f=1, top-512 of archive) | 0.471 / 512 / 1.81 | 0.538 / 512 / 2.36 | 0.422 / 512 / 1.38 |
| DSA top-512 groups (shipped) | 0.375 / 512 / 1.00 | 0.326 / 512 / 1.00 | 0.295 / 512 / 1.00 |

Readings:

- **q0.9 matches DSA's own mass with 4-7x fewer fetched rows** (78-149 groups
  = 310-600 rows against DSA's 2048). The fetch side of the design works.
- **The kept set is the byte problem, not the fetch.** At 15-18k context the
  dump's kept table is already 1-3k rows -- as many as DSA attends in total --
  so bytes/DSA at q0.9 is 0.78-1.67 and is set by rho·S, not by the rule. It
  grows with S; at 128k kept alone is 2x DSA's whole attention set.
- **q0.75 / budget beat DSA by +0.06 to +0.21 mass at 1.15-2.4x bytes**: the
  kept set is worth real recall DSA does not have, if bytes are not the metric.
- Three workloads order the same way at every row; no workload flips a reading.

What this fixes in the design: the threshold is a per-layer quantile of the
kept groups' index scores (a kthvalue over a fixed-size gather, in-graph), not
the max; it stays on the indexer's own scale, so there is still no
calibration -- but it is a knob, like the margin was, and must be reported as
one. And the bytes-vs-DSA claim needs the kept set bounded (rho, or
sinks+recent plus tier 1 capped at ~1k rows); with recent-only kept the
threshold has nothing to stand on (fires 0-0.1 groups, mass 0.07-0.12).

### Kept-set sweep: with tier 1 preserved, the bytes land at 0.65-0.95x DSA (2026-10-03)

The kept set is the method (user: tier 1's spectral selection is non-negotiable),
so the question is what the branch costs WITH it. `mexp/glm53/branch_kept.py`
rebuilds kept from the dump's sigma record as sinks + top-K sigma picks + the
last W rows (W is what a smaller CLOSE_BLOCK buys; K=512 is rho=1/32 at this
context, K=1024 is rho=1/16) and sweeps the fetch threshold. Mass / fired
groups / bytes-vs-DSA, DSA's own mass in the header:

| W | K | q | cd_g2 (DSA 0.375) | cd_w2 (DSA 0.326) | cd_w3 (DSA 0.295) |
|---|---|---|---|---|---|
| 256 | 512 | 0.9 | 0.248 / 29 / 0.56 | 0.186 / 28 / 0.54 | 0.255 / 48 / 0.57 |
| 256 | 512 | 0.75 | 0.367 / 297 / 0.97 | 0.221 / 94 / 0.64 | 0.287 / 170 / 0.76 |
| 256 | 1024 | 0.9 | 0.335 / 50 / 0.79 | 0.252 / 43 / 0.77 | 0.312 / 78 / 0.82 |
| 256 | 1024 | 0.75 | 0.453 / 387 / 1.30 | 0.314 / 161 / 0.95 | 0.355 / 217 / 1.04 |
| 512 | 1024 | 0.9 | 0.348 / 59 / 0.90 | 0.259 / 15 / 0.82 | 0.355 / 71 / 0.91 |
| 1024 | 512 | 0.75 | 0.416 / 368 / 1.37 | 0.281 / 73 / 0.91 | 0.384 / 222 / 1.15 |

Readings:

- **At DSA-equal mass the branch reads 0.65-0.95x DSA's bytes** (W=256, K=512,
  q0.75: -0.01 / -0.10 / -0.01 mass at 0.97 / 0.64 / 0.76 bytes). The tail W is
  the lever that got it there: the dump's own kept carried a 1.7k-row tail.
- cd_w2 is the hard workload at every cell: its kept-only mass is the highest
  of the three (0.239) yet the fetch recovers least; K=1024 at q0.75 is what
  reaches DSA there (0.314 vs 0.326) at 0.95x bytes.
- Shrinking the tail costs mass the fetch has to buy back (W=256 vs the dump's
  1.7k tail: q0.9 mass 0.248 vs 0.343 on cd_g2), so W and q move together; W
  below 256 was not swept because RECENT_WINDOW is 256.
- Offline, 13-18k context, bytes by arithmetic (33 B/token index + 1 KB/row);
  the served arm must carry a VKSTATS byte counter before any of these numbers
  is quoted as a measurement.

### The lean floor: without DSA's own scoring chain, tier 1 is BELOW DSA (2026-10-03)

`c521io-32k4k-indexeronly-lean` (tier 1 only + `SGLANG_ENABLE_VESTIGEKV_LEAN_GRAPH=1`;
with no tier 2 there is no overflow, so the lean graph is taken every step --
smoke `variant[lean=249 topk=0]`, check_builds clean, no traceback):

| 32k/4k | DSA | indexer-only-lean | indexer-only | vestigekv (sketch) |
|---|---|---|---|---|
| median ITL | 10.832 | **10.656 (0.984x)** | 10.946 (1.011x) | 11.055 (1.021x) |
| p99 ITL | 11.095 | **10.920 (0.984x)** | 11.240 (1.013x) | 11.688 (1.053x) |
| std ITL | 0.504 | 0.379 (0.75x) | 0.437 (0.87x) | 0.831 (1.65x) |
| output tput | 80.626 | **81.592 (1.012x)** | 79.625 (0.988x) | 78.533 (0.974x) |
| median TTFT | 6414.7 | 6521.7 (1.017x) | 6566.4 (1.024x) | 6689.4 (1.043x) |

So DSA's indexer scoring + top-k costs **0.29 ms/step** on this tree
(10.946 - 10.656), more than the whole of tier 1's bookkeeping (0.11 ms); the
only arm on GLM that has ever been faster than DSA is the one that does not run
DSA's selection. This is a cost CONTROL, not a method (tier 1 alone is not
usable). Its bearing on the branch rule: the rule needs the indexer's logits
every step, so it pays the same 0.29 ms DSA pays and lands near the
indexer-only row (~1.01x) plus its own select; the floor it cannot reach is
this one.

The 4k/124k shape replicates the lean floor (`c521io-4k124k-indexeronly-lean`,
check_builds clean, no traceback): median 10.670 vs DSA 10.874 (**0.981x**),
p90 0.980x, p99 0.985x, tput 93.616 vs 91.854 (**1.019x**), TTFT 1.010x; std
0.365 vs 0.283 (1.29x -- the one axis it loses, as at 32k/4k it won it, so std
is not a stable reading at this shape). Indexer-only without lean is 1.009x /
1.029x p99 here, so DSA's scoring chain is 0.31 ms/step at this shape (0.29 at
32k/4k). Both shapes, both readings, same order.

## The branch rule is live (2026-10-03): `SGLANG_VESTIGEKV_BRANCH_Q=0.75`

Tier 2 = grouped recall on DSA's own pooled index logits above the q0.75
quantile of the kept groups' scores, cap 512 groups, written into the existing
fetch buffers (stale by one, like the scan it replaces); no sketch, no build,
no calibration; lean graph never taken. Smoke `c521br-smoke` (32k/256, STATS):

    build=0.0ms x0  cap x0  overflow=0  fetch[p50=420 p90=2048 p99=2048]
    fetched=691/call  kept=1120/call  seq=31750  attended_frac=0.0570
    VKCAL 0  VKBUILDMS 0  no traceback

So per layer-step at 32k the arm attends 1811 rows (1120 kept + 691 fetched)
against DSA's 2048, and the cap binds at p90 as every fetch here has. Bytes by
the live counters: 33 B x 31750 + 1811 KB = 2.86 MB vs DSA 3.15 MB = **0.91x**
-- consistent with the offline W=256/K=512/q0.75 cell (0.64-0.97x). Cost pair
(`c521br-{32k4k,4k124k}`) and `c521br-mk3` (multikey_3 n=50 vs DSA 1.000) are
queued behind it; nothing is a claim until check_builds and those land.

### Branch v1 cost, both shapes: the selection kernel, not the rows (2026-10-03)

`c521br-{32k4k,4k124k}` (v1: truncate to the best 512 groups with a torch
topk over ~34k pooled groups per layer-step, plus a 2048-row req_to_token
gather), check_builds clean, no traceback:

| shape | median ITL (vs DSA) | p99 ITL | output tput |
|---|---|---|---|
| 32k4k | 13.131 (1.212x) | 13.373 (1.205x) | 67.840 (0.841x) |
| 4k124k | 13.193 (1.213x) | 14.481 (1.260x) | 75.730 (0.824x) |

The rows attended were 0.88x DSA's (smoke: 1811 vs 2048), so this is the
selection, not the attention: ~+2.2 ms/step over indexer-only. v1 is retired;
v3 (threshold + count, overflow -> fence to DSA's own top-k, no top-k of ours)
replaces it and these two records move to superseded/ when v3's land.

### Quality under the branch rule: 50/50 on multikey_3, first run (2026-10-03)

`c521br-mk3` (niah_multikey_3 @ 65536, n=50, seed 0, branch rule q0.75 with
v1's truncation semantics; the rule's fired set is the same, the overflow
policy differs from v3+): **1.000** -- against DSA's 1.000 and 1.000 and the
sketch arm's 0.540 and 0.660 on the same task and tree. check_builds clean, no
traceback. ONE run: the second (`c521br3-mk3`, fence semantics on the fused
operators) is queued, and nothing is a pair until it lands.

### v5 (fused operators) in the server: the smoke, and the second multikey draw (2026-10-03)

`c521br3-smoke` (32k/256, STATS on, `SGLANG_VESTIGEKV_BRANCH_Q=0.75`, fence at
the budget), 250 steps x 11 layers:

    build=0.0ms x0  overflow=0  ovf_by_layer=[0 x 11]
    fetch[p50=344 p90=1216 p99=1896]  fetched=470/call  kept=1120/call
    seq=31750  attended_frac=0.0501  no traceback  VKCAL 0

No step fenced: the largest fire was 474 groups, under the 512 budget. v1 had
hit its cap at p90 (fetch p90 = 2048) on the same shape: v1's quantile was
over kept ROWS (the recent window's 4-rows-per-group weight pulled the
threshold down), v5's is over kept GROUPS, which is what the offline curves
used. Attended per layer-step: 1590 rows (1120 kept + 470 fetched) against
DSA's 2048 -> bytes 33 B x 31750 + 1590 KB = 2.64 MB vs DSA 3.15 MB = **0.84x**,
by the live counters. The smoke's ITL (12.6 ms) is a STATS-on number and is
not quoted; the cost pair follows.

`c521br2-mk3` (multikey_3 @65536 n=50, v5): **1.000**. With `c521br-mk3`
(1.000, v1 semantics) that is two draws of the branch rule at DSA's level
(1.000 / 1.000) against the sketch arm's 0.540 / 0.660 -- the first GLM
result that survives the pair rule. `c521br3-mk3` is a third draw.

### Third multikey draw and the v5 32k/4k cost (2026-10-03)

`c521br3-mk3` (multikey_3 @65536 n=50, v5 fence semantics): **1.000**. Three
draws of the branch rule, 1.000 / 1.000 / 1.000, against DSA's 1.000 / 1.000
and the sketch arm's 0.540 / 0.660. check_builds clean, no traceback.

`c521br3-32k4k` (v5, STATS off, decode-only stream, seed 0), same tree:

| 32k/4k | median ITL | p90 | p99 | std | output tput | TTFT |
|---|---|---|---|---|---|---|
| DSA | 10.832 | 10.920 | 11.095 | 0.504 | 80.626 | 1.000x |
| **branch v5** | **11.408 (1.053x)** | 11.543 (1.057x) | **11.711 (1.056x)** | 0.374 | **76.812 (0.953x)** | 1.021x |
| branch v1 | 13.131 (1.212x) | 13.231 (1.212x) | 13.373 (1.205x) | 0.436 | 67.840 (0.841x) | 1.022x |
| indexer-only | 10.946 (1.011x) | 11.036 (1.011x) | 11.240 (1.013x) | 0.437 | 79.625 (0.988x) | 1.024x |
| sketch | 11.055 (1.021x) | 11.162 (1.022x) | 11.688 (1.053x) | 0.831 | 78.533 (0.974x) | 1.043x |

The fused operators took the rule from 1.21x to 1.05x. What is left: +0.46
ms/step over indexer-only (11.408 - 10.946) = 42 us per layer-step for the
threshold + fire + compaction + row gather -- in line with the 81 us/layer
measured in-graph on a GPU shared with a serving process. The rule is now
within 3 points of the sketch arm on median and ahead of it on p99 (1.056x vs
1.053x, tied) and on jitter (std 0.374, below DSA's 0.504), at 0.84x DSA's
bytes and DSA's quality. The next cost lever is the operator chain itself
(five launches + compact_fired's two + three torch nodes per layer).

### v5 at 4k/124k: the same reading at the long-generation shape (2026-10-03)

`c521br3-4k124k` (v5, STATS off, decode-only stream, seed 0), same tree,
check_builds clean, no traceback:

| 4k/124k | median ITL | p99 ITL | output tput | TTFT |
|---|---|---|---|---|
| DSA | 10.874 | 11.496 | 91.854 | 1.000x |
| **branch v5** | **11.510 (1.058x)** | **12.096 (1.052x)** | 86.765 (0.945x) | 1.016x |
| branch v1 | 13.193 (1.213x) | 14.481 (1.260x) | 75.730 (0.824x) | 1.027x |
| indexer-only | 10.977 (1.009x) | 11.826 (1.029x) | 90.883 (0.989x) | 1.025x |
| sketch | 11.131 (1.024x) | 11.818 (1.028x) | 89.682 (0.976x) | 1.042x |

Both shapes, both readings, the same order: the branch rule on the fused
operators is 1.05-1.06x DSA on median and p99 and 0.95x on throughput, with
DSA's quality (1.000 x3 on multikey_3) and 0.84x DSA's bytes at 32k. The v1
records (`c521br-{32k4k,4k124k}`) are retired to superseded/.

### The byte claim, measured (2026-10-03)

VKSTATS now carries `bytes[step dsa ratio per_tok]`, computed from the same
per-scan sums as `attended_frac` (33 B/token for the pooled index-k read both
arms make, 1 KB per attended bf16 latent row, DSA = the index read + 2048
rows). `c521br4-smoke` (v5, 32k/256, STATS, 250 steps x 11 layers):

    fetch[p50=360 p90=1056 p99=1876]  fetched=451/call  kept=1120/call
    attended_frac=0.0495  bytes[step=2.66MB dsa=3.14MB ratio=0.845 per_tok=83.7B]
    ovf_by_layer=[0 x 11]  no traceback, no logging error

So at 32k the branch rule reads **0.845x DSA's bytes per layer-step** (83.7
B/token against DSA's 99 B/token), with no fenced step, at DSA's quality
(1.000 x3) and 1.05-1.06x its ITL. Two smokes agree on the fetch distribution
(p50 344 / 360, fetched 470 / 451 rows per call).
