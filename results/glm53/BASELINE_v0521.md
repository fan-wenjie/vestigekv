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
