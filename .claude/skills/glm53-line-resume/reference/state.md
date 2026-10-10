# Where the line stands, and what not to redo

## Settled

**Latency crossed.** `SGLANG_DSA_VK_SELECT=1` with q=0 is **1.0058x** end-to-end
against DSA at 4k/124k decode-only: throughput 92.421 / 92.414 / 92.426 / 92.318
tok/s against a floor of **91.8617 +- 0.0322** (n=6), i.e. 16.5 sigma. Median
1.0049x. The kernel is 2.1x DSA's rank top-k at the served fire rate, flat in
context (4.36 -> 4.35 us while DSA goes 6.40 -> 14.58), holding through bs=8.

**Quality did not.** `ruler_cwe` @65536, n=50, paired per sample on identical seeds:

    vk    0.758  0.702  0.712  0.784  0.706     mean 0.7324  sd 0.0366
    DSA   0.960  0.956  0.920  0.954            mean 0.9475  sd 0.0185
    gap 0.2151, se 0.0188, t 11.44, 95% CI 0.172-0.258, complete separation

Every `niah_*` ties DSA at 1.000. So **glm-line-order-and-paper-rule's crossover is
NOT established**; the defensible claim is a speed/quality trade with a workload
dependence.

**The cause, measured rather than inferred.** The arm finds the right words and then
repeats itself -- at 64k it duplicates items in 26/50 cwe docs against DSA's 9/50 and
derails into self-correction in 18/50 against 5/50, while naming a comparable number
of items (7.94 vs 8.32). Recency telemetry (`SGLANG_DSA_VK_SELECT_TELEMETRY=1`,
decode graphs off) measured the deficit `dsa_frac - sel_frac` on recent groups:

    j (groups back)    0       1       2     ...     7
    4k             +0.089  +0.087  +0.074        +0.047
    64k            +0.165  +0.180  +0.172        +0.147
    worst layer 7 at 64k: newest group selected 42.3% where exact top-k takes 76.9%

Nothing in DSA's selection exempts recent rows -- the only unconditional ones are
`seq_lens % pool_size`, the 0-3 leftover tokens with no pooled group -- so the
model's own output is threshold-gated the moment it completes a group of 4 tokens.

## Do not redo

**W=16 recency window: REFUTED.** Implemented correctly and committed
(`SGLANG_DSA_VK_SELECT_RECENT_W`, default 0; window excluded from candidacy so it
cannot define the carried threshold, and paid out of the budget so row counts stay
byte-identical). Result: cwe 0.738 / 0.694 / 0.668, diff vs W=0 **-0.032, se 0.026,
t -1.24** -- nothing, while DSA stays 0.248 away. multikey held at 1.000.
**Do not retry at larger W:** W=25 reaches 91.6% of a deficit W=16 already took
73.9% of, against a measured response of zero.

**Nine structural predictions refuted.** scatter 40->4.3 us; merge 13->0.01; 5->2
stages 107->-0.02; contention large->4%; page-table scatter->0.00; n_sel->0.00; wide
expand +0.2 tok/s->0 and worse below budget; W=16->0. The only two that ever worked
were **deletions of work**: skipping DSA's top-k (157.4 us/step) and removing a
redundant scan of the global score row (1.017 tok/s, 0.9947x -> 1.0058x).

**Norm / low-rank screening on NoPE GQA (Nemotron-3-Nano): closed.** Centered
effective rank 63-93 of 128 at 90% energy; sound Cauchy-Schwarz prune rate
**0.0000%** at every layer and budget; norm-as-selector at or below random. The
bound is loose by ~sqrt(d) because q and r are near-orthogonal in 128 dims -- and
`mexp/glm53/sigma_conditions.py` already carried that argument in its docstring,
including "isotropy is what makes the estimate exact AND the bound loose by
sqrt(d); both cannot be had". **Search that file before re-deriving it.**
The verdict is cache-specific: centring lifts GLM's index-k from 0.1406 (random
0.1234) to 0.6825 and does nothing on Nemotron's K. Measure `CV(||r||)` and
`t/(||q|| E||r||)` per model rather than concluding norm screening is dead.

**Phase probe: a bounded null with no range.** Phase is `p % 4` -- pooling is
globally aligned (`pool_start_id = first_pos // kpool`, chunk starts asserted
aligned). Both arms read **1.000 at all four phases** (n=30), and decoys do not help
(1.0000 at 4, 16, 64, 256). That bounds any niah-family phase gap under ~10 points
and excludes arXiv:2609.36322's 40-point magnitude here -- but it measures nothing,
because niah is saturated on this model. A probe with range must target
**aggregation**, not lookup: the cells with headroom are `qa_hotpot` 0.68-0.78,
`qa_squad` 0.757-0.843, `fwe` 0.773-0.927, `cwe` 0.668-0.784 on vk.

**GLM-5.3 has no vestigial sidecar.** `qk_rope_head_dim = 0`, so the 64-dim un-roped
branch the project is named for exists on Kimi-Linear and not here; the dsa arm runs
sigma on DSA's index-k instead. A GQA model has no such branch at all, which is the
structural reason the Nemotron direction closed.

## Open, and not mine to decide

1. The **aggregation phase probe** (align every cwe target occurrence to one phase).
   Two probes already failed the same way, so review the design before running it.
2. **Keep or revert the W=16 kernel change** -- default-0 and harmless, but now dead
   code with two tests guarding it.
3. **Writing the line up as the trade it measured**, with the unmeasured phase
   question and the fixed-window limitation stated as limitations.
