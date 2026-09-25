# The dividend inside prior algorithms

Technical note accompanying the VestigeKV paper's appendix *The dividend inside
prior algorithms*, which carries the same derivations. **This file is not a
substitute for that appendix and does not repeat it to be read instead** -- it
exists because a reviewer has no obligation to read an appendix at all, so the
material has to survive somewhere a reader who wants it can find it without the
PDF, and because the status table at the end is easier to keep current here than
in a frozen submission.

Nothing in it is a new measurement: every item is a derivation from the setup of
the paper's theory section, and where a measured record bounds the claim, the
bound is stated. The bounds are the part worth carrying: three of the seven
dividends are *empty* on this checkpoint.

## Why there is a family of them, not one

Two structural facts about a NoPE-MLA cache do the work:

1. the attention score is a single **fixed bilinear form** — no position enters it;
2. rankings **commute with time** (the stationarity corollary): the order of two
   archived rows under a given query direction does not change as the sequence grows.

Under RoPE neither holds, and each published cache-management policy below pays
for that in a different currency. Transplanted to NoPE-MLA, each gains a
distinct guarantee.

## Accumulated-attention eviction (H2O, TOVA)

These methods score row `u` by the attention mass observed under past queries,
`ŝ(u) = Σ_t softmax-weight(q_t, u)`. What that statistic estimates:

- **NoPE**: `E[ŝ(u)] = n · E_q[ w(λ⟨Q, C̃_u⟩) ]` — no `t` enters. One number per row.
- **RoPE**: `E[ŝ(u)] = Σ_t E_q[ w(λ qᵀ R_{t−u} W_K C̃_u) ]` — a function of the
  age profile `{t − u}`.

Under NoPE the estimand is time-invariant, so the estimate transfers to every
future query from the same distribution. Under RoPE a future query at
`t' = t + L_gen` evaluates the integrand at a rotation the statistic never
sampled; by the band expansion of the stationarity corollary the mismatch
oscillates per frequency, so the heavy-hitter statistic has a **shelf life** set
by the band sum. NoPE removes the shelf life.

## Observation-window voting (SnapKV)

SnapKV votes with the last window's queries immediately before generation. The
score a vote certifies and the score generation consumes differ, for the same
`(q, u)`, by exactly

```
s_{t'}(q,u) − s_t(q,u) = λ qᵀ (R_{t'−u} − R_{t−u}) W_K C̃_u,    t' = t + L_gen
```

which **vanishes identically** under NoPE (`R ≡ I`) and oscillates per frequency
band under RoPE. The vote's validity horizon is therefore unbounded under NoPE
and equal to the generation length under RoPE.

## Attention sinks (StreamingLLM)

The same difference term applied to a fixed sink row `u`: under NoPE the sink's
score against any query direction is length-invariant, so the mechanism is
**exactly stable**; under RoPE the difference rides the lowest frequency bands
and drifts with distance.

## Page-bound pruning (Quest)

Quest skips pages whose per-page score bound falls below the running best. Under
NoPE the score is linear in the row for a fixed form, so interval arithmetic over
a per-page box `B(P) = [ℓ, h] ⊇ {C̃_u}_{u∈P}` gives a true bound:

```
max_{u∈P} λ⟨Q, C̃_u⟩  ≤  λ Σ_d max(Q_d ℓ_d, Q_d h_d)
```

and pruning by it is **admissible**: no page containing the argmax is skipped
beyond box slack.

> **Measured caution.** Admissible is not automatically useful. On this
> checkpoint, boxes over the 256-dim index at page size 64 prune **under 0.2% of
> pages** — per-dimension maxima over 64 rows exceed every actual row's score in
> high dimension. Tight page digests are the open problem; the guarantee is the
> opening, not a working index.

Under RoPE the left side is `max_{u∈P} λ qᵀ R_{Δ_u} W_K C̃_u` with `Δ_u` varying
*inside* the page, so an honest bound must also maximize over the rotation orbit
— the radius inflation of the digest-obstruction appendix — and tightness is lost
with page span. Combined with the measured layer-uniformity of NoPE page digests,
this is the theoretical opening for a hierarchical recall index with a per-step
scan sublinear in context length. **We have not built or measured one.**

## Trainable projections (MatryoshkaKV)

For an orthogonal projector `Π` on the content latent, the NoPE score error is
exactly `|λ⟨Q, (I − Π) C̃_u⟩|` — a clean, query-uniform objective. Under RoPE the
same objective is position-dependent unless `Π` near-commutes with the whole
rotation family,

```
sup_Δ ‖ R_Δᵀ Π R_Δ − Π ‖  ≈  0
```

which ties the projector to the frequency pairing and shrinks the feasible set to
band-aligned subspaces.

> **Measured bound.** On the frozen Kimi Linear checkpoint the read-out is
> full-rank (rank 512), so this dividend accrues to future training, not to the
> deployed weights.

## Lossless merging (KeepKV)

KeepKV merges cache entries with the ambition of losslessness. The paper's
impossibility proposition is the exactness certificate that ambition needs, and
it is available only under NoPE-MLA: identical rows merge exactly with an `ln m`
bias, the RMSNorm homogeneity class enlarges the merge set, and **under RoPE the
merge class is empty**.

> **Measured bound.** On real corpora the tolerant merge class is empty at 5%
> tolerance, so the certificate currently has no application domain on this
> checkpoint.

## Position-independent reuse (Irminsul, Kamera)

Cross-request cache reuse is the exchangeability lemma applied across sequences:
a NoPE-MLA row is valid at any position in any context, so reuse is **zero-copy**.
RoPE reuse pays a re-rotation of every moved row. These systems exploit the
property; the lemma names it.

## Recall as search over a static set — and why the scan stays linear

The exchangeability lemma and the stationarity corollary say the archive is a
*static* point set and each step's trigger a maximum-inner-product query against
it. Under RoPE it is not: the effective key moves with `t − u`. So the obvious
next step is an index that prunes the scan.

The certificate does extend from a row to a ball of rows, adding
`λ‖q_sk‖ R` for a ball of radius `R`, and a ball whose bound stays under the kept
maximum could be skipped whole. **That slack is what kills it**: a radius falls as
`B^(−1/r)` in the rank-`r` sketch space, so at `r = 64` no practical number of
balls `B` tightens the bound, and the pruning test admits nearly every ball while
the certificate itself fires a few rows per thousand.

The deployment therefore answers the query by a linear scan with the per-row
certificate. A sublinear trigger needs a test that is **approximate** —
calibrated, as `z` already is, rather than sound.

## Training prospects

Everything in the paper is post-hoc. With training, the same mathematics marks
prospects — these are **not claims**:

- trained index channels could shrink sketch residuals by construction;
- repeated-span latents activate the impossibility proposition's collapse
  (cache size `O(distinct content)`);
- eviction-aware fine-tuning gains a stable target.

## Status of each item

| item | what NoPE-MLA confers | measured status on this checkpoint |
|---|---|---|
| H2O / TOVA | estimand loses the age profile | derivation only |
| SnapKV | vote drift term vanishes identically | derivation only |
| StreamingLLM | sink exactly length-invariant | derivation only |
| Quest | admissible interval arithmetic | **empty**: boxes prune < 0.2% of pages |
| MatryoshkaKV | query-uniform projection objective | **empty**: read-out already full-rank (512) |
| KeepKV | exactness certificate for merging | **empty**: tolerant merge class empty at 5% |
| Irminsul / Kamera | zero-copy cross-request reuse | exploited by those systems already |
