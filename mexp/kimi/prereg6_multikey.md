# Pre-registration 6: the multi-key gap, and the change aimed at it

Status: **frozen 2026-09-17 06:45**, after step 0's calibration dump and before
any arm ran. Step 0 dropped one of the two arms and replaced the other's target;
both changes are recorded below with the measurement that caused them.

## The gap, and where it is

RULER 4k-64k, n=10, the delivered implementation against dense. Every
difference, per task:

| task | dense | delivered | diff |
|---|---|---|---|
| niah_multikey_3 | 1.000 | 0.860 | **-0.140** |
| niah_multikey_2 | 1.000 | 0.920 | **-0.080** |
| ruler_qa_hotpot | 0.760 | 0.680 | **-0.080** |
| ruler_fwe | 0.947 | 0.900 | -0.047 |
| ruler_cwe | 0.980 | 0.976 | -0.004 |
| ruler_qa_squad | 0.562 | 0.608 | +0.047 |

Every other task is exactly 0. The 65-cell mean is 0.9179 against dense's
0.9413, so **the whole -0.0234 sits on tasks that need more than one archived
row at once**: two keys, three keys, and a two-hop question.

## Why, structurally

The conformal calibration solves, per calibration query, for the z that makes
the certified score of its **single best** archived row reach that row's true
score (`recall_tier.py`: `z_req = (abest_val - idxs_t) / cert_t`, where
`abest_val` is a running maximum). So the guarantee it buys is marginal and
about one row: P(the best archived row is fired) >= RECALL_TARGET = 0.90.

A marginal guarantee does not compose. A question needing k distinct archived
rows gets roughly tau^k, and the measured means fit that shape: multikey_2 at
0.920 and multikey_3 at 0.860 back out a per-key 0.959 and 0.951, consistent
across k, while every single-needle task sits at 1.000 because tier-1 usually
keeps a single needle and recall is never exercised.

Step 0 measured both candidates before either ran. One did not survive it.

**The clamp does not bind on Kimi.** `Z_MAX = 8.0`, and the dumped Kimi
snapshots calibrate z between 0.48 and 3.59 with the stored `zp` at 2.28. The
GLM-5.3 snapshots that suggested this arm sit at 6.98 to 8.31, so the clamp
binds there and not here. **Arm A1 is dropped before running**, and the
observation is kept because it says the two geometries are not interchangeable
for this question.

**The target is the wrong set, in both directions.** Certifying the top-k was
the first proposal; measuring it says the question of which k does not arise.
What actually reaches the output is the set of archived rows whose true score
beats the best kept row -- nothing else is ever attended -- and that set is
data-determined per query and tiny. Over 12 Kimi snapshots its size has a
median of 0, a p90 between 0 and 21, and a maximum of 57, and **44% to 100% of
calibration queries have no such row at all**: the kept set already holds
everything that matters and recall has nothing to do.

Against that, today's target is mis-specified twice over. It certifies the best
archived row whether or not that row beats the kept maximum, so on the 94% of
queries where it does not, the requirement raises z for a row the output would
never see; and where several rows do beat it, only one of them is guaranteed,
which is exactly the multi-key failure. The measurement shows both signs: on 7
of 12 snapshots the correct target calibrates to a **lower** z than today's
(0.483 against 2.286 at the extreme), and on the snapshots where several rows
beat the maximum it calibrates higher (3.589 against 2.797).

## Step 0 is done

`caldump-kimi-64k` ran and the numbers above come from it. The rule that was to
pick k is retired with arm A2's fixed k: the set is not chosen, it is the one
the data defines.

## Arms

All on the delivered tree, 13 tasks x 4k-64k, **n=50** (one sample moves a cell
by 0.02, against 0.10 at n=10, and the effects under test are 0.08-0.14), stats
on.

- **A0** the delivered default. Its n=50 numbers come from
  `ruler-vestigekv-n50`, already queued; no separate run.
- **A2** the corrected target: `z_req = max over {archived rows whose true
  score beats the best kept row} of (true - idxs) / cert`, and a query with no
  such row imposes no requirement instead of the one it imposes today. One
  expression in `recall_tier.py`.

A1 is dropped (the clamp does not bind on Kimi) and A3 with it.

## Gates

Targeted mean = the mean of niah_multikey_2, niah_multikey_3 and
ruler_qa_hotpot. A0 measures 0.820 at n=10; dense is 0.920.

- **Q1** targeted mean >= **0.870**, half the gap to dense, at n=50.
- **Q2** 65-cell mean >= A0's - 0.010: the arm may not pay for multi-key out of
  the rest.
- **Q3** no task below A0 by more than 0.03.
- **C1** fallback rate at 64k <= 0.50 (A0 measures 0.36).
- **P1** the 256k production stream within 2% of A0's 4.364 ms/token.
- **S1** every registered unit and kernel test passes; the head needle
  recovers.

An arm that fails any gate is reported as failed and not adopted.

## Choice and adoption

Among arms passing every gate, take the highest targeted mean; with only one arm
left there is no tie to break. The chosen arm's
constant or expression becomes the default in one commit, and the paper's RULER
table is re-measured under it.

If none passes, the default is unchanged and the paper reports the multi-key
gap as measured, with this pre-registration's arms as the attempts that did not
close it. That outcome is worth stating: it is the difference between a method
that is weak on multi-key and one whose weakness is understood.

## Amendment 1 (2026-09-17): what RULER's multi-key gap is a gap in

**RETRACTED the same day; see Amendment 1R below. The text is kept because the
arms it justified were built on it.**

Tier-1's signal is the norm of a row's residual after the close window is
projected onto a 31-dimensional trigonometric basis: what it keeps is what the
block's low-frequency structure fails to explain. That is a **prior about
natural text**, a shortcut for the ordinary case, not a general relevance
score. RULER is built to defeat exactly such priors, so a gap there is partly a
gap outside the method's stated applicability rather than a defect inside it.

The measured fallback rates order themselves by how natural the text is, not by
how many decode steps amortise the calibration: the model's own continuation
0.003/0.055, real documents 0.241/0.171, needles in a haystack 0.360/0.324,
random tokens 0.407/0.439. And on real documents the rate is 24% while the
accuracy is *identical to dense* over 300 questions. A high rate is the net
bounding what the prior stops covering, which is the design working.

## Amendment 1R (2026-09-17): the naturalness ordering was the confound

Amendment 1's ordering is **also an ordering by decode length** -- those four
runs generated about 250k, 1, 14 and 14 tokens. The two axes were never
separated, and "not by how many decode steps amortise the calibration" was
asserted, not measured. Measured, it is wrong.

`lit-continue-64k-short` and `lit-continue-64k-long` continue a real novel
(Journey to the West, The Count of Monte Cristo, Don Quixote, truncated to
exactly 64k tokens) and vary only the generated length. With
`stats-stream-64k-x130` and `stats-stream-64kprefill-long` holding the length
and varying only the text, the 2x2 closes:

| | 14 decode steps | 4096 decode steps |
|---|---|---|
| random tokens | 0.407 (97.8% provisional) | 0.0023 (0.6%) |
| **a real novel** | **0.483 (97.9% provisional)** | **0.0032 (0.6%)** |

- **Decode length moves the rate 151x.** Same novel, same context.
- **Text type moves it 1.6x, in the WRONG direction.** At 14 steps a novel
  falls back MORE than random tokens (0.483 against 0.407) and far more than
  RULER (0.305). Prose is the most favourable input a prior on ordinary
  language can get; it is the worst case measured.

So the spectral prior's fit to the text does **not** explain the fallback rate.
What explains it is the share of steps served before calibration completes:
97.9% for the novel at 14 steps, 0.6% at 4096, and the per-state overflow rates
are the same in both (0.455 and 0.403 provisional).

One mechanism does track the text: at the Z_MAX clamp the number of rows fired
is content-dependent -- 2483 for the novel, 1958 for random tokens, 1663 for
RULER -- so natural prose makes the *provisional* index fire more and overflow
more. That is a statement about the clamp, not about the prior.

**Consequences.** The paper may not order fallback by naturalness, and may not
present a high rate on RULER as the method working at the edge of its stated
applicability. Arm A3 below was derived from Amendment 1 and its premise is
gone: it is **dropped**, and its prerequisite (a sigma-histogram statistic
separating text types) is not worth building, because the thing it would select
for does not drive the rate.

## Arm A3: fall back where the prior fails, and only there

If A2 does not close the gap, the next thing to try is not a better certificate
but an honest admission that the certificate has nothing to work with. The
detector is already computed: `sigma_fused` builds a 1024-bin histogram of the
close window's sigma values at every block close. A block whose sigma
distribution has no structure -- no rows standing out from the rest -- is a
block where the spectral prior has failed, and its rows should be attended
rather than ranked.

The arm is: at block close, derive a structure statistic from the histogram
that already exists, and mark a block that fails it so its rows are attended
densely. Fallback rises only where the prior fails, and natural text pays
nothing.

**Prerequisite, and it is falsifiable on its own.** The statistic must separate
the text types before any serving arm is built: computed offline on calibration
dumps from a random-token stream, a RULER haystack and a LongBench document, it
must order them the way the fallback rates do. If one histogram statistic
cannot tell random tokens from a real document, the detector does not exist and
the arm is dropped without running -- the same way A1 was.

**Accuracy is the binding constraint, not the rate.** A3 is adopted only if it
holds Q2 and Q3 (no loss anywhere) while improving Q1, and its cost gate is P1
rather than C1: raising the fallback rate is the mechanism, so capping the rate
would cap the mechanism. What may not rise is the latency, and what may not
fall is any accuracy.

## Outcome (2026-09-17): A2 and A4 attack an step that is not failing

Three measurements, none of which needed the arms to finish.

**The calibrated certificate misses nothing.** On the Kimi snapshots, at the
fitted zp, **1081 of 1081** archived rows whose true score beats the kept
maximum are already fired -- 100% at gain 0, and every candidate entropy gain
up to 2.0 recovers the same 1081 while firing more rows
(`mexp/kimi/ent_margin_offline.py`). A4 has nothing to recover. A2 aims at the
same step from the other side, choosing which rows the quantile must cover, and
that step is not where rows are lost either.

**A2's probe moved nothing on multi-key.** At n=10 against A0 it changed 6 of
100 multi-key answers with a net of exactly zero: multikey_2 0.920 and
multikey_3 0.860 in both arms. Its 43-of-650 total perturbation is the
run-to-run floor.

**The truncation experiment found a real but secondary effect.** With the
fallback off, keeping the same number of rows spread over the archive instead
of as a positional prefix lifts multikey_2 from 0.200 to 0.367 and multikey_3
from 0.100 to 0.300. Position is costing something, and it is a free fix, but
0.300 against the 0.860 the fallback delivers says it is not the main term.

**Both arms are stopped**, A2's remaining half unrun. What the three together
say is that the failures are not on the steps served by a correctly calibrated
index: that index is perfect on its calibration queries. They are on the steps
served by something else -- the provisional identity-basis index with z at
Z_MAX before calibration completes, or an index that has gone stale against a
context that has moved. A RULER answer is about fourteen steps and its layer
builds once, so most of its steps are exactly those. That is the next thing to
measure, and it needs an instrument that attributes a miss to the index that
served it, which does not exist yet.

## The amortisation reading, confirmed (2026-09-17)

`stats-stream-64kprefill-long` resolves the variable `stats-stream-64k-x130`
left confounded, and it resolves it the way the outcome above predicted.

| run | context arrives by | decode steps | text | fallback |
|---|---|---|---|---|
| RULER 64k | 64k prefill | 14 | needle question | 0.360 |
| `stats-stream-64k-x130` | 64k prefill | 14 | random continuation | 0.41 |
| `stats-stream-64kprefill-long` | 64k prefill | **4096** | random continuation | **0.0027 / 0.0044** |
| long stream at 64k | decoded up to | 258050 | random continuation | 0.00014 |

Rows 2 and 3 differ in **one** variable, the decode count, and the rate falls
by about a hundredfold. The pre-registered reading was "near 0.0002 puts the
cause on the step count; near 0.3-0.4 puts it on the context's provenance"; at
0.0027-0.0044 it is two decades from the first and two from the second, and on
the log scale that decides a ratio it sits with the step count. The context's
provenance is not what drives the rate, so the paper's "long-decode number"
qualifier needs no second clause about how the context was built.

What is left is the mechanism, and it is the one the outcome above named: a
fourteen-step answer spends nearly all of its steps on an index that has not
finished calibrating, so it is not a statement about the question being asked.
`idxstate-ruler-64k` and `idxstate-stream-64kprefill` measure that directly,
with the prediction and its falsifier registered in README before either ran.

## Three recall-side mechanisms tested and refuted (2026-09-17)

The outcome above rested on "1081 of 1081 rows recovered", and that number was
measured **in sample**: `ent_margin_offline.py` scores the fitted zp on the very
queries zp was fitted on, and a conformal quantile recovers its target in
sample by construction. The ruling that stopped A2 and A4 therefore rested on
an artifact, whatever its conclusion. `cert_holdout_offline.py` refits zp on
half the calibration points with the delivered rule and scores the other half.

| | row recall | query recall (ALL needed rows) |
|---|---|---|
| in sample | 2154/2157 = 99.9% | 478/481 = 99.4% |
| held out, random split | 973/974 = 99.9% | 229/230 = 99.6% |
| held out, position split | 1283/1286 = 99.8% | 279/282 = 98.9% |

Out-of-sample query recall by how many rows the query needs: 100% at k=1, 97.8%
at k=2, and 100% at every k from 3 to 8. **It does not decay with k**, which is
the decay the whole multi-key story predicted. The prediction registered in
README before running -- "out-of-sample row recall lands near 0.90 and query
recall falls off with k" -- is falsified, and the A2/A4 ruling survives a test
that could have overturned it.

Two further loss channels, both checked because the holdout script does not
model them:

- **The entropy gate.** A closed gate sets the threshold to +inf and the query
  fires nothing, which no offline scan above would see. All 42 snapshots report
  `gate_off: True` -- the self-disable at `GATE_SELF_DISABLE_FRACTION` fires
  everywhere on this workload -- so the gate is not a loss channel here.
- **Negative zp.** Seven of 42 snapshots calibrate zp below zero (to -1.584),
  which deflates the index score and fires FEWER rows than the sketch alone.
  Those seven have 100% query recall and clamping at `max(zp, 0)` changes not
  one fired row.

**What this closes.** The recall path is not where RULER loses rows, and that
now rests on a held-out measurement rather than an in-sample one. The median
fired-row count is 0: the kept set already holds what beats it. So there is no
headroom in the certificate, the gate, or the quantile, and a mechanism aimed
at any of them is aimed at a step that is not failing.

**What is left**, in the order the evidence supports: (1) the gap is at least
partly n=10 noise -- the cross-run answer churn is 40 of 650 = 6% and
multikey_3's gap is 14 of 100 -- which `ruler-vestigekv-n50-tgt` settles; (2)
tier-1's keep decision, which none of the above tests, since every offline scan
here conditions on the kept set as given.

## The index-state buckets, measured (2026-09-17)

Instrument verified before reading: `calls/step=1.00`, and RULER reports
`steps=1850`, the same count `engine/` gives. The earlier bucket numbers
(prov 28%) came from the run whose prologue was entered ~3x per step and are
**superseded**; they inflated the fresh bucket 11-fold.

| | RULER 64k | stream, 64k prefill + 4096 steps |
|---|---|---|
| provisional | **10487 scans (81%)**, 1662.7 rows, ovf **0.3152** | **180 (0.6%)**, 1812.8 rows, ovf **0.3278** |
| fresh | 2463 (19%), 1061.0 rows, ovf 0.1839 | 28520 (99.4%), 2.3 rows, ovf 0.0003 |
| stale | 0 | 0 |
| fallback | 0.305 | 0.00233 |

**A provisional index overflows at the same rate in both workloads**, 0.315
against 0.328. What differs by 130x is not the rate but the MIX: 81% of
RULER's scans are served before calibration completes, against 0.6% of the
stream's. The arithmetic closes -- 0.006*0.3278 + 0.994*0.0003 = 0.0023 against
0.00233 measured -- so the fallback rate is the state mix times the per-state
rate and nothing else.

The registered prediction is confirmed: RULER's scans ARE mostly provisional,
and its overflows are concentrated there. `stale` is exactly 0 in both, so a
fourteen-token answer never outgrows its index and one of the three states this
instrument was built for is dead weight on this workload.

**What it does and does not say.** It explains the FALLBACK gap completely, and
it says the gap is about how many steps run before calibration, not about the
text or the question. It does not yet explain the multi-key ACCURACY gap: a
provisional index over-fetches (1663 rows against fresh's 1061) and its
overflows fall back to dense, both of which are correct-but-slow, not wrong.

## The omitted-mass arm: rejected, and the metric that recommended it (2026-09-17)

**The finding that motivated it stands.** VestigeKV attends about 6% of rows
and its attended set holds only ~57% of the dense softmax mass (min 24% over
the Kimi dumps), so every retained weight is inflated by Z/Zv, mean 2.9x. No
measurement before this one looked at the softmax SCALE; they all asked whether
the row that beats max1 is fired, and it is.

**The correction is exactly one turn of the online-softmax recurrence**
`O_n = lerp(O_{n-1}, v_n, sigmoid(s_n - lse_{n-1}))` against a synthetic row
standing for the whole omitted set, with mass M from the certified bound the
scan already computes and value that set's mass-weighted centroid. Offline it
cut the attention-output error against dense from 0.471 to 0.195.

**It destroys retrieval.** On its first working run it failed the head needle --
answering `7-ZEBRA-REVIEWED-NORTHERN-DEPOTS` where the code is `7-ZEBRA-4419`,
the prefix right and the tail confabulated -- and cost 0.109 over the 65-cell
RULER grid, damaging every task family at every length above 4096. At 4096 it
is exactly 0 on all 13 tasks, because a 4096-token context has no closed block
and so no archive: that zero is what proves the arm really ran.

**It is not a bug.** Two unit tests recompute logM and the centroid
independently from the tier's own operands and both match
(`test_vestigekv_recall_tier.py::TestOmittedMass`); a blend-off control on the
same tree answers the needle correctly. On the real dumps the certified bound
inflates the omitted log-mass by only 0.56 nats, giving sigma 0.517 where the
true omitted scores would give 0.647. So the arm replaces **48% of the
attention output with the archive's centroid**, and a correct estimate would
still replace 35%.

**Why the offline metric recommended it anyway.** Relative L2 against dense is
minimised by moving toward the mean, so mixing in a centroid improves it
*by construction* while flattening exactly the peak that retrieval reads. The
metric could not distinguish "closer on average" from "still pointing at the
needle", and 0.471 -> 0.195 was the metric rewarding the failure mode. **Any
future arm that changes the attention output must be gated on a retrieval
check, not on output error.** The head needle costs one second and would have
ended this arm before the first RULER run.

**Rejected.** The mass gap is real and remains the one unexplained term; the
centroid is not the way to close it.

## Adoption ruling (owner, 2026-09-17)

If A2 passes every gate, it is adopted as the final algorithm without checking
back. "Passes every gate" is the whole of the condition and it is already the
definition of leading across the board: Q1 requires the multi-key gain, Q2 and
Q3 forbid paying for it anywhere else, C1 caps the fallback rate and P1 caps
the latency. An arm that improves multi-key by less than Q1's threshold, or
that gains there and loses elsewhere, is **not** adopted under this ruling --
it is reported with its numbers and the decision returns to the owner.

Adoption is these steps, in order, and nothing else:

1. Merge `vestigekv-multikey` into the delivered branch
   `vestigekv-fused-fallback`; the arm is one expression in `recall_tier.py`.
2. Re-run the registered unit and kernel suites and the head needle (gate S1).
3. Re-measure the 256k production stream on the merged tree and confirm P1
   against A0's 4.364 ms/token.
4. Regenerate the paper's RULER macros from A2's n=50 results
   (`make_ruler_numbers.py --n 50`) and rewrite the certificate's description:
   it certifies every archived row that beats the kept maximum, not the best
   archived row.
5. Record the outcome in this file, including every gate's measured value.

## What this does to the paper

The body's task-family claim is unaffected either way -- single-needle,
multi-query, multi-value and variable-tracking are all exactly 0 against dense
at n=10 and are re-confirmed at n=50 by gate B2 of pre-registration 3. What
changes is the paragraph explaining the multi-key gap: today it can only report
it, and an adopted arm turns it into a cost that was identified, priced and
paid. The certificate's description gains "every archived row that beats the kept
maximum" in place of "the best archived row" if A2 is adopted. No knob is
added: the set is defined by the data, not configured.

## Not revised after the data

The gate thresholds, the targeted-mean definition and the choice rule. The arm
list shrank before any arm ran, on step 0's measurement and for a stated
reason; that is the pre-registration working, not a revision after data. The deadline is 2026-09-26; an arm
that has not reported by 2026-09-23 is dropped rather than rushed.
