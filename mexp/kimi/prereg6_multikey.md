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

Two consequences for this pre-registration, both recorded before A2 reported.

- **Q1's threshold is not re-argued here**, but its meaning is narrower than it
  reads. Recovering half of RULER's multi-key gap is worth having; failing to
  is not by itself evidence that the method is weak on multi-key retrieval in
  natural documents, and the paper must not report it as if it were.
- **A third arm follows from the framing**, below.

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
