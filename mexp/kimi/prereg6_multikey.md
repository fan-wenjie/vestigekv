# Pre-registration 6 (DRAFT): the multi-key gap, and two changes aimed at it

Status: **draft**. It freezes when the quality queue empties and the Kimi
calibration dump below has run, because one arm's parameter is chosen from that
dump. Nothing else is revised after data.

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

Two things follow, and each gets an arm.

**The clamp may bind before k does.** `Z_MAX = 8.0`. On dumped GLM-5.3
snapshots the calibrated z at k=1 is 6.98 to 8.31 -- one layer already above
the clamp, so its conformal requirement is not met even for one row and the
clamp silently weakens the guarantee. This is a constant, not an algorithm
change, and it is tested first because it is cheaper and because a result here
changes what the second arm has to explain.

**The target is the wrong set.** Certifying the top-k instead of the best is
one expression: `z_req = max over the top-k archived rows of
(true - idxs) / cert`. The quantile then certifies all k jointly. On the same
GLM snapshots, k=3 costs a few percent of z and 10-33% more fired rows at p90
(`mexp/kimi/z_topk_offline.py`), which is cheap against a 0.140 gap -- but that
is GLM geometry (rank 128, no sidecar) and Kimi's is different, so the number
that picks k comes from a Kimi dump, not from those snapshots.

## Step 0, before the arms: a Kimi calibration dump

One RULER job at 64k with the snapshot dump on, then
`z_topk_offline.py --dir results/kimi/caldump --ks 1,2,3,4`. It reports, for
Kimi's geometry, the calibrated z and the fired-row count at each k. **k for
arm A2 is whichever k is the largest with a fired-row p90 no more than twice
k=1's**, chosen from that table and fixed before any arm runs. If no k>1
satisfies that, A2 runs at k=2 and the table is reported as the reason.

## Arms

All on the delivered tree, 13 tasks x 4k-64k, **n=50** (one sample moves a cell
by 0.02, against 0.10 at n=10, and the effects under test are 0.08-0.14), stats
on.

- **A0** the delivered default. Its n=50 numbers come from
  `ruler-vestigekv-n50`, already queued; no separate run.
- **A1** `Z_MAX = 16`. One constant.
- **A2** joint top-k calibration at the k step 0 picks. One expression.
- **A3** A1 and A2 together, run only if both pass their own gates.

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

Among arms passing every gate, take the highest targeted mean; ties go to the
one that changes less, which orders A1 before A2 before A3. The chosen arm's
constant or expression becomes the default in one commit, and the paper's RULER
table is re-measured under it.

If none passes, the default is unchanged and the paper reports the multi-key
gap as measured, with this pre-registration's arms as the attempts that did not
close it. That outcome is worth stating: it is the difference between a method
that is weak on multi-key and one whose weakness is understood.

## What this does to the paper

The body's task-family claim is unaffected either way -- single-needle,
multi-query, multi-value and variable-tracking are all exactly 0 against dense
at n=10 and are re-confirmed at n=50 by gate B2 of pre-registration 3. What
changes is the paragraph explaining the multi-key gap: today it can only report
it, and an adopted arm turns it into a cost that was identified, priced and
paid. The certificate's description gains "the top-k" in place of "the best"
if A2 is adopted, and the recommended configuration gains a second knob.

## Not revised after the data

The gate thresholds, the arm list, the targeted-mean definition, the choice
rule, and the rule that picks k in step 0. The deadline is 2026-09-26; an arm
that has not reported by 2026-09-23 is dropped rather than rushed.
