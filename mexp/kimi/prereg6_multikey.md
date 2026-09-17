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
