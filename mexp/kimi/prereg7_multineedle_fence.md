# Pre-registration 7: the multi-needle fence

Status: **frozen 2026-09-18 01:0x**, written after pre-registration 6's arm
failed Q1 and BEFORE the three confirming jobs report. All three
(`fence-mk256-n50-all13`, `fence-mk256-stream-256k`, `randfence-20-ruler`) were
queued before this file existed; none has produced a number.

## Why a new pre-registration rather than a re-reading of the old one

Pre-registration 6's arm failed Q1 at n=50: targeted mean 0.851 against 0.870.
Per-task, it lost on nothing and gained on four of five. Both statements are
true, and the second is not licensed by the first document, because Q1's
targeted mean averaged `ruler_qa_hotpot` -- multi-HOP reasoning, which the
mechanism does not address and which gained exactly 0.000 -- into a target
named for multi-KEY retrieval. Re-cutting that analysis after seeing the data
is what pre-registration exists to prevent, so the old gate stands as failed
and this states the corrected one before the confirming data arrives.

## The mechanism, stated as it is now understood

`--vestigekv-multikey-fence-rows T`: a lane firing more than T archived rows
attends its full row set instead of ranking, which is dense and therefore
exactly correct for that lane. It costs fallback and nothing else.

**It is not known to be a detector.** The answer-step attribution says the
fired-row count fences 32% of records and covers 19.5% of the steps that lose
their top-scoring archived row -- worse than chance, because 99.5% of what it
fences is a provisional tier while 68% of the misses are on calibrated ones.
So the working hypothesis is that the fence gains by making a fraction of steps
exactly dense, not by choosing which. `randfence-20-ruler` tests that.

## Target family and controls, fixed here

- **Target: multi-needle retrieval** -- `niah_multikey_2`, `niah_multikey_3`.
  These are what the mechanism addresses: several archived rows must reach one
  output.
- **Control A, multi-hop**: `ruler_qa_hotpot`. Chaining two retrievals is a
  different failure and the mechanism is not expected to move it. Reported
  separately; it is NOT in the target.
- **Control B, no harm**: the remaining ten tasks.
- **Control C, selection**: `randfence-20-ruler`, a random fence at matched
  cost.

## Gates

All at n=50, 4k-64k, against A0 at the same n.

- **N1** both multi-needle tasks improve, and `niah_multikey_3` by >= 0.05.
- **N2** no task among all thirteen falls by more than 0.02.
- **N3** the 256k production stream is within 2% of A0's 4.364 ms/token. The
  fence fires on lanes emitting many rows, which is 81% of RULER's scans and
  0.6% of a long decode's, so the prediction is that it is nearly free exactly
  where the speedup lives. If it is not, the fence is a short-answer-only knob
  and must ship off by default with that stated.
- **N4** governs the CLAIM, not adoption. If `randfence-20-ruler` matches the
  gain within 0.02 on the target, the fence is described as "attend densely on
  a fraction of steps", the fired-row count is not called a detector, and the
  simpler formulation (a rate) is preferred over the threshold. If random does
  NOT match, the count is doing real work and may be described as such.

An arm failing N1, N2 or N3 is reported failed and not adopted. N4 cannot
block adoption; it decides what the paper is allowed to say.

## N3 FAILS (2026-09-18): the fence is not free where the speedup lives

Server-side ms/token, same script and window as the delivered curve,
`td-stream-512k` against `fence-mk256-stream-256k`:

| context | A0 | fence 256 | cost |
|---|---|---|---|
| 8k | 3.955 | 3.946 | 0% |
| 32k | 4.088 | 4.141 | +1.3% |
| 64k | 4.125 | 4.371 | **+6.0%** |
| 128k | 4.184 | 4.737 | **+13.2%** |

N3 allows 2%. At 128k the cost is 13.2% and **rising with context**, and it
eliminates the speedup: dense is 4.786 there, A0's 4.184 is 1.144x, and the
fence's 4.737 is 1.010x.

**My prediction was wrong and the reason is worth keeping.** I argued the fence
would be nearly free on a long decode because its fresh bucket fires 2.3 rows
per scan (p50 0, p90 <= 2), far under a threshold of 256. The firing RATE is
indeed low; what I ignored is the cost of each firing. A fenced lane attends
its full row set, and at 128k that is about thirty times a compressed step, so
a rare event at a divergent price still diverges. Rate alone never priced this.

**Consequence.** By the adoption rule the arm is failed and not adopted as a
default. What N3's own text already allowed stands: the fence may exist as a
documented, off-by-default, short-answer knob, and the paper must say that the
speedup is gone above about 64k with it on.

## Adoption

If N1, N2 and N3 pass: the fence ships, with its default set by N3 -- on if the
long-decode cost is within 2%, off with a documented short-answer
recommendation if not. The threshold stays 256 and is reported as an
empirical constant, because the derivation from the tier's own calibration was
tried and fails: that distribution puts the fired count at 0 to 74 while
serving puts it at 0 to 5629, so nothing fitted at build time can set it.

## What this cannot claim, whatever the gates say

The fence does not fix multi-key. It makes some steps dense. The measured
mechanism of the failure -- on calibrated tiers at answer steps, multikey_3
loses its top-scoring archived row on 22.85% of records against single_1's
0.00% -- is untouched by it, and remains the open problem.

## Not revised after the data

The target family, the gate thresholds, and the rule that N4 governs the claim
rather than the adoption. The deadline is 2026-09-26.
