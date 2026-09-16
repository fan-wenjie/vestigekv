# Pre-registration 5 (DRAFT): the recall margin, asked again on a package that reproduces its reference

Status: **draft**. It freezes when `td-stream-256k` reports, because one line
of its package is whether the tier-decode router is adopted. Nothing below is
revised after data except that one line, and the freeze commit says which way
it went.

## Why the question is open a third time

Pre-registration 1 measured a real quality gain from a recall margin and
rejected it on cost. Pre-registration 4 re-asked it under a repaired cost
structure and every candidate failed, but the verdict was not about the margin:
its own no-margin arm scored 0.278 on the targeted mean where the arm it was
meant to reproduce scored 0.767. A reference that does not reproduce is not a
reference, so nothing in pre-registration 4 answers the question, and its
thresholds are not reused.

The component that broke it is known. Pre-registration 2's outcome isolates
prefill calibration: alone it costs 0.135 of the 65-cell mean and 0.58 on both
multi-key tasks, because prompt queries are not exchangeable with decode
queries, so a certificate calibrated on the prompt under-recalls during
decoding. It is out of this package and out of the default.

Two things measured since also change what the margin costs, and both pull the
same way.

1. **The per-step CSR copy is gone** (tier-decode router, if adopted). A larger
   fired set used to cost a copy of every fired row into one array on every
   step, on top of reading them. Now it costs only the read. The margin's price
   is whatever the extra rows cost the attention kernel, which is what the
   margin was always supposed to be trading against.
2. **The fallback is load-bearing** (pre-registration 3, outcome C1). Turning
   it off costs 0.0565 of the 65-cell mean, concentrated on the multi-key
   tasks. A margin that lowers the fallback rate is therefore not automatically
   an improvement: it has to lower it by fetching the rows the fallback was
   rescuing, not by giving up on them.

## The package under test

Today's default, plus exactly two changes, each already implemented and
tested:

- `--vestigekv-recall-capacity 16384` -- without it every candidate's fetch
  p50 measures the buffer width rather than what the margin fires, which is
  what censored pre-registration 1's cost column. 64 KB per (layer, lane).
- `--enable-vestigekv-tier-decode`, **if and only if** `td-stream-256k` meets
  its gate; otherwise the package is the capacity change alone. Either way it
  moves no row, so it cannot affect a quality arm.

Explicitly **not** in the package: prefill calibration, the rebuild-overflow
trigger, attended splits. Each has failed a gate of its own and each is a
confound here.

Candidates, fixed here and taken from pre-registration 1's table, not revised
after data: **max 2**, **max 3**, **lse 4.6**, against **max 0** as the
reference.

## The gate pre-registration 4 lacked

**R0, reproduction.** The max-0 arm is the same configuration
pre-registration 1 measured as "max, 0" except for the capacity. Its targeted
mean must land within 0.05 of 0.767 and its 15-cell mean within 0.05 of 0.847.

If R0 fails, **no candidate is read and no conclusion about the margin is
drawn**. The package is wrong, the run is reported as a failed reproduction,
and the component is found before anything else is asked. This is the only
gate whose failure stops the analysis rather than rejecting a candidate.

## Arms and measurements

Per candidate, on the 5 targeted tasks (niah_single_1, niah_multikey_2,
niah_multikey_3, ruler_fwe, ruler_qa_hotpot) at 16k/32k/64k, 10 samples per
cell, stats on -- the same shape as pre-registrations 1 and 4, so all three
tables are comparable cell for cell.

- Q  quality: the 15 cells, and the targeted mean over multikey_2, multikey_3
  and qa_hotpot.
- C  cost: the job's own VKSTATS fetch p50/p90/p99 and fallback rate.
- P  performance, for the chosen candidate only: the timed 4k->256k stream
  under the production protocol, and a stats-on twin.
- S  smoke, for the chosen candidate only: the head needle and the three
  saved 64k replay prompts.

## Gates

A candidate is adopted only if R0 passed and it passes all of these.

- **Q1** targeted mean >= 0.878, what max 2 measured in pre-registration 1.
- **Q2** 15-cell mean >= 0.920, and no single task mean below the max-0 arm by
  more than 0.05.
- **C1** fetch p50 <= 8192, half the capacity: the cost column must stop being
  censored.
- **C2** fallback rate <= 0.10.
- **P1** 256k timed stream within 0.02x of the max-0 arm on the same package.
  Stated against the package's own no-margin arm rather than an absolute
  speedup, because the package's baseline is what the tier-decode outcome
  moves.
- **P3** no short-context regression: at 8k, 16k and 32k, within 1% of the
  max-0 arm.
- **S1** every registered unit and kernel test passes.
- **S2** the head needle recovers and the three replay prompts answer as they
  do under the max-0 arm.

## Choice and adoption

Among candidates passing every gate, take the highest targeted mean; ties go
to the smaller margin, and to the max base over the lse base unless lse wins
on both targeted mean and fallback. The chosen candidate's flags become the
engine defaults in one commit, and the remaining quality jobs run under that
default.

If no candidate passes, the default keeps margin 0 and the paper reports the
margin as measured and rejected, with pre-registration 1's gain and this
package's cost side by side.

## What this does to the paper

If a margin is adopted, the trigger's description gains the margin and the
capacity as configured values, the deployment's RULER tables are re-measured
under the new default, and the recommended-configuration section's claim of
one quality knob becomes two and must say so. If none is, the appendix
paragraph on where the gap comes from is rewritten from this package's numbers
rather than pre-registration 1's censored ones, and the body keeps one knob.

## Not revised after the data

The thresholds, the candidate list, the task list, the choice rule, and R0's
stopping rule. The single open line is whether the tier-decode router is in the
package, which `td-stream-256k` decides before this freezes.
