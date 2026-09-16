# Pre-registration 4 (DRAFT): the recall margin under a repaired cost structure, and the new default

Status: draft. **To be frozen** (committed) once the performance line has
finished, with its measured numbers filled into "Cost structure" below and
nothing else changed. No arm of this pre-registration runs before the freeze.

## Why the question is reopened

Pre-registration 1 ruled the recall margin negative. Its quality result was
unambiguous and is not re-litigated:

| setting | multikey_2 | multikey_3 | qa_hotpot | fwe | single_1 | 15-cell mean | targeted mean | fetch p50 | fallback |
|---|---|---|---|---|---|---|---|---|---|
| max, 0 | 1.000 | 0.733 | 0.567 | 0.933 | 1.0 | 0.847 | 0.767 | 2137 | 0.448 |
| max, 2 | 0.967 | 1.000 | 0.667 | 0.967 | 1.0 | 0.920 | 0.878 | 4096 | 0.747 |
| max, 3 | 1.000 | 1.000 | 0.667 | 0.967 | 1.0 | 0.927 | 0.889 | 4096 | 0.846 |
| lse, 4.6 | 1.000 | 1.000 | 0.733 | 0.967 | 1.0 | 0.940 | 0.911 | 4096 | 0.912 |

It failed on cost, and four things measured since change what that cost is.

1. **The cost was the fence, not the fetch.** An overflowing lane packs the
   request's whole row set: at 256k that is 262144 row ids copied by a single
   program (264 us/step, about 4 GB/s) plus a dense attention (stage1 106 ->
   301 us). The gather now grid-strides (`PACK_GRID_CAP` programs per lane),
   which is a pure implementation fix and changes no row set.
2. **Every candidate's fetch p50 was 4096, the capacity itself.** The
   measurement was censored: what the margin actually fires is unknown, and
   the fallback rate it reports is the rate of hitting a buffer, not of
   needing the whole cache. The fetch buffer is int32 and per (layer, lane):
   16384 rows is 64 KB.
3. **The provisional index dominated the short-answer regime.** A 14-step
   answer spends most of its steps on the identity-basis index with z clamped
   to Z_MAX. Prefill calibration (pre-registration 2) removes that phase, and
   its builds now start at the first closed block rather than at 16k prompt
   tokens.
4. **The index was fitted once and never refitted.** A 4k->256k stream ran 14
   builds in 258k steps, two per layer; a fit made at 8k context over-fires at
   256k. `--vestigekv-rebuild-overflow-fraction` refits a layer whose scan
   keeps overflowing, off the token path.

## Cost structure (filled at the freeze, from the performance line)

- 256k timed stream, server-side ms/token: dense _, origin/vestigekv _,
  current default _, grid-strided pack _, + rebuild trigger _, + attended
  splits _.
- stats-on 256k stream at the chosen package: fetch p50/p90/p99 _,
  cumulative fallback _.
- gather microbenchmark: contiguous _ GB/s against scattered _ GB/s at each
  arm's best grid (decides whether a contiguous kept arena is also in the
  package).

## The package under test

Everything below is already implemented on engine branch `vestigekv-perf`,
each behind its own flag and each with tests:

- grid-strided CSR gather (no flag: an implementation fix, no row set moves);
- `--enable-vestigekv-prefill-calibration`;
- `--vestigekv-rebuild-overflow-fraction 0.05`;
- `--vestigekv-recall-capacity 16384`;
- `--vestigekv-recall-margin` / `--vestigekv-recall-threshold`, the variable
  under test;
- `--enable-vestigekv-attended-splits` is **not** in the package unless it
  passes its own performance gate; it changes no row set, so it cannot affect
  the quality arms either way.

Margin candidates, fixed here from pre-registration 1's table and not
revised after data: **max 2**, **max 3**, **lse 4.6**, against **max 0** (the
same package with no margin) as the reference.

## Arms and measurements

Per candidate, on the 5 targeted tasks (niah_single_1, niah_multikey_2,
niah_multikey_3, ruler_fwe, ruler_qa_hotpot) at 16k/32k/64k, 10 samples per
cell, stats on: the same shape as pre-registration 1's sweep, so the tables
are comparable cell for cell.

- Q  quality: the 15 cells, and the targeted mean over multikey_2,
  multikey_3, qa_hotpot.
- C  cost: the job's own VKSTATS fetch p50/p90/p99 and fallback rate.
- P  performance, for the chosen candidate only: the timed 4k->256k stream
  (server-side metric) and a stats-on twin.
- S  smoke, for the chosen candidate only: the head needle and the three
  saved 64k replay prompts.

## Gates

A candidate is adopted only if it passes all of these.

- **Q1** targeted mean >= 0.878 (what max 2 measured under the old cost
  structure): the package must keep the margin's quality gain, not merely
  its cost.
- **Q2** 15-cell mean >= 0.920, and no single task mean below the max-0
  reference by more than 0.05.
- **C1** fetch p50 <= 8192 (half the new capacity): the measurement must stop
  being censored, so the number reported is what the margin fires.
- **C2** fallback rate <= 0.10.
- **P1** 256k timed stream: speedup over dense >= 1.25x (origin/vestigekv
  measures 1.276x on this box and protocol, today's default 1.184x).
- **P2** stats-on 256k stream: cumulative fallback <= 0.02, fetch p90 <=
  8192.
- **P3** no short-context regression: at 8k, 16k and 32k the candidate is
  within 1% of today's default.
- **S1** every registered unit and kernel test passes (per commit, already).
- **S2** the head needle recovers and the three replay prompts answer as they
  do today.

## Choice and adoption

Among candidates passing every gate, take the highest targeted mean; ties go
to the smaller margin, and to the max base over the lse base unless lse wins
on both targeted mean and fallback. The chosen candidate's flags become the
engine defaults in one commit, and **the quality line then runs entirely under
that default**: LongBench v2 both arms, RULER 4k-64k at n=50 both arms, the
128k-1M pair, and the pre-registration 2 arms re-based on it (an amendment
recorded there, not a silent rerun).

If no candidate passes, the default keeps margin 0 and whichever performance
fixes passed their own gates, and the quality line runs under that. A failed
gate is reported as failed.

## What this does to the paper

If a margin is adopted, the trigger's description gains the margin and the
capacity as configured values, the deployment's RULER tables are re-measured
under the new default, and the appendix paragraph "Where the gap comes from,
and what a margin buys" is rewritten from the new data rather than edited.
The recommended-configuration section's claim of one quality knob becomes two
(target and margin) and must say so. The future-work paragraph on clustered
recall stays future work: clustering prunes the scan and changes no fired
row, so it cannot move any number in this pre-registration.

## Not revised after the data

The thresholds, the candidate list, the task list and the choice rule.
