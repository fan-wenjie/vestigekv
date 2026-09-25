# Pre-registration 4 (DRAFT): the recall margin under a repaired cost structure, and the new default

Status: **frozen 2026-09-16 15:40**, with the performance line's measured
numbers filled into "Cost structure" below and nothing else changed. No arm of
this pre-registration ran before this commit.

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

## Cost structure (measured, 2026-09-16)

256k timed streams, server-side ms/token at 256k, and the speedup over dense:

| protocol | dense | origin | current default | grid-strided pack | + rebuild trigger | + attended splits | fence disarmed |
|---|---|---|---|---|---|---|---|
| radix off | 5.564 | 4.360 (1.276) | 4.701 (1.184) | 4.395 (1.266) | 4.518 (1.232) | 4.517 (1.232) | -- |
| radix on | 5.576 | 4.346 (1.283) | 4.771 (1.169) | 4.879 (1.143) | -- | -- | 4.343 (1.284) |

Recall at 256k, stats on, margin 0, over 258050 steps: production protocol
fetch p50/p90/p99 = 0/180/2115, overflow 4846, fallback 0.00268; quality
protocol 0/0/29, overflow 8, fallback 0.00000. The overflow count is
back-loaded: 148 of the 4846 had happened by step 84750.

**Correction (2026-09-16, later the same day): those are TP0's counters, not
the job's.** Each rank keeps its own, and they differ by 20x: TP0 overflow
4846 (rate 0.0027), TP1 overflow 99016 (rate 0.0548). The aggregate is about
0.029, not 0.0027. This is the same asymmetry the profile found as rank skew
(TP1's pack 264 us against TP0's 13.5 us): the ranks hold different heads, so
they fire different row sets and overflow independently. Every statement below
that reads a fallback rate off one rank understates it, and the reading that
the fence "fires on 0.27% of scans" is TP0's alone.

Three readings the arms below inherit.

- **The branch costs nothing except the fence.** Disarming it reaches
  1.284x where origin measures 1.283x, so the 443 commits between them are
  free; arming it costs 12% while firing on 0.27% of scans, which is a cost
  of arming, not of firing, and is not yet located.
- **The certificate goes stale with context.** Overflow is concentrated in
  the second half of the stream, which is what a fit made at 8k context
  serving 256k looks like, and is the defect the rebuild trigger and the
  prefill calibration both address.
- **Two structural options are closed.** A contiguous arena for the attended
  tier buys nothing (contiguous, every-32 and random index lists are within
  1% at every grid, 2255 against 2232 GB/s), and sizing the KV splits from
  the attended rows is a 2.8% regression at 256k.

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

## Outcome (2026-09-16 17:00): every candidate fails, and the package is why

The four arms on the frozen package (prefill calibration, rebuild trigger
0.05, capacity 16384), 5 targeted tasks at 16k/32k/64k, 10 per cell:

| arm | mk2 | mk3 | hotpot | fwe | s1 | 15-cell | targeted | fetch p50/p90/p99 | fallback |
|---|---|---|---|---|---|---|---|---|---|
| pre-reg 1, margin 0 | 1.000 | 0.733 | 0.567 | 0.933 | 1.0 | 0.847 | 0.767 | 2137/-/- | 0.448 |
| m0 | 0.200 | 0.067 | 0.567 | 0.789 | 1.0 | 0.524 | 0.278 | 0/3/656 | 0.000 |
| m2 | 0.200 | 0.067 | 0.567 | 0.778 | 1.0 | 0.522 | 0.278 | 1/1444/3968 | 0.000 |
| m3 | 0.200 | 0.100 | 0.633 | 0.878 | 1.0 | 0.562 | 0.311 | 11/3646/3968 | 0.000 |
| lse 4.6 | 0.200 | 0.133 | 0.633 | 0.900 | 1.0 | 0.573 | 0.322 | 80/3146/3968 | 0.000 |

Q1 (targeted >= 0.878) and Q2 (15-cell >= 0.920) fail by a wide margin on all
four, so no candidate is adopted and the default keeps margin 0.

The verdict is not about the margin. m0 and m2 have the same targeted mean
(0.278), and m0 is the package with no margin at all yet already 0.489 below
the pre-registration 1 arm it should reproduce. The package broke recall, and
the component is prefill calibration -- see pre-registration 2's outcome,
where it alone costs 0.135 of the 65-cell mean and 0.58 on both multi-key
tasks. The margin question is therefore still open and has to be asked again
on a package that does not include it; this pre-registration's thresholds are
not reused, because its reference arm is not a valid reference.
