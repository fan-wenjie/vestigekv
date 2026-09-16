# Pre-registration 2: prefill calibration and the recall margin on Kimi Linear

Frozen 2026-09-16 07:35 (this commit's timestamp), after the exploratory
margin-2 256k streams were read (their numbers are the reference points
below) and BEFORE any run listed under "Arms" started. The first
pre-registration (prereg_recall_margin.md) ended negative because its
short-answer cost gate measured a regime dominated by the provisional index.

## What changes
- `--enable-vestigekv-prefill-calibration` (engine commit 5b7d0cb): calibrated
  builds from absorbed prompt queries during prefill, installed before the
  first decode step; the provisional index then serves only requests whose
  prompt gave no build (prefix below one close block, or a build still in
  flight). Smoke (smoke-prefillcal-needle, smoke-prefillcal-replay-64k, not a
  measurement): on a 64k prompt the calibrated build installs at seq 16384
  during prefill (VKCAL async=True, proxy=False, zp 3.0 against the
  provisional 8.0); needle and replay answers intact.
- `--vestigekv-recall-margin` / `--vestigekv-recall-threshold` as candidates,
  now evaluated on calibrated steps.

## Reference points already on record (results/kimi, engine 5b7d0cb or earlier same-code)
- A0 quality: ruler-vestigekv, 13 tasks x 4k-64k, 10/cell: mean 0.923 (dense 0.941).
- A0 short-answer cost (stats-vestigekv-ruler-64k): fetch p50/p90/p99 =
  1023/4096/4096, fallback 0.360.
- A0 long-decode cost (stats-vestigekv-stream-128k): fetch p50/p90/p99 =
  0/0/42, fallback 0.000.
- Streams, median ITL (ms) over the 4096 tokens ending at 64k / 128k / 256k:
  dense 4.337 / 4.764 / 12.703; A0 (margin 0) 4.166 / 4.177 / 7.769;
  margin 2 without prefill calibration 4.230 / 4.509 / 7.433, i.e. give-back
  rate r = 0.37 / 0.57 / -0.07: the margin's cost at 64k-128k exceeds the
  0.25 gate even on a long stream, although its cumulative fallback stays
  0.0055 and fetch p90 <= 242 (stats-vestigekv-stream-256k-m2). A margin
  therefore has to earn its place through gate G4 below; prefill
  calibration alone changes no per-step cost after calibration and is
  expected to pass G4 trivially.

## Arms (all vestigekv, Kimi Linear Instruct, quality-line flags; queue ids pc-A<k>-*)
A0 prefill-cal off, margin 0 (today's default)  -- the reference, from the records above
A1 prefill-cal on, margin 0
A2 prefill-cal on, margin 2, max base           (from the first sweep, fixed here)
A3 prefill-cal on, margin 4.6, lse base         (fixed here)

## Measurements (one queue job each per arm)
Q  pc-A<k>-ruler: 13-task RULER 4k-64k, 10/cell, stats off (as ruler-vestigekv);
   the sweep's five targeted tasks at 16k-64k are read from the same run.
C1 pc-A<k>-stats64k: the 13 tasks at 64k, stats on (as stats-vestigekv-ruler-64k):
   fetch p50/p90/p99 and fallback rate.
C2 pc-A<k>-stream256k-stats: stats-on 4k->256k stream: fetch p90/p99 and
   fallback, cumulative and per 64k window.
L  pc-A<k>-stream256k: timed 4k->256k stream, read at 64k/128k/256k (median
   ITL over the 4096 tokens ending there), against the dense and A0 streams
   already on record (stream-baseline-256k, stream-vestigekv-256k).

## Gates
G1 quality: 65-cell mean >= A0's mean - 0.01; no task mean below A0's by more
   than 0.05. (A1 must not lose anything: it changes only who calibrates.)
G2 short-answer cost (C1): fallback rate <= 0.10 and fetch p50 <= 1024 for A1;
   for A2/A3 the same thresholds (a margin must not reopen the provisional
   problem the prefill calibration closes).
G3 long-decode cost (C2): cumulative fallback <= 0.02 and fetch p90 <= 512.
G4 latency (L): give-back rate r = (t_arm - t_A0) / (t_dense - t_A0) <= 0.25
   at every point where A0 beats dense, else t_arm <= 1.01 x t_dense.

## Choice rule
A1 becomes the default if it passes G1-G4 (its targeted-task mean is expected
to be >= A0's; it is not required to gain). Among A2/A3 passing G1-G4, the
one with the higher targeted-task mean (multikey_2, multikey_3, qa_hotpot)
becomes the default only if it beats A1's targeted mean by >= 0.05; ties go
to A2 (max base). Anything failing is reported as such; thresholds are not
revised after results.

## Confirmation
The chosen default is committed (engine flag defaults), and the runs queued
after these arms use it: ruler-vestigekv-long (128k-1M) and the n=50 4k-64k
pair of pre-registration 3; the health of the long-context runs (fallback,
fetch) is read from the same stats machinery. If no arm passes, the default
stays A0 and those runs proceed with it.
