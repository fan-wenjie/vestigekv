# What the paper needs when the data is in

Frozen here so the edit is mechanical rather than remembered. Nothing in the
paper changes until every arm below has reported; the owner asked for one pass.

## Ready now

**Latency curve.** Server-side ms/token, production protocol, against the dense
arm of `results/latency_stream_serverlog_dense_instruct.log` (whose 64k / 128k /
256k reproduce this line's own dense baseline to three decimals, so the two
campaigns are comparable):

| context | dense | paper's backend | delivered | delivered speedup |
|---|---|---|---|---|
| 64k | 4.360 | 4.156 | 4.125 | 1.057 |
| 128k | 4.786 | 4.224 | 4.184 | 1.144 |
| 256k | 5.590 | 4.387 | 4.364 | **1.281** |
| 508k | 7.120 | 4.785 | 4.651 | **1.531** |

The body says 1.28x at 256k and 1.46x at 496k. Both are superseded by the two
bold figures, measured on the delivered implementation
(`td-stream-512k`, 4k prefill decoded to 512k). The figure
`fig_latency_curve.png` is regenerated from the same run.

**Kernel count.** Done 2026-09-17: six fused kernels, not seven, and stage 1 is
a fork of the stock kernel rather than untouched.

**Still open from that pass**: the two int64-CSR sentences (body near the
deployment notes, and the appendix paragraph). Only the eager path builds a CSR
now, so both have to name the path they describe.

## Waiting on the queue

- **RULER table and Table 1** -- `ruler-vestigekv-n50` / `ruler-baseline-n50` on
  the delivered tree; regenerate with `make_ruler_numbers.py --n 50`. The n=10
  numbers stay archived (prereg 3, gate B1/B2).
- **LongBench v2** -- `lb2-vestigekv` (running) against the finished
  `lb2-baseline` (0.333 over 300 questions); macros via
  `make_lb2_numbers.py`. Gate A1 is a paired difference, so report
  `lbKDelta`, `lbKGain`, `lbKLoss`, not two accuracies.
- **Per-task fallback column** -- the 13 `fb-<task>` jobs; seven still to run.
  These stay on `engine/`, which the text must say (prereg 3 amendment 2).

## Statements that changed and need the text to follow

- The speedup is a **long-decode** number. At matched 64k context the stream
  falls back on 0.00014 of scans and RULER on 0.36, and the control that holds
  context, generated length and request count fixed while changing only the text
  (`stats-stream-64k-x130`) falls back on 0.41 -- more than RULER. So the
  regime, not the question, is what decides, and the text must not let a reader
  carry 1.28x across to a short-answer workload.
  **Resolved 2026-09-17**: the regime means the DECODE COUNT, not how the
  context was built. `stats-stream-64kprefill-long` holds the 64k prefill and
  the text fixed and moves only the step count, 14 to 4096; fallback goes 0.41
  to 0.0027/0.0044. So the qualifier is "long-decode" and nothing more -- no
  second clause about prefill versus decoded-up-to is needed.
- **What the tier path and the affine arm each cost in quality**: 0 of 650
  answers and exactly +0.0000 for the tier row read; 40 of 650 and +0.0056 for
  the affine reordering, against a 40-of-650 run-to-run floor. Worth a sentence
  where the paper claims bit-exactness against dense.
- **Every `fallback=` figure measured on `engine-fused` between 09-17 06:17
  (`ff60fab`) and 09-17 10:5x (`69eefcb`) is wrong, low by the prologue's
  call multiplicity (~3x).** `steps` and `scan_calls` counted once per
  *entry* to `_decode_prologue`, which runs once per step but is called
  twice-plus; `overflow` is a device counter incremented once, so the rate
  divided. Affected done jobs: `current-stats-64k-eager`, `lb2-vestigekv-stats`,
  `trunc-prefix-nofb`, `trunc-spread-nofb`, and the first `idxstate-*` pair.
  **Not affected**: every accuracy number from those jobs (a stats counter does
  not reach the answers); every per-call ratio (`fetched/call`, `kept/call`,
  `attended_frac`), whose numerator and denominator inflate together; and
  everything measured on `engine/`, which counts in `_ingraph_host_step`.
  The dump now reports `calls/step` so the multiplicity is measured, not
  inferred. Anything quoted from an affected job must be re-measured.
- **The fallback rate is a decode-length number, not a text-type number.** The
  2x2 (random tokens / a real novel) x (14 / 4096 decode steps) at a fixed 64k
  context: 0.407, 0.0023, 0.483, 0.0032. Length moves it 151x; text moves it
  1.6x and the most natural text is the WORST. Any sentence ordering fallback
  by how natural the text is must go, including the one that reads a high
  RULER rate as the method working at the edge of its applicability.
- **The speedup cannot be measured on RULER, and must not be claimed there.**
  A 14-token answer never amortises the index: 81% of RULER's scans run on a
  tier that has not finished calibrating, and it falls back on 0.305 with every
  knob at its default. There is no speedup at that length to lose, so "what
  does the multi-key fence cost in latency" is not a question RULER can answer
  -- asking it there measures the wrong regime. The speedup is a LONG-GENERATION
  number and the method's prior is fitted to natural text; both belong in the
  same sentence as 1.28x, every time it appears.
- **Where the fence costs anything is exactly where there was nothing to gain.**
  The fence fires on lanes that emit many archived rows, which is the
  provisional regime: 81% of RULER's scans and 0.6% of a long decode's. On the
  long stream the fresh bucket emits 2.3 rows per scan (p50 0, p90 <= 2), far
  under a fence of 256, so the prediction is that fence 256 leaves the 256k
  stream's 4.364 ms/token essentially unchanged and adds under a point of
  fallback. `fence-mk256-stream-256k` and `-stats-256k` measure it; until they
  report, the paper states the fence's cost only as the RULER fallback rate.
- The fallback rate is **per rank**: TP0 0.0027 and TP1 0.0548 on the same 256k
  job. Any figure quoted from one rank understates it 20x.
