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
- **What the tier path and the affine arm each cost in quality**: 0 of 650
  answers and exactly +0.0000 for the tier row read; 40 of 650 and +0.0056 for
  the affine reordering, against a 40-of-650 run-to-run floor. Worth a sentence
  where the paper claims bit-exactness against dense.
- The fallback rate is **per rank**: TP0 0.0027 and TP1 0.0548 on the same 256k
  job. Any figure quoted from one rank understates it 20x.
