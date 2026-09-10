# Infrastructure optimizations (serving-path engineering)

These are engineering optimizations to the sglang serving port that reduce
the per-request startup transient. They are **bit-exact** with respect to
the model's output (each is gated by a parity test) — they change *when*
and *how* work is scheduled, never *what* is computed. They are distinct
from the algorithm (which rows are kept/recalled); the algorithm numbers
are unchanged.

The per-request startup transient comes from tier-2 calibration: the first
8–64 decode steps of every request fit the conformal certificate on live
queries, and the index is (re)built several times as the calibration
window grows. Two optimizations cut that cost.

## 1. Operand reuse across the calibration ladder

**Problem.** Fitting the certificate needs a growing window of live decode
queries, so the calibrated index is rebuilt 5–9 times per request as the
window doubles (8→16→32→64). Each rebuild recomputed the full scan
operands — the rank-r basis `V`, and the archive projections `csk`, `rho`,
`side` over every archived row — an O(archive) pass that scales with
context length (~10 ms at 8k, more at longer S).

**Key observation.** The scan operands are pure functions of *(prefix rows,
basis, tier-1 keep mask)* and are **independent of the calibration
queries**. Tier-1 selection is frozen between block closes, so inside a
calibration window every rebuild sees the *identical* archive row-set
(verified: the archive count is constant across all builds of a window).

**Optimization.** A calibrated rebuild adopts the previous tier's operands
instead of recomputing them, refitting only the small calibration scalars
(`zp`, `thr_g`) and regathering the growing kept set. The calibration
ladder's 5–9 full archive passes collapse to **one**.

**Correctness.** Bit-exact by construction; the guard is "same close
epoch", owned by the caller, and a changed row-set is refused. Both the
bit-identity and the guard are driven in `TestOperandReuse`
(`test_vestigekv_scan_kernel.py`).

## 3. Per-batch-class grid sizing (see the PR)

The in-graph kernels launch a grid sized for the worst-case context and
batch; `update()` fills live pairs contiguously from slot 0, so the grid is
clipped per batch-size class to the live pair count and capacity-tail
placeholders are never launched. Measured +0.32→+0.16 ms/step at
`--cuda-graph-max-bs 16` serving bs=1. Bit-exact (placeholders produced
nothing).

## Scope and honest limits

These reduce the **startup transient**, which is a per-request fixed cost
bounded by the calibration window and independent of context length. They
do **not** change steady-state decode. In particular, at a 64k prefill with
a 4k decode window, the startup transient is under 0.3% of the request
(a 4k-token decode is ~33 s; the transient is tens of ms), so it is not the
reason bs=1 throughput sits near the dense baseline there — that is the
steady-state attention share (~7% of the step at 64k), where compression
has little to save. The startup optimizations matter most for short
requests and for keeping the long-context latency curve clean at its start.
