# RULER v2: measured, deliberately NOT in the paper

Run 2026-09-24 on the sglang `vestigekv` branch (engine 18465b53ca), both arms
started by `mexp/kimi/{baseline,vestigekv}.sh`, NeMo-Skills acting as a client
only. Procedure and every pitfall: README.md, "RULER v2 (Kimi line)".

## Why it is not in the paper

Not a quality judgement on the numbers. Two reasons, decided 2026-09-24:

1. v2 is weighted towards multi-key and multi-value needle retrieval, which
   the v1 suite already covers through `niah_multikey_1/2/3`. It is a harder
   restatement of ground the paper already reports, not a new dimension.
2. The deadline is close. Introducing a second benchmark now moves reviewer
   attention off the results the paper is built on, and v2 numbers do not
   compare with any v1 number already collected, so it cannot be folded into
   an existing table -- it would need its own section.

Nothing here is referenced by a paper macro. Do not merge these into a v1
table or into a `.tex` generated from one.

## What was measured, n=20 per task, 11 tasks

| length | dense baseline | VestigeKV r=32 | VestigeKV r=256 |
|--------|----------------|----------------|-----------------|
| 8192   | 71.5           | 66.9  (-4.7)   | not run         |
| 32768  | 71.7           | 44.2  (-27.5)  | 52.2  (-19.5)   |

At 32768 every one of the 11 tasks regressed at r=32. The dense baseline is
flat across the two lengths (71.5 -> 71.7), so this is VestigeKV degrading
with context, not the tasks getting harder.

## The finding worth carrying forward

Raising the sketch rank from 32 to 256 lifts its offline coverage of the true
softmax mass from 0.786 to 0.998 at 32k and costs 0.5% of the decode step, yet
recovers only 8.0 of the 27.5 points, 29%.

So coverage explains under a third of the gap, and every offline conclusion
that rests on it -- the truncation rule, the bin count, the quality of the
fitted basis -- is bounded by that. The remaining 71% is unattributed.

Raw results live beside this file, one directory per arm and length.
