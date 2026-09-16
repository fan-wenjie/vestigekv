# Pre-registration 3: real-document quality, RULER at n=50, and what the dense fallback carries

Frozen 2026-09-16 07:15 (this commit's timestamp), before any of the jobs below
produced a result. Engine: `vestigekv` branch at 5b7d0cb (every `--vestigekv-*`
flag at its default: margin 0, max threshold, prefill calibration off). Jobs are
in mexp/kimi/queue.jsonl and described in README.md.

Motivation. (i) The 4k-64k RULER comparison (0.923 vs 0.941) rests on 10
samples per cell (one sample = 0.10). (ii) RULER is synthetic; the paper needs
one real-document benchmark to show that the multi-key / counting / two-hop
gap does not translate into a large loss on realistic tasks. (iii) The
short-answer RULER regime falls back to dense attention on 0.36 of scans
(stats-ruler-64k: fetch p50 1023, p90 at the 4096 capacity; the provisional
index serves most of a 14-step answer), so a reviewer can ask whether the
RULER quality is the method's or the fallback's. The three questions get one
job each, with the interpretation rules fixed here.

## A. LongBench v2, <= 120k-token subset (longbench2-baseline, longbench2-vestigekv)

Task: mexp/kimi/longbench2/ (lm-eval task `longbench2_kimi_120k`): the 300 of
503 LongBench v2 questions whose context is at most 122880 Kimi Linear tokens
(mexp/kimi/longbench2/subset_120k.json; all six domains: single-doc QA 121,
multi-doc QA 86, dialogue history 39, in-context learning 35, code 15,
structured 4). Official zero-shot prompt and answer regex; raw
/v1/completions, serial, greedy, 128 tokens; CTX 135168, one running request.

Metric: accuracy over the 300 questions, per arm; both arms answer the same
prompts, so the paired difference is what is gated.

- Gate A1 (no large degradation): acc_vk >= acc_dense - 0.05 (15 questions;
  ~1.3 SE of the paired difference at acc ~0.4).
- Report A2: accuracy per domain, per official length bucket (short/medium)
  and per difficulty; the number of questions the arms answer differently in
  each direction; parse rate per arm (a parse-rate gap above 0.05 is reported
  as a formatting effect, not a quality effect).
- Determinism check A3 (longbench2-vestigekv-stats): the same job with
  SGLANG_DEBUG_VESTIGEKV_STATS=1 must reproduce the stats-off accuracy
  exactly (question by question); its VKSTATS line gives the fetch
  percentiles and fallback rate on real documents.

## B. RULER 4k-64k at 50 samples per cell (ruler-baseline-n50, ruler-vestigekv-n50)

Same protocol as the n=10 runs (13 tasks x 5 lengths, seeds 0, stats off); one
sample now moves a cell by 0.02.

- Gate B1: 65-cell mean acc_vk >= acc_dense - 0.03.
- Confirmation rule B2 for the paper's task-family claim: the task means
  (250 samples each) of niah_single_1/2/3, niah_multiquery, niah_multivalue
  and ruler_vt must each be within 0.02 of dense; otherwise the claim "every
  single-needle, multi-query, multi-value and variable-tracking cell matches
  dense" is withdrawn and the per-task table is reported as measured.
- The n=50 numbers replace the n=10 numbers in the paper's Table 1 and
  RULER table (make_ruler_numbers.py --n 50); the n=10 files stay archived.

## C. What the fallback carries (ruler-vestigekv-nofallback, fb-<task>)

C1, the control: the 13 tasks x 4k-64k at n=10 with
`--disable-vestigekv-recall-overflow-fallback` (an overflowed scan attends
the first 4096 fired rows instead of the request's full row set; nothing
else changes). Interpretation rule, fixed now:

- If the no-fallback 65-cell mean is >= the default arm's mean - 0.02, the
  dense fallback is not what carries the RULER quality: the certified fetch
  does, and the fallback is a bounded-cost safety net. Cells where the
  control drops below the default arm by more than 0.10 are listed as the
  cells the safety net actually rescues.
- If it is lower by more than 0.02, the paper says so: on short-answer
  prompts the provisional-index regime relies on the fallback, and the
  prefill-calibration follow-up (prereg 2) is the fix under test.

C2, attribution: one stats-on job per task (fb-<task>, 5 lengths, n=10, a
fresh server per job so its VKSTATS counters start at zero) gives the
fallback rate and fetch p50/p90 per task. Reported as a column next to the
RULER scores; no gate. Expected pattern, stated before the data: fallback is
highest on the tasks with many near-tied distractors (multi-key, multi-value,
cwe) and lowest on single-needle tasks; the pattern, not the level, is what
the criterion argument predicts.

## Outcome C1 (2026-09-16): the quality does depend on the fence

65 cells, 13 tasks x 4k-64k at n=10, both vestigekv arms on the same engine
and flags except `--disable-vestigekv-recall-overflow-fallback`:

| arm | 65-cell mean |
|---|---|
| dense | 0.9413 |
| vestigekv, default | 0.9226 |
| vestigekv, no fallback | 0.8661 |

The paired delta is **-0.0565**, past the -0.02 line, so C1 takes its second
branch: on these prompts the provisional-index regime relies on the fallback,
and the fallback is not a bounded-cost safety net that never matters. The
cells it rescues, by the rule's >0.10 criterion, are almost exactly the
multi-key family:

| task | length | default | no fallback |
|---|---|---|---|
| niah_multikey_3 | 64k | 0.90 | 0.20 |
| niah_multikey_3 | 32k | 0.80 | 0.20 |
| niah_multikey_3 | 16k | 0.90 | 0.50 |
| niah_multikey_2 | 64k | 0.80 | 0.40 |
| niah_multikey_2 | 32k | 1.00 | 0.60 |
| niah_multikey_2 | 16k | 0.90 | 0.60 |
| ruler_qa_squad | 64k | 0.47 | 0.28 |
| ruler_qa_squad | 16k | 0.57 | 0.44 |

This corrects a performance-side reading, not just a quality one. The audit
had noted that the fence fires on 0.27% of scans and recaptures nothing, and
took from that the suggestion that disarming it is free. It is free in
ms/token and it is not free in accuracy: the 0.27% of scans that fire are
concentrated on the prompts the multi-key tasks are made of. Any design that
removes the fence's *cost* has to keep the fence's *rows* -- which is the
tier-decode router's premise and the reason the debug fence stub is not an
implementation.

## What is not revised after the data

The thresholds above, the subset file, the prompt template and the job list.
A failed gate is reported as failed; the paper's text is changed to match
the outcome, not the other way round.

## Amendment 1 (2026-09-16): how A is scored, and why

Three protocol deviations from section A, all of them about how an answer is
read off the model and none about which questions are asked or which flags the
engine runs. The subset, the prompt, the arms and every gate above are
unchanged.

**A-i. The answer is scored, not generated.** Section A says "greedy, 128
tokens" with the official answer regex. Measured on the baseline arm: the
model reasons before answering and does not reach the answer line inside 128
tokens, so 1% of 300 answers parsed and both arms sat at the floor (acc
0.003). Raising the budget makes the benchmark a reasoning-length measurement
rather than a retrieval one. The task now uses upstream lm-eval's own
LongBench v2 shape (`lm_eval/tasks/longbench2/_longbench_common_yaml`):
`output_type: multiple_choice` over the four one-token choices A/B/C/D, with
the prompt ending at "Answer:". Gate A1 and report A2 are unchanged except
that the parse rate is no longer defined -- every question yields a choice, so
the parse-rate clause of A2 is withdrawn rather than reported.

**A-ii. The four choices are scored in one request, not four.** lm-eval's
`local-completions` scores a choice with `echo=True, logprobs=1,
max_tokens=1`, which sglang serves with `logprob_start_len=0`; that value
clamps the radix prefix match to zero
(`schedule_batch.py::_compute_max_prefix_len`), so the four requests of a
question re-prefill the same shared context four times and prefix caching
cannot help by construction -- measured as `#cached-token: 0` on all 562
prefill batches of a radix-on run. For a single-token continuation lm-eval's
score is exactly the next-token logprob at one position, so one
`max_tokens=1, logprobs=20` request answers all four. The measurement is the
same quantity; only the number of prefills changes. Verified before adoption
on the shortest subset documents: the two scorings agree on the argmax and on
the logprob values themselves. A choice absent from the top-20 is ranked
below every present choice; the count of questions where that happens is
reported.

**A-iii. Both arms keep the prefix cache disabled.** It follows from A-ii:
with one request per question there is no shared prefix between consecutive
requests, so `RADIX=on` buys nothing and the arms stay on the quality line's
default protocol.

Both arms and the determinism twin A3 run under A-i through A-iii. Nothing
measured under the generation shape is carried forward.
