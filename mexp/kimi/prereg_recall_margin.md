# Pre-registration: choosing `--vestigekv-recall-margin` on Kimi Linear

Frozen 2026-09-16 04:20 (this commit's timestamp), before any `margin-*` sweep
result was read. The sweep itself was queued earlier (mexp/kimi/queue.jsonl:
margin-0/1/2/3 with the max base, margin-lse-2.3/4.6 with the lse base; tasks
niah_single_1, niah_multikey_2, niah_multikey_3, ruler_fwe, ruler_qa_hotpot at
16k/32k/64k, 10 samples per cell, stats on). One sample moves a cell by 0.10.

Reference points, already measured on the same engine (results/kimi):
- vestigekv at margin 0, 13 tasks x 4k-64k: mean 0.923 (baseline dense MLA 0.941).
- calibrated-regime cost at margin 0 (stats-stream-128k): fetch p50/p90/p99 =
  0/0/42 rows, fallback 0.000; short-answer regime (stats-ruler-64k): fetch p50
  1023, fallback 0.36 (provisional index, 8-step calibration window).

## Gates (a candidate must pass all of them)

1. No degradation: 15-cell mean >= margin-0's 15-cell mean - 0.02, and no
   task mean (3 cells, 30 samples) below margin-0's by more than 0.05.
2. Short-answer cost (the sweep job's own VKSTATS): fallback rate <= 1.5 x
   margin-0's fallback rate, and fetch p50 <= 2048 rows (half the capacity).
3. Long-decode cost (confirmation run at the chosen value, stats-stream-128k):
   cumulative fallback <= 0.02 and fetch p90 <= 512 rows.
4. Latency (confirmation run, stats off, 4k -> 128k stream): mean ms/token
   <= margin-0's mean + 3%.

## Choice rule

Among candidates passing gate 1 and 2, take the one with the highest mean over
the three targeted tasks (niah_multikey_2, niah_multikey_3, ruler_qa_hotpot),
but only if that mean beats margin-0's by at least 0.05; otherwise the default
stays 0 and the sweep is reported as a negative result. Ties: the smaller
margin; the max base over the lse base unless lse wins on both the targeted
mean and fetch p50. Gates 3 and 4 are checked after the choice; a failure
falls back to the next candidate in the same order, then to 0.

## Confirmation

The chosen value becomes the flag default (engine commit), the 13-task 4k-64k
vestigekv RULER is rerun with it (gate: 65-cell mean >= 0.923 - 0.01, and the
targeted tasks not below their sweep values by more than 0.05), and the
128k-1M jobs run with the same default. Thresholds above are not revised after
results are seen; anything that fails is reported as such.
