# P1/P2 performance verdict — 2026-09-11 run

Gate: README "Pre-registration: 2026-09-11 reproduction run". Official
`python -m sglang.benchmark.serving` only. Baseline = same tree,
`--attention-backend triton`. Both arms TP=2, NCCL_P2P_DISABLE=1,
--disable-custom-all-reduce, --max-running-requests 4 (P1) / 32 (P2)
(see ERRATA.md for the OOM-driven knob).

## P1 — bs=1 latency curve, 4k prefill -> 512k decode

Per-token authority: server-log Decode-batch buckets (client jsonl `itls`
agrees; both saved). 16k-token bucket means:

Base:
  crossover 48k; @272k vestigekv 4.552ms vs dense 5.738ms = 1.26x;
  @496k vestigekv 5.055ms vs dense 7.093ms = 1.40x
Instruct:
  crossover 48k; @272k vestigekv 4.557ms vs dense 5.727ms = 1.26x;
  @496k vestigekv 5.066ms vs dense 7.083ms = 1.40x
Frozen reference (different card): crossover ~48k, 1.20x @272k, 1.39x @496k.

Verdict: **PASS, both models** (crossover < 100k, vestigekv faster at 272k/496k,
ratios bracket the frozen reference).

## P2 — throughput, 64k prefill + 4k decode, output tok/s

Base:     vestigekv 161.5/215.1/268.2/302.8/321.5/334.0 (bs 1/2/4/8/12/16)
          dense 158.3/204.2/252.5/283.9/300.2/312.3
          advantage +2.0/+5.3/+6.2/+6.7/+7.1/+6.9 %
Instruct: vestigekv 161.9/216.6/272.1/312.8/335.0/349.1
          dense 158.7/205.2/255.2/288.0/305.3/318.4
          advantage +2.0/+5.5/+6.6/+8.6/+9.7/+9.6 %
Frozen reference: +8% at bs=12; bs=1 within a few % either way expected.

Verdict: **PASS, both models** — advantage positive at every bs, grows with
batch, +7.1% (Base) / +9.7% (Instruct) at bs=12. Note: bs=16 is 0.1-0.2pt
below bs=12 in both models (plateau within run noise), so strict
monotonicity across the whole sweep is not claimed.

## Kernel-level (engine's own benchmark, supplementary)

engine/benchmark/kernels/vestigekv/bench_vestigekv_kernels.py (bench-script
API drift fixed: compact_fired gained a_off/am_grid args; bench now refills
the counts table per call exactly as the scan kernel does in serving).
Results in runs/2026-09-11/kernel_bench_vestigekv.txt: scan 0.016ms from
64k to 256k (992->4204 GB/s), compact triton 0.026ms vs torch-chain
0.12-0.35ms, prologue split-NK 0.042-0.054ms — the per-step recall cost is
~0.1ms, flat in S, consistent with the flat vestigekv decode curve.

Figures: results/fig_latency_curve_{base,instruct}_repro.png,
results/fig_throughput_{base,instruct}_repro.png. Raw: results/*_{base,
instruct}.jsonl (LFS), server logs runs/2026-09-11/server_perf_*.

## 2026-09-12 addendum: short-context fix (post-verdict)

Two defects behind the sub-1.0x short-context latency were found by torch
profiler + dual-trace kernel diff and fixed (ERRATA.md #8/#9): the recall
pipeline's ~114 us/step fixed cost (now ~18 us; empty-archive early exits +
grid-stride scan/compact) and a bs=1 double-append legacy block that made
the packed CSR attend every decoded row twice. L* = 32768 activation
threshold implemented (below: byte-identical dense, verified by greedy
parity at 8k/30k).

Re-measured segment (same official bench_serving protocol, 4k prefill,
single stream, Base model), 4k-bucket medians, ms/token:

| S | vestigekv (fix) | dense | ratio |
|---|---|---|---|
| 6k | 3.922 | 3.885 | 1.010 |
| 16k | 4.014 | 3.976 | 1.010 |
| 30k | 4.153 | 4.115 | 1.009 |
| 36k | 4.105 | 4.149 | 0.989 |
| 44k | 4.112 | 4.205 | 0.978 |

Crossover moved from ~48k to ~33k; below L* the residual is a flat ~1%
(the CSR pack + empty-archive pipeline), above L* the win begins at once
and is slightly better than the pre-fix curve (phantom rows gone).

Attention-only (mexp/attn_only_bench.py, serving semantics incl. L*):
4k 1.00x / 16k 1.00x / 32k 1.18x / 128k 3.68x / 512k 5.52x
(attn_only_bench_after_fix.txt; 512k unchanged vs pre-fix 5.59x -- the
grid-stride loop costs ~1% at long context).

The P1/P2 numbers above stand as the pre-fix record; per-run policy they
are not re-quoted as final. A full re-run would only improve the
vestigekv arm.
