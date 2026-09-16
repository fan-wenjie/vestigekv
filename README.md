# VestigeKV — experiments

> **Initialize the submodule before anything else:**
> ```bash
> git clone --recurse-submodules <this-repo>
> # in an already-cloned checkout:
> git submodule update --init --depth 1
> ```
> The only code dependency is `engine/` (an sglang fork, branch
> `vestigekv`, one commit, shallow-cloned). Without it, everything under
> `mexp/` raises `ImportError`.

## What it does

**VestigeKV** is a training-free KV-cache compression backend for NoPE-MLA
models (Kimi Linear family), shipped as the sglang attention backend
`vestigekv_mla`. It evicts by a query-independent signal the cache already
carries — the 64-dim un-roped sidecar branch, a vestige of RoPE that NoPE
training repurposes into a salience channel — keeping a global top-(S/32) of
rows attended while a **certified per-step recall tier** (rank-64 sketch +
conformal certificate) keeps every archived row reachable: no row is ever
dropped. The whole recall step runs as **seven fused Triton kernels inside
the decode CUDA graph** — one launch per step, no extra fixed cost, and the
graph never recaptures for VestigeKV reasons. Cache rows stay bit-exact
bf16; no quantization anywhere.

## Results at a glance

> Throughout this repo and the paper, **k = 1024 tokens** (KiB-style, not
> 1000): 4k prefill = 4096 tokens, 64k = 65536, 512k = 524288.

| figure | what it shows |
|---|---|
| ![decode latency vs S](results/fig_latency_curve.png) | bs=1 streaming decode: one request prefilled at 4k and decoded continuously to 512k; per-token latency vs. sequence length, vestigekv vs. dense (same tree, triton backend). Crossover ~28k; 1.28x at 256k, 1.46x at 496k. |
| ![throughput vs batch](results/fig_throughput.png) | Throughput: 64k prefill + 4k decode, batch 1..32; attention cannot batch while shared weights amortize, so the vestigekv advantage grows with batch (+12.4% at bs=12). |

*(Both figures are produced by `mexp/bench/plot_curve_jsonl.py` and
`plot_batch_throughput.py` from the frozen raw data in `results/`; the exact
collection commands are in the experiment steps below.)*

## The vestigekv startup transient, and why bs=1 sits near dense

Every request pays a small **startup transient**: tier-2 fits its conformal
certificate on the first 8--64 live decode queries, rebuilding the index as
that window grows. Engineering has reduced it (all bit-exact; see
[docs/INFRA_OPTIMIZATIONS.md](docs/INFRA_OPTIMIZATIONS.md)):

- calibrated rebuilds now **reuse the scan operands** across the window
  (they are independent of the calibration queries), collapsing the 5--9
  per-request archive passes to one;
- the build path's `kept_len` device readbacks are removed by a host-side
  mirror (they had stalled step 0 on the prefill tail);
- the in-graph kernels are clipped **per batch-size class** to the live
  pair count, so capacity placeholders are not launched.

What remains is the synchronous provisional index at the first decode step
(tier-2 must serve from step 1). The transient is per-request-bounded and
independent of context length: the first bucket of the latency curve runs
~0.2 ms/step above steady state (gone by ~112k).

**Why bs=1 throughput sits ~2--3% below dense at 64k+4k is NOT this
transient.** A 4k-token decode request runs ~33 s; the transient is tens of
ms, under 0.3% of it. The real reason is the **steady-state attention
share**: at 64k, attention is only ~7% of the decode step (MoE + linear
layers + the two-node pipeline dominate), so compressing it 32x saves at
most a fraction of a percent of the step, which the recall scan roughly
offsets -- vestigekv and dense are within launch-noise at bs=1. The
advantage appears where attention's share grows: with batch (the throughput
curve, +12.4% by bs=12, since attention cannot batch) and with context length
(the latency curve, 1.28x at 256k and 1.46x at 496k).

## Experiment steps

Reproducibility repository for **VestigeKV: The NoPE-MLA KV Cache Carries Its
Own Eviction Signal in a Vestigial Branch**. This is the *experiment project*:
pre-registrations, the paper, results (`out/*.json`), and the minimal
experiment-side tooling. The system under test lives in the **sglang
submodule** (`sglang`, attention backend `vestigekv_mla`); the
official sglang benchmark is the authoritative measurement path.

Every experiment here is **pre-registered**: its decision rule is frozen (dated,
in a `PREREGxx.md`) before its data exists. Verdicts are appended after. Each
experiment also has a step-by-step **skill file** under `skills/` (indexed
below) — invoke or read the skill to reproduce that one experiment.

## Code layout

| | |
|---|---|
| `engine` | **the only code dependency** (submodule): serving engine + VestigeKV backend |
| `mexp/` | active experiment scripts: data collection + what sglang lacks (see `mexp/README.md`) |
| `tools/` | health check, config check, serving launch harness |
| `archive/` | frozen scripts of published gates (mini-sglang era; not runnable — its dependency is retired) |

`mexp/_bootstrap.py` puts the submodule's `python/` on `sys.path`; a sibling
checkout at `/home/user/fft/sglang` takes over when present.

## Setup

```bash
# 0. Submodule (REQUIRED — see the box at the top)
git submodule update --init --depth 1

# 1. Python env (conda): torch (cu128), transformers, datasets, safetensors, fla
conda activate sglang        # or your env with the above

# 2. Kimi Linear checkpoint in the HF cache (moonshotai/Kimi-Linear-48B-A3B-Base)
#    downloaded once; the scripts read it from ~/.cache/huggingface

# 3. (serving experiments only) a GPU; dual-machine experiments need the second
#    box reachable — see mexp/README.md.

# 4. Weights daemon (step 0 of every serving experiment): warms the checkpoint
#    into RAM, holds it there, and prints the content-level WEIGHT-FP once.
#    Every launch script prints the same WEIGHT-FP line, so any run's log can
#    be checked against the daemon's declaration — both arms provably load
#    the same bytes. Keep it running for the whole session:
python tools/weight_daemon.py &
# expected: WEIGHT-FP 6503cda5dbec0bfc14f65b9e2f17c335c31a89ecc1141ce3a2f481fea6f97f56
```

## Baseline arm (IMPORTANT)

The dense baseline uses THE SAME checkout as the experiment arm — just select
the stock backend: `--attention-backend triton`. Do NOT check out upstream
sglang for the baseline: the VestigeKV diff outside its own package is four
inert stock-file edits, and running both arms from one tree is precisely the
demonstration that they do not perturb stock behavior. Both arms' launch logs
print the same WEIGHT-FP line (see step 0), attesting identical weights.

## How to run an experiment

Each experiment is a skill under `skills/`. Read `skills/<name>.md` for its
PREREG, exact command, and verdict, then run the command it lists. The
directory-queue runner is retired to `archive/` along with the pre-port probe
scripts and the mini-sglang-era gate scripts (frozen, not runnable).

## Pre-registration: 2026-09-11 reproduction run

Frozen on 2026-09-11, before this run's performance data exists. Honesty
note: the Base-arm quality numbers (GSM8K-Platinum, MAUVE) were collected
*before* this prereg was written and are reported as-is, outside its gate;
every other item below is frozen prior to collection. Deviations from the
frozen PR tree (the TP=2 `_q_heads` fix, the GSM8K-Platinum protocol) are
recorded in `runs/2026-09-11/ERRATA.md`.

**Arms.** Experiment arm: `--attention-backend vestigekv_mla`. Baseline arm:
**the same tree with `--attention-backend triton`** (see "Baseline arm" —
never upstream sglang). Both arms: single node, TP=2, `NCCL_P2P_DISABLE=1`,
`--disable-custom-all-reduce`, byte-identical weights (WEIGHT-FP attested by
the weights daemon and both launch logs). Any arm asymmetry beyond the
backend flag invalidates the comparison.

**Measurement.** Official sglang benchmark only:
`python -m sglang.benchmark.serving`. If the official benchmark cannot
measure a quantity, that experiment is NOT run (no custom benchmarks).

**Models.** Both `moonshotai/Kimi-Linear-48B-A3B-Base` and
`moonshotai/Kimi-Linear-48B-A3B-Instruct`, all four (model x arm) cells.

**Q1 — quality gate (frozen rule).** GSM8K-Platinum full set (n=1209,
64-shot; `mexp/quality/gsm8k_platinum.jsonl`) accuracy per arm, and MAUVE
against the reference corpus. Gate: vestigekv MAUVE >= dense MAUVE - 0.10,
and GSM8K-Platinum accuracy reported for both arms with the delta stated
plainly. Long-context decode-path fidelity is gated by (a) serving needle
(Q4 below) for retrieval and (b) long-context MAUVE (64k prefill,
`M7_T=65536 mexp/m7_mauve_serving.py`, same frozen MAUVE bar) for
generation-distribution fidelity. The greedy continuation-agreement gate
was retired: its frozen bar (median first-divergence >= 128) is
unreachable even for a pure attention-kernel swap on the SAME dense model
(dense flashinfer vs dense triton: median 37.5, worst 2 — ERRATA #20); it
measured numeric sensitivity, not compression fidelity. The retired serving
bpb could never see compression at all (ERRATA #18).
The performance experiments below run only if this gate passes.

**P1 — decode latency curve (frozen rule).** bs=1, 4k prefill, continuous
decode to 512k, per arm per model (`--random-input-len 4096
--random-output-len 520192 --max-concurrency 1 --output-details`; the
**server decode log is the per-token authority**: the client stream path
(main-process SSE, `--stream-interval 1`) has a per-token cost that grows
with stream length and saturates well below this testbed's generation
rate, so client ITL/E2E penalise the FASTER arm — ERRATA #19; the client
jsonl is kept only for the ITL distribution below the cap).
Decision rule: report per-token
latency vs. sequence length for both arms; the claim is confirmed if a
crossover exists below 100k and vestigekv is faster than dense at 272k and
496k (frozen reference points from the published run: crossover ~48k, 1.20x
at 272k, 1.39x at 496k — hardware-dependent, so the direction and ordering,
not the exact ratios, are the bar). Outcome 2026-09-12: PASS — crossover
~28k, 1.29x at 272k, 1.46x at 496k, both models.

**P2 — throughput vs. batch (frozen rule).** 64k prefill + 4k decode, batch
sweep 1, 2, 4, 8, 12, 16, per arm per model (`--random-input-len 65536
--random-output-len 4096 --max-concurrency $BS`). Decision rule: report
output throughput vs. batch for both arms; the claim is confirmed if
vestigekv's throughput advantage over dense is monotonically non-decreasing
across the sweep and positive by bs=12 (frozen reference: +8% at bs=12;
bs=1 within a few percent of dense either way is expected and NOT a
failure — see "The vestigekv startup transient"). Post-freeze extension
(user decision 2026-09-12): sweep extended to 24 and 32; the bs=32 dip on
BOTH arms is chunked prefill occupying the batch window, not a method
effect. Outcome 2026-09-12: PASS — advantage positive at every point,
+12.4% at bs=12 (Base), +16.1% (Instruct), peak ~1.14-1.21x.

## Experiment index

> Filled from the experiment catalogue. Each row links its skill file; the
> skill carries the exact reproduce command, the frozen bar, and the
> verdict. The dated pre-registration files themselves (frozen before their
> data, never edited after) ship with the paper's supplementary materials,
> not this repository; the prereg column names the file each skill answers
> to.

<!-- INDEX moved to EXPERIMENTS.md -->

The full catalogue -- ~40 pre-registered experiments across three phases
(core compressor, deployment-port validation, dual-machine serving), each
with its frozen bar and verdict -- lives in **[EXPERIMENTS.md](EXPERIMENTS.md)**.

## Reproduce the headline results by hand

All commands below run from the repository root after Setup (submodule,
env, checkpoint, weights daemon). The engine is NOT installed: every
command runs with `PYTHONPATH=$PWD/engine/python` so `import sglang`
resolves to the vendored fork in place. Deployment is a single node with
2 GPUs: tensor-parallel (TP=2), no pipeline split, no second machine
(the paper's two-node PP=2 layout is deployment-specific; the protocol
is unchanged).
NOTE: on this container platform CUDA P2P between the two GPUs hangs
(NCCL `P2P/CUMEM` never completes the first collective), so every launch
below exports `NCCL_P2P_DISABLE=1`; TP collectives then go over SHM.

```bash
# --- launch the vestigekv arm (single node, 2 GPUs, TP=2) ---------------
# SGLANG_VESTIGEKV_ACTIVATION_MIN_TOKENS=0 disables the short-context dense
# fallback (default 32768): every request takes the compressed path from
# token 0, so the numbers below measure the method itself at every length,
# including the short-context regime where its fixed cost is a disadvantage.
NCCL_P2P_DISABLE=1 PYTHONPATH=$PWD/engine/python \
SGLANG_VESTIGEKV_ACTIVATION_MIN_TOKENS=0 \
python -m sglang.launch_server \
  --model-path <kimi-linear-48b> --trust-remote-code \
  --attention-backend vestigekv_mla --tp-size 2 \
  --context-length 524288 --max-total-tokens 589824 \
  --cuda-graph-max-bs 2 --disable-custom-all-reduce --sampling-backend pytorch
# (radix cache stays ON here too; if the radix-on mamba bookkeeping leaves
# too little headroom for --max-total-tokens 589824, drop the explicit cap
# and let the pool auto-size -- bs=1 long-decode needs only ~520k tokens.)
# The dense baseline is the SAME command with --attention-backend triton
# (same tree; see "Baseline arm").

# --- metric 1: bs=1 latency curve (4k prefill -> 512k continuous decode) --
PYTHONPATH=$PWD/engine/python python -m sglang.benchmark.serving --backend sglang \
  --model <kimi-linear-48b> --num-prompts 1 --dataset-name random \
  --random-input-len 4096 --random-output-len 520192 --random-range-ratio 1 \
  --max-concurrency 1 --warmup-requests 0 --output-details \
  --output-file results/latency_stream_4k-512k_<arm>.jsonl
# keep the node-0 server log: it is the per-token authority above ~240k

# --- metric 2: throughput (64k prefill + 4k decode, batch sweep) ----------
# relaunch with --context-length 73728 --cuda-graph-max-bs 32 (same flags, incl. --disable-custom-all-reduce)
#   --max-running-requests 32
# (radix cache stays ON on both arms -- the production default. Cost on this
# hybrid model: ~16.3GB of mamba ssm_state for radix bookkeeping that random
# prompts never hit; the auto-sized pool still holds bs=32 at 0.89 usage.
# --cuda-graph-max-bs 32 + --max-running-requests 32 keep every decode step
# inside a CUDA graph -- no eager decode on either arm, matching production.
# The sweep below uses only exactly-captured batch sizes
# (bs=[1,2,4,8,12,16,24,32]), so no padding waste in either arm.
# No --max-total-tokens: let the pool auto-size so bs=16..32 are not
# capacity-bound. The vestigekv arm keeps
# SGLANG_VESTIGEKV_ACTIVATION_MIN_TOKENS=0.)
# Radix-on correctness of the vestigekv arm is covered by
# mexp/radix_probe.py (fresh vs full-prefix-hit outputs; vestigekv matches
# the triton backend's radix behavior exactly -- see ERRATA #13).
for BS in 1 2 4 8 12 16 24 32; do
  PYTHONPATH=$PWD/engine/python python -m sglang.benchmark.serving --backend sglang \
    --model <kimi-linear-48b> --num-prompts $((BS*2)) --dataset-name random \
    --random-input-len 65536 --random-output-len 4096 --random-range-ratio 1 \
    --max-concurrency $BS --warmup-requests 0 --output-details \
    --output-file results/throughput_64k+4k_bs${BS}_<arm>.jsonl
done

# --- quality: gsm8k-platinum long-shot + MAUVE (per arm) ------------------
# gsm8k uses GSM8K-Platinum (madrylab/gsm8k-platinum, arXiv:2502.03461):
# the cleaned revision of the FULL gsm8k test set (n=1209, label errors
# fixed, ill-posed questions dropped). The original gsm8k test set has
# known label noise, so the legacy n=800 subset protocol is retired.
# relaunch with --context-length 16384 --cuda-graph-max-bs 4 (same flags, incl. --disable-custom-all-reduce)
#   --max-running-requests 4 --disable-radix-cache
# (the vestigekv arm keeps SGLANG_VESTIGEKV_ACTIVATION_MIN_TOKENS=0, so
# gsm8k/MAUVE measure the compressed path, not the dense fallback).
# Quality is the REPRODUCIBILITY line, unlike the perf line above: radix
# OFF (cache hits shift chunk boundaries and flip greedy near-ties --
# ERRATA #13) and the client runs gsm8k serially (run_quality.sh uses
# --parallel 1), so every request recomputes the 64-shot prefix with a
# batch pinned to 1 and runs are bitwise repeatable on the same hardware.
# The perf metrics keep radix ON (production default) -- different goal.
bash mexp/quality/run_quality.sh vestigekv
bash mexp/quality/run_quality.sh dense
bash mexp/quality/run_quality.sh score     # after both arms; needs the GPU

# --- Kimi Linear RULER on 2x RTX PRO 6000 (engine branch vestigekv, the Kimi line) ---------
# Engine branches: `vestigekv` = upstream main + the VestigeKV backend + the general fixes
# (capacity fence / flags / int32 / eager step, stats, per-request calibration basis,
# rank flag, debug dumps, state-job unlink); it serves Kimi Linear only. The GLM-5.3 port
# lives on `vestigekv-glm53` (geometry, side pool, model hooks, key ring) with its box
# branch `vestigekv-pro6000x2` and the experiment branch `vestigekv-dsa-index`.
# Arms: baseline = dense MLA (--attention-backend triton, same tree), vestigekv = the
# vestigekv_mla backend at every flag default. Quality line as for GLM: CUDA graph ON
# (--cuda-graph-max-bs-decode 4), --disable-radix-cache, --max-running-requests 4,
# --max-mamba-cache-size 4, --chunked-prefill-size 4096, --context-length 73728,
# --random-seed 0, --sampling-backend pytorch, TP=2, NCCL_P2P_DISABLE=1; clients serial and
# greedy, RULER data generation and lm-eval seeded 0; model moonshotai/Kimi-Linear-48B-A3B-Instruct.
# Jobs (mexp/kimi/queue.jsonl, same runner as the GLM line with --line kimi):
#   ruler-baseline, ruler-vestigekv -> 13 RULER tasks x {4k,8k,16k,32k,64k}, 10 samples/cell
#   stats-vestigekv-ruler-64k, stats-vestigekv-stream-128k -> the 13 tasks at 64k (10/cell) and a
#       4k-prefill 126976-token decode on the vestigekv arm with SGLANG_DEBUG_VESTIGEKV_STATS=1:
#       the server log's VKSTATS lines carry the recall fetch per scan (p50/p90/p99 rows) and the
#       fallback rate (overflowed scans / scans); separate jobs because the bookkeeping syncs.
#   margin-{0,1,2,3} -> smoke sweep of --vestigekv-recall-margin (engine commit "recall rows within
#       a margin of the kept max") on the tasks the max-recall criterion loses (niah_single_1,
#       niah_multikey_2/3, ruler_fwe, ruler_qa_hotpot) at 16k/32k/64k, 10/cell, stats on: each
#       job's scores plus its VKSTATS fetch p50/p90/p99 and fallback rate give the
#       quality-vs-fetch trade-off; the chosen margin then gets a plain (stats-off) latency check.
#       Decision rule and degradation gates are pre-registered in
#       mexp/kimi/prereg_recall_margin.md (frozen before any sweep result was read).
#   margin-lse-{2.3,4.6} -> the same sweep with --vestigekv-recall-threshold lse (margin taken
#       from the kept log-sum-exp: a dropped row holds <= e^-margin of the whole kept mass;
#       2.3 and 4.6 are mass fractions 0.1 and 0.01), the control for the max-based margin.
#   stream-baseline-256k, stream-vestigekv-256k -> metric 1 on this box (bs=1, 4k prefill,
#       258048-token decode, stats off): the dense vs margin-0 gain at 64k/128k/256k that the
#       pre-registered latency gate (give-back rate <= 0.25 of that gain) is measured against;
#       python mexp/glm53/stream_curve.py --line kimi --output-len 258048 (results/kimi/latency_stream_*).
#   stream-vestigekv-256k-m2, stats-vestigekv-stream-256k-m2 -> exploratory (outside the frozen
#       rule): the 4k -> 256k stream at margin 2, timed and with stats, so the long-decode cost
#       of a margin (gates 3 and 4 of the pre-registration) is on record even though the
#       short-answer gate 2 failed every max-base candidate in the sweep.
#   smoke-prefillcal-needle, smoke-prefillcal-replay-64k -> engineering smoke of
#       --enable-vestigekv-prefill-calibration (engine commit "calibrate the recall index during
#       prefill"): the head needle and three saved 64k prompts with stats on; the server log's
#       VKCAL lines show whether the prefill-time builds install before decode (async=True at a
#       seq below the prompt length) and what they fire. Not a pre-registered measurement.
#   longbench2-baseline, longbench2-vestigekv -> LongBench v2 on the 300 questions whose context is
#       <= 120k Kimi tokens (mexp/kimi/longbench2/: lm-eval task longbench2_kimi_120k, official
#       zero-shot prompt + answer regex, subset_120k.json from make_subset.py), raw prompts, serial,
#       greedy, 128 tokens, CTX=135168 with one running request; results/kimi/longbench2/
#       results_<arm>.json (accuracy overall and per domain/length/difficulty, parse rate) and
#       samples_<arm>.json. longbench2-vestigekv-stats -> the same with stats on: must reproduce
#       the stats-off answers exactly and gives the fetch/fallback figures on real documents.
#   ruler-vestigekv-nofallback -> the 13 tasks x 4k-64k, 10/cell, stats on, with
#       --disable-vestigekv-recall-overflow-fallback: an overflowed scan attends its first 4096
#       fired rows instead of the full row set, so this arm cannot escape to dense attention; the
#       control for "the RULER quality is the fallback's, not the method's".
#   fb-<task> (13 jobs) -> one stats-on RULER job per task (5 lengths, 10/cell) on a fresh server
#       each (the VK_JOB env entry only breaks the runner's server-reuse signature), so each server
#       log's last VKSTATS line is that task's fallback rate and fetch p50/p90/p99.
#   ruler-baseline-n50, ruler-vestigekv-n50 -> the 13 tasks x 4k-64k at 50 samples/cell (one sample
#       = 0.02), the numbers that replace the n=10 tables; compare with
#       python mexp/glm53/compare_ruler.py --out results/kimi/ruler --n 50.
#       Gates and interpretation rules for these three groups are pre-registered in
#       mexp/kimi/prereg3_realdoc_and_fallback.md (frozen before any of them ran).
#   pc-A{1,2,3}-{ruler,stats64k,stream256k-stats,stream256k} -> pre-registration 2
#       (mexp/kimi/prereg2_prefill_calibration.md): arms A1 = --enable-vestigekv-prefill-calibration,
#       A2 = A1 + --vestigekv-recall-margin 2, A3 = A1 + margin 4.6 with --vestigekv-recall-threshold
#       lse; per arm the 13-task 4k-64k RULER (10/cell), the 64k stats run, and the stats-on and
#       timed 4k->256k streams, read against the A0 (default) and dense records already on file.
#       The default chosen by its rule is committed before ruler-vestigekv-long and the n=50 pair run.
#   ruler-baseline-long, ruler-vestigekv-long -> the 13 tasks x {128k,256k,512k,1M}, 5 samples/cell,
#       with CTX=1064960 (1M + 16k; SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 because the
#       model's derived context is exactly 1048576), --max-running-requests 2, two mamba slots,
#       --cuda-graph-max-bs-decode 2 (lm-eval's RULER generator is seeded 0 here too)
# weight-cache daemon (one process per GPU holding the TP=2 bf16 shards; a server launched
# while its ready files name live pids loads the weights over CUDA IPC -- common.sh adds
# --weight-cache-mode client, and its GPU guard then only refuses a second server;
# WEIGHT_CACHE=off forces disk loading). NOT USABLE for Kimi Linear on this tree: the IPC
# load itself takes 0.7 s, but the client-mode server dies with an illegal memory access
# at CUDA-graph capture (KDA decode kernel) and, with graphs off, in the MoE fused gate:
# sglang's IpcModelLoader replaces parameters by object and rebuilds only Mamba-style
# conv1d views, while Kimi Linear's KDA layers, MoE gate and absorbed MLA weights keep
# construction-time references (results/kimi/weight_daemon.log and the smoke logs of
# 2026-09-16 07:44-07:48). Fixing it means per-model reference rebuilding in sglang
# (kimi_linear.py: qkv_conv1d/A_log/dt_bias, the gate's correction bias, w_kc/w_vc);
# the queue runs with disk loading (~60 s per server launch) until then.
nohup bash mexp/kimi/weight_daemon.sh > results/kimi/weight_daemon.log 2>&1 &
bash mexp/kimi/weight_daemon.sh status     # or stop
python mexp/glm53/queue_runner.py --line kimi
# health monitor (read-only; one line per 30 min in results/kimi/health.log: runner/server/
# daemon liveness, current job, client progress or STALL, new tracebacks in the newest
# server log counted per file, GPU, disk):
bash mexp/health_check.sh kimi 1800 &
# regression bisection over engine commits on the needle tasks (serves each commit from a
# throwaway worktree via ENGINE=<dir>, counts garbage answers; results/kimi/bisect.log):
#   bash mexp/kimi/bisect_niah.sh <engine-commit> [n=10] [tasks]
# by hand:
#   bash mexp/kimi/baseline.sh ; python mexp/glm53/run_ruler.py --arm baseline --n 10 \
#     --model moonshotai/Kimi-Linear-48B-A3B-Instruct --out results/kimi/ruler
#   bash mexp/kimi/vestigekv.sh ; (same client with --arm vestigekv)
# results: results/kimi/ruler/results_<arm>_n10_4096-8192-16384-32768-65536.json;
# two-arm table: python mexp/glm53/compare_ruler.py --out results/kimi/ruler

# --- GLM-5.3-Flash-NVFP4 on 2x RTX PRO 6000 Blackwell (SM120; branch vestigekv-pro6000x2) ---
# Arms: baseline = the model as shipped (DSA: indexer top-k 2048 + KPool 4:1, Triton DSA
# kernels); vestigekv = DSA off + vestigekv_mla over the dense-MLA substrate, fp8 side pool
# (the DSA index-cache format), every --vestigekv-* flag at its default (capacity 4096,
# overflow fallback on, activation threshold 0, sketch rank 64 -- a queue job can raise
# the rank with "server_args": ["--vestigekv-index-rank", "256"]). "baseline" means DSA on this model
# (Dense MLA on Kimi Linear); mexp/glm53/dense_mla.sh is the substrate ablation, not a baseline.
# Quality/RULER line, fixed in mexp/glm53/common.sh: CUDA graph ON (--cuda-graph-max-bs-decode 4),
# --disable-radix-cache, --max-running-requests 4, --max-mamba-cache-size 4,
# --chunked-prefill-size 1024 (2048 and 4096 OOM the DSA prefill: weights take 88 GB/GPU, ~4.5 GB is left for the pool plus prefill working memory), --mem-fraction-static 0.955 (pool ~100k tokens: 0.95 gave 61k, below the 64k RULER prompts), --context-length 73728,
# --random-seed 0, --language-model-only (vision tower skipped), --sampling-backend pytorch;
# clients are serial and greedy, RULER data generation and lm-eval seeded 0.
# Jobs are a JSONL queue (mexp/glm53/queue.jsonl: one job per line = arm + server env +
# client + args); the runner launches each job's server, runs the client, keeps the
# server across same-config jobs, and records results/glm53/queue_state.jsonl. Edit the
# queue while it runs: it is re-read before every job. Registered queue (in order):
#   ruler-baseline, ruler-vestigekv   -> 13 RULER tasks x {4k,8k,16k,32k,64k}, 10 samples/cell
#   stream-baseline-128k, stream-vestigekv-128k -> metric 1 (bs=1, 4k prefill, 126976-token
#       decode; per-token latency curve, 64k point included) with CTX=135168, one mamba slot,
#       --mem-fraction-static 0.96, --cuda-graph-max-bs-decode 1
#   stats-vestigekv-stream-128k, stats-vestigekv-ruler-64k -> the same two workloads on the
#       vestigekv arm with SGLANG_DEBUG_VESTIGEKV_STATS=1 (separate jobs: the bookkeeping syncs
#       every step, so it never runs on a timed arm). The server log's VKSTATS lines (every 50
#       decode steps, cumulative) carry fetched rows per scan p50/p90/p99 and
#       fallback = overflow scans / scans; the stream job's successive lines are the
#       context-length curve, the RULER-64k job the value at 64k prompts.
#   replay-vestigekv-r64, -r128, -r256 -> the same three saved 64k RULER prompts
#       (niah_single_2, qa_squad, cwe from the baseline samples) replayed on the vestigekv arm
#       at sketch ranks 64, 128 and 256 with stats on; the per-job server log's VKCAL lines
#       carry need-vs-fire per calibrated build (mexp/glm53/replay_prompts.py). The r128 job
#       also sets SGLANG_DEBUG_VESTIGEKV_DUMP_DIR=results/glm53/caldump: one snapshot per
#       (slot, layer) of the rows, keep set and calibration queries the index was fitted on;
#       python mexp/glm53/cert_offline.py refits sketch bases and ranks on them offline
#       (certificate zp, fire counts vs need, fallback fraction) without re-serving.
#       They run before the two stats jobs: the rank decision gates the stats config.
#   dsadump-vestigekv-{64k,8k,4k} -> DSA-indexer certificate study (engine branch
#       vestigekv-dsa-index / serving vestigekv-dsa-index-pro6000x2, checked out in engine/ for
#       these jobs): saved RULER prompts (niah_single_2, cwe, fwe, qa_squad; 2-3 samples per
#       task) replayed with --ignore-eos (64 tokens, so every request reaches a calibrated
#       build), GRAPH=0 (the indexer stash copies to the host every step, which a captured
#       graph cannot) and SGLANG_DEBUG_VESTIGEKV_DUMP_DIR=results/glm53/dsadump, whose snapshots also
#       carry the indexer keys / KPool gates of every token and the indexer query heads of the
#       calibration queries; python mexp/glm53/dsa_cert_offline.py scores the indexer-based
#       certificate (mass coverage, conformal z) against the content-sketch one on them.
#   ruler-dense-mla-short -> fwe, cwe, qa_squad, niah_single_1 at 4k and 8k (10/cell) on the
#       dense_mla substrate (DSA off, no VestigeKV): RULER "4k" prompts are ~3.9k tokens, below
#       one 4096 close block, so the vestigekv arm attends them dense; this separates the
#       substrate's effect (a DSA-trained model run dense) from VestigeKV's at 4k/8k.
#   memtrace-vestigekv-64k -> 13 tasks x 3 samples at 64k on the vestigekv arm with
#       SGLANG_DEBUG_VESTIGEKV_MEM_DIR=results/glm53/memtrace and stats OFF: the server log's
#       VKMEM lines (allocated / reserved at every request's first prefill chunk) and one
#       allocator snapshot per five requests (mem_req<N>.pickle; diff two with
#       python mexp/glm53/memdiff.py). Background: ruler-vestigekv OOMed at its 27th 64k
#       prompt twice while the same prompts with stats on served 119; the stats syncs
#       change what the allocator sees, so the trace must not sync.
python mexp/glm53/queue_runner.py
# by hand (same commands the runner issues):
#   bash mexp/glm53/baseline.sh ; python mexp/glm53/run_ruler.py --arm baseline --n 10
#   bash mexp/glm53/vestigekv.sh ; python mexp/glm53/run_ruler.py --arm vestigekv --n 10
#   CTX=135168 MAX_REQS=1 MAMBA_SLOTS=1 MEM_FRAC=0.96 CHUNK=512 GRAPH_BS=1 bash mexp/glm53/<arm>.sh
#   PYTHONPATH=$PWD/engine/python python -m sglang.benchmark.serving --backend sglang \
#     --model nvidia/GLM-5.3-Flash-NVFP4 --num-prompts 1 --dataset-name random \
#     --random-input-len 4096 --random-output-len 126976 --random-range-ratio 1 \
#     --max-concurrency 1 --warmup-requests 0 --output-details \
#     --output-file results/glm53/latency_stream_4k-126976_<arm>.jsonl
# results: results/glm53/ruler/results_<arm>_n10_4096-8192-16384-32768-65536.json (+ samples),
#          results/glm53/latency_stream_4k-126976_<arm>.jsonl, server_<arm>_<job>.log per job.
#          RECORD NOTE: the vestigekv stream file on disk is the stats-on run (a stats job
#          used to write the same name and overwrote the timed one; the runner now suffixes
#          _stats). The timed vestigekv curve, read before that overwrite (fixed engine,
#          per-token median over the 4096 tokens ending at each point): 8k 11.535, 16k 11.613,
#          32k 11.731, 64k 11.885, 128k 12.207 ms vs baseline 11.265/11.283/11.301/11.340/11.398
#          (ratio 0.977 -> 0.934; means 11.908 vs 11.342 ms/token).
# two-arm RULER table (task x length, a/b, means): python mexp/glm53/compare_ruler.py
# per-token latency vs context from the stream jobs: python mexp/glm53/stream_curve.py
# GSM8K-Platinum (64-shot, n=1209, serial) on either arm: relaunch with CTX=16384, then
#   PYTHONPATH=$PWD/engine/python python -m sglang.test.few_shot_gsm8k --num-shots 64 \
#     --num-questions 1209 --data-path mexp/quality/gsm8k_platinum.jsonl --parallel 1 --port 30000
# Head-needle probe (10.6k tokens, code at position 0): python mexp/glm53/needle.py

# --- quality: long-context MAUVE (64k prefill; the decode-path fidelity gate)
# 16 fineweb-edu contexts of 65536 tokens (compression genuinely engaged),
# 256-token generations, per-context sampling_seed shared across arms; same
# frozen MAUVE bar (vk >= dense - 0.10). One-command pipeline (4 cells):
bash mexp/quality/run_mauve64k.sh
# (manual per-arm variant: M7_T=65536 python mexp/m7_mauve_serving.py gen
#  results/quality_mauve64k_ctx.json results/quality_mauve64k_<arm>.json,
#  then ... score with the same ctx file.)

# --- quality: greedy continuation agreement — RETIRED (ERRATA #20) --------
# The frozen bar (median first-divergence >= 128, min >= 8) is unreachable
# even for a pure attention-kernel swap on the SAME dense model (dense
# flashinfer vs triton: median 37.5, worst 2): the gate measured numeric
# sensitivity, not compression fidelity. Data kept for the record:
#   results/quality_continuation_{vestigekv,dense}[_instruct].json
#   results/quality_continuation_dense_flashinfer.json  (kernel-swap control)

# --- quality: needle-in-a-haystack on the serving path (per arm) ----------
# Serving counterpart of the harness needle: fixed-seed passcode spliced at
# a random depth into fineweb-edu filler, compressed at prefill BEFORE the
# query exists, greedy answer judged by exact string. Same prompts for both
# arms; frozen bar: vk intact rate >= dense intact rate (>= 8 trials or
# ABORT). Default L=131072 n=8; L=524288 n=4 is the long-context
# confirmation (prefill-bound, ~13 min/trial).
python mexp/quality/needle_serving.py gen 131072 8 results/quality_needle_vestigekv.json
# ... relaunch the dense arm, then:
python mexp/quality/needle_serving.py gen 131072 8 results/quality_needle_dense.json
python mexp/quality/needle_serving.py compare \
  results/quality_needle_vestigekv.json results/quality_needle_dense.json \
  results/quality_needle_verdict.json

# --- Q5: Chinese-corpus needle (serving, both arms) ---------------------------
# Same protocol as Q4 but haystack AND needle are Chinese: 三国演义 filler
# (mexp/quality/sanguoyanyi.txt, ~492k tokens) + a modern-Chinese fact
# ("{place}的接头暗号是「{code}」。") at random depth. Arabic digits never
# occur in the classical text, so the code is an unambiguous foreign token
# sequence; answers are accepted in Arabic or Chinese numerals. Same frozen
# bar. Scheduled AFTER all other experiments (exploratory, not a gate).
python mexp/quality/needle_chinese.py gen 131072 8 results/quality_needle_zh_vestigekv.json
# ... relaunch the dense arm, then:
python mexp/quality/needle_chinese.py gen 131072 8 results/quality_needle_zh_dense.json
python mexp/quality/needle_chinese.py compare \
  results/quality_needle_zh_vestigekv.json results/quality_needle_zh_dense.json \
  results/quality_needle_zh_verdict.json

# --- figures --------------------------------------------------------------
python mexp/bench/plot_curve_jsonl.py \
  --arm dense:results/latency_stream_4k-512k_dense.jsonl \
  --arm vestigekv:results/latency_stream_4k-512k_vestigekv.jsonl \
  --srv dense:results/latency_stream_serverlog_dense.log \
  --srv vestigekv:results/latency_stream_serverlog_vestigekv.log \
  --clip 8192 --ylim 3.5:8 --out results/fig_latency_curve.png
python mexp/bench/plot_batch_throughput.py \
  --arm dense:results/throughput_64k+4k_bs1_dense.jsonl,... \
  --arm vestigekv:results/throughput_64k+4k_bs1_vestigekv.jsonl,... \
  --out results/fig_throughput.png
```

### Algorithm claims (single machine, HF-forward harness)

The needle-retrieval, observed-attention baselines (H2O / SnapKV /
StreamingLLM), the RoPE-collapse control, the sketch-rank and branch-width
ablations, and teacher-forced bits-per-byte are measured by the HF-forward
harness (`harness/`, no two-node serving). Each paper table names its
`--ops` and its PREREG in the matching skill file (see `EXPERIMENTS.md`).
The harness needs the pinned 4.x stack (ENV_PINS.txt); the serving env runs
transformers 5.x, which breaks the 4.x-era modeling code, so run the harness
with the isolated target dir on PYTHONPATH (checkpoints as plain dirs, see
`harness/extract.py`):
`pip install --target=$HOME/.cache/vestigekv/tf457 transformers==4.57.1 "huggingface-hub<1.0" "kernels<=0.9,>=0.6.1"`
`export PYTHONPATH=$HOME/.cache/vestigekv/tf457`

```bash
# needle + observed-attention baselines at 8k (Table: baselines / tab:main)
python harness/e2e.py --arch kimi --seq-len 8192 --n-docs 0 \
  --needle-trials 24 --gpu-expert-layers 18 --seed 0 --ops baselines \
  --out results/harness_baselines_8192.json
# RoPE-collapse negative control (Table: rope) -- same op set on a RoPE-MLA
python harness/e2e.py --arch dsv2 --seq-len 8192 --n-docs 0 \
  --needle-trials 24 --gpu-expert-layers 18 --seed 0 --ops twotier \
  --out results/harness_rope_8192.json
# sketch-rank sweep (Table: ablation) and bits-per-byte (Table: bpb) select
# their own --ops/--rhos; see the PREREG in each skill file for the exact line.

# post-training survival (Appendix: SFT): the SAME command on the Instruct
# checkpoint, with the Base weights run as an in-session control. Both arms
# use byte-identical modelling code (the Base copy) so the only variable is
# the weights; register the paths as `kimi_instruct` / `kimi_base_mirror`
# in harness/extract.py.
for ARCH in kimi_base_mirror kimi_instruct; do
  python harness/e2e.py --arch $ARCH --seq-len 8192 --n-docs 0 \
    --needle-trials 24 --gpu-expert-layers 18 --seed 0 \
    --ops twotier,imp_only,recent --rhos 32,128 \
    --out results/harness_needle_8192_${ARCH#kimi_}.json
done
# static anatomy (branch/content row norms), weights only, no forward pass:
python harness/dissect.py <model-dir> results/harness_anatomy_<arm>.json

# --- tab:tier Wilson n-boost (PREREG34, skills/prereg34-tier-wilson-boost.md)
# Pure measurement, no bar; pooling/containment rules frozen in the skill.
# (a) 8k +recall row and the survival pair: seed 0 extended 24 -> 64 trials.
#     The first 24 trials MUST bit-reproduce the n=24 files above (the
#     containment check script is in the skill file); ABORT otherwise.
for ARCH in kimi_base_mirror kimi_instruct; do
  python harness/e2e.py --arch $ARCH --seq-len 8192 --n-docs 0 \
    --needle-trials 64 --gpu-expert-layers 18 --seed 0 \
    --ops twotier,imp_only,recent --rhos 32,128 \
    --out results/harness_needle64_8192_${ARCH#kimi_}.json
done
# (b) 32k digk64 cells: seeds 11+12 extended 12 -> 16 trials each (pooled 32).
for SEED in 11 12; do
  python harness/e2e.py --arch kimi --seq-len 32768 --n-docs 0 \
    --needle-trials 16 --gpu-expert-layers 18 --seed $SEED \
    --rhos 32,64,128 --ops digk64,recent \
    --out results/harness_tier_32768_seed${SEED}.json
done
# (c) 65k digk64 cells: historically single seed 11 n=12; re-run seed 11
#     (covers both historical rho splits) and add seed 12 (pooled 24).
for SEED in 11 12; do
  python harness/e2e.py --arch kimi --seq-len 65536 --n-docs 0 \
    --needle-trials 12 --gpu-expert-layers 18 --seed $SEED \
    --rhos 32,64,128 --ops digk64,recent \
    --out results/harness_tier_65536_seed${SEED}.json
done
```
