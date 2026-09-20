# VestigeKV — experiments

> **Initialize the submodule before anything else:**
> ```bash
> git clone --recurse-submodules <this-repo>
> # in an already-cloned checkout:
> git submodule update --init --depth 1
> ```
> The only code dependency is `engine/` (an sglang fork, branch
> `vestigekv`). Without it, everything under `mexp/` raises `ImportError`.
> The branch is no longer the single commit it began as; the one-commit form
> is preserved as `vestigekv-pre-rebase`. For a reviewer without this
> repository, `tools/make_supplement.py` emits the same code as upstream
> sglang at a pinned revision plus one patch.

## What it does

**VestigeKV** is a training-free KV-cache compression backend for NoPE-MLA
models (Kimi Linear family), shipped as the sglang attention backend
`vestigekv_mla`. It evicts by a query-independent signal the cache already
carries — the 64-dim un-roped sidecar branch, a vestige of RoPE that NoPE
training repurposes into a salience channel — keeping a global top-(S/32) of
rows attended while a **certified per-step recall tier** (rank-64 sketch +
conformal certificate) keeps every archived row reachable: no row is ever
dropped. The whole recall step runs as **six fused Triton kernels inside
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
| `docs/` | the third tier of the paper's evidence --- see below |
| `results/` | two figures and `results.zip`; run records for every number (`results/README.md`) |

### `docs/` --- evidence graded out of the paper

The paper grades its evidence into three tiers so a reader can tell what the
argument rests on. The body (9 pages) carries the core performance data and
whatever is needed to understand the algorithm and its implementation. The
appendix (7 pages) defends the claims a reviewer is most likely to contest ---
the proofs, why NoPE is required, the full RULER grid, what the baselines
actually do here, the measurement gates, where the gap is, and the cost
accounting. Everything true but not load-bearing for either lands here.

| path | what it is |
|---|---|
| `docs/multikey-findings.md` | working notes on the multi-key gap: the two mechanisms and their split, what was ruled out (and two earlier arguments that do not hold), the open per-step question and the free instrument queued for it, and three detection checks with their measured cost and false-alarm rates. Most of it is NOT in the paper; the file says which parts are |
| `docs/paper-appendices/` | material cut from the paper, each file the LaTeX exactly as it stood, with `README.md` giving the reason for each |
| `docs/paper-appendices/deployment-spec.tex` | the per-stage operational specification: block close, index build, decode step, per-layer cascade |
| `docs/paper-appendices/related-work-taxonomy.tex` | every family sorted by what its warm-up costs and whether its decision is reversible |
| `docs/paper-appendices/scope-boundaries.tex` | the long form of the limitation: three factors, the direction each pushes, the remove-one-factor evidence |
| `docs/paper-appendices/supporting-tables.tex` | four cited-but-unprinted tables (branch anatomy, bits/byte, per-tier recovery, defaults) |
| `docs/paper-appendices/tier1-ablation.tex` | the tier-1-only ablation, a configuration the paper does not ship |
| `docs/paper-appendices/interval-estimates.tex` | Wilson intervals on the small-n needle rates |
| `docs/nope-dividends.md` | what NoPE-MLA confers on seven prior algorithms; no new measurements, three empty on this checkpoint |
| `docs/INFRA_OPTIMIZATIONS.md` | serving-stack tuning notes |
| `docs/whitepaper/`, `docs/defense/` | earlier write-ups, kept as dated records |

Numbers in the `.tex` files are macros generated from `results/results.zip` by
the same scripts the paper uses, so these files and the paper cannot disagree.

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
# --vestigekv-activation-min-tokens 0 serves every request through the
# compressed path from token 0 rather than dense below a threshold, so the
# numbers below measure the method itself at every length, including the
# short-context regime where its fixed cost is a disadvantage. It is passed
# explicitly although 0 is now the default, because the command is the record.
# (The old SGLANG_VESTIGEKV_ACTIVATION_MIN_TOKENS env is deprecated and IGNORED;
# so is SGLANG_VESTIGEKV_TOPJ, whose per-head cap became
# --vestigekv-recall-capacity 4096 with a dense fallback on overflow.)
NCCL_P2P_DISABLE=1 PYTHONPATH=$PWD/engine/python \
python -m sglang.launch_server \
  --model-path <kimi-linear-48b> --trust-remote-code \
  --attention-backend vestigekv_mla --tp-size 2 \
  --vestigekv-activation-min-tokens 0 \
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
# capacity-bound. The vestigekv arm passes
# --vestigekv-activation-min-tokens 0.)
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
# (the vestigekv arm passes --vestigekv-activation-min-tokens 0, so
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
#   WRITING A JOB: copy an existing line of the same client and edit it. The client's `args`
#       are not optional in the way they look -- `replay` raises KeyError without `samples`,
#       `stream` silently runs to 128k without `output_len`, and `ruler` writes an UNTAGGED
#       results file (overwriting the arm's canonical run) unless the job carries `server_args`,
#       `tasks` or `lengths`. Each of those three cost a rerun; a hand-written job has hit all
#       of them.
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
#       python mexp/glm53/stream_curve.py --line kimi --output-len 258048 (results/kimi/latency_stream_*);
#       the paper's metric is the server-side one, --source server (the scheduler's gen-throughput
#       lines within +/-2k tokens of each context, median with p10-p90): the client curve's last
#       window carries an end-of-stream artifact (dense 5.6 -> 12.7 ms at 256k on the client
#       side, no such step in the server log) and is read only up to ~200k.
#   stream-vestigekv-256k-m2, stats-vestigekv-stream-256k-m2 -> exploratory (outside the frozen
#       rule): the 4k -> 256k stream at margin 2, timed and with stats, so the long-decode cost
#       of a margin (gates 3 and 4 of the pre-registration) is on record even though the
#       short-answer gate 2 failed every max-base candidate in the sweep.
#   smoke-prefillcal-needle, smoke-prefillcal-replay-64k -> engineering smoke of
#       --enable-vestigekv-prefill-calibration (engine commit "calibrate the recall index during
#       prefill"): the head needle and three saved 64k prompts with stats on; the server log's
#       VKCAL lines show whether the prefill-time builds install before decode (async=True at a
#       seq below the prompt length) and what they fire. Not a pre-registered measurement.
#   lb2-baseline, lb2-vestigekv -> LongBench v2 on the 300 questions whose context is
#       <= 120k Kimi tokens (mexp/kimi/longbench2/: subset_120k.json from make_subset.py, the
#       prompt of upstream lm-eval's longbench2 task ending at "Answer:"), serial, greedy,
#       CTX=135168 with one running request; results/kimi/longbench2/results_<arm>_<tag>.json
#       (accuracy overall and per domain/length/difficulty) and samples_<arm>_<tag>.json.
#       lb2-vestigekv-stats -> the same with stats on: must reproduce the stats-off answers
#       exactly and gives the fetch/fallback figures on real documents. The four choices
#       A/B/C/D are scored at the "Answer:" position by ONE request per question
#       (max_tokens=1, logprobs=20), not generated and not four echo requests -- the model
#       does not reach an answer line inside 128 tokens (1% of 300 parsed), and lm-eval's
#       echo shape pins logprob_start_len=0, which clamps the radix prefix match to zero and
#       re-prefills the shared context four times. Deviations recorded in
#       mexp/kimi/prereg3_realdoc_and_fallback.md, amendment 1.
#         python mexp/kimi/run_longbench2.py --arm <baseline|vestigekv> --port 30000 --tag <job id>
#       Scoring-equivalence check behind that change (one top-k request against four echo
#       requests, argmax and logprob values, shortest subset documents):
#         python mexp/kimi/lb2_scoring_equiv.py [--n 3] [--port 30000]
#       The older generate-and-regex shape ran as longbench2-baseline / longbench2-vestigekv
#       and is superseded; its results files are archived, not carried forward.
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
#   base-ruler-probe -> whether Table 1's RULER row CAN have a Base column. That row reads
#       "---" for both Base arms and the caption says every row passes, which is a claim about
#       the cells that exist. GSM8K (64-shot) and MAUVE (continuation) work on a base model;
#       RULER asks zero-shot instruction-style questions, which a model with no instruction
#       tuning may simply not answer -- and if DENSE Base cannot do the task, the dense/vestigekv
#       comparison on that row is vacuous and the dash is the honest entry, not a gap.
#       One arm only (dense Base), two tasks, one length, n=10: enough to see whether the
#       baseline is on the floor. Run the full 13x5 grid on Base only if this says otherwise.
#         MODEL=moonshotai/Kimi-Linear-48B-A3B-Base VK_JOB=base-ruler-probe \
#           bash mexp/kimi/baseline.sh
#         python mexp/glm53/run_ruler.py --arm baseline --port 30000 \
#           --model moonshotai/Kimi-Linear-48B-A3B-Base --n 10 \
#           --tasks niah_single_1,ruler_qa_squad --lengths 32768 \
#           --out results/kimi/ruler --tag base-ruler-probe
#       READ: dense Base near 0 on both tasks => the dash stays and the caption says why.
#       Dense Base comparable to Instruct => the row is fillable and the full grid is worth 4.7h.
#       RESULT 2026-09-19, 1.1 min: niah_single_1 1.00, ruler_qa_squad 0.62 with sensible
#       answers ("France", "10th and 11th centuries"). Instruct dense is 0.75 on qa_squad at
#       32k, so Base is in the same band, not on the floor. The row is fillable; the hypothesis
#       that a base model cannot answer RULER's zero-shot prompts is WRONG. Full grid follows.
#   ruler-base-n50-{baseline,vestigekv} -> Table 1's missing Base column, at the Instruct line's
#       own protocol so the two halves of that row are comparable: 13 tasks x 4k-64k, n=50,
#       both arms, quality line (radix off, seed 0). Writes to results/kimi_base/ruler so it
#       cannot collide with the Instruct grid, whose files carry the same arm/n/lengths stem.
#         MODEL=moonshotai/Kimi-Linear-48B-A3B-Base VK_JOB=ruler-base-n50 \
#           bash mexp/kimi/{baseline,vestigekv}.sh
#         python mexp/glm53/run_ruler.py --arm {baseline,vestigekv} --port 30000 \
#           --model moonshotai/Kimi-Linear-48B-A3B-Base --n 50 \
#           --lengths 4096,8192,16384,32768,65536 --out results/kimi_base/ruler
#       Macros come from make_ruler_numbers.py's kimi_base line (tag KB); Table 1's two dashes
#       become \rulerKBDMean and \rulerKBVMean.
#   mk3-copyfidelity-n50-{baseline,vestigekv} -> the multi-key study. The generations say the
#       multikey_3 gap is TWO failures, not one: 6 of 9 errors across two arms return the correct
#       UUID with one character wrong, and 3 return a different needle. Only the second is the
#       selection failure the paper describes. Three checks (mexp/kimi/analyze_multikey.py) rule
#       out interference as the cause of the first: no near-miss output is any UUID in its own
#       haystack; the substituted character appears in 6.4% and 6.1% of the other haystack UUIDs
#       against 1/16 = 6.25% for a hex digit at chance; and the same shape appears in
#       niah_single_3, whose haystack holds ONE UUID. So it is copy fidelity, and a key-value
#       haystack raises its RATE (4/50 vs 1/50) without changing its mechanism.
#       THIS RUN TESTS THAT RATE CLAIM, which 4-vs-1 cannot carry. Two tasks only --
#       niah_multikey_3 (key-value haystack) and niah_single_3 (prose haystack), the same
#       36-char UUID copy in both, so the haystack is the only difference -- at 32k and 64k,
#       n=50, BOTH arms. The dense arm is not a formality: it says whether the copy ever fails
#       without compression, and the whole reading depends on the answer.
#       Samples MUST be kept: the split is invisible in the scores (results/README.md).
#       LAUNCH COMMANDS, exactly as issued (registered 2026-09-19 before the run):
#         # server -- quality line, so common.sh defaults apply (CTX=73728, radix OFF, seed 0)
#         VK_JOB=mk3-copyfidelity-n50 bash mexp/kimi/{baseline,vestigekv}.sh
#         # client
#         python mexp/glm53/run_ruler.py --arm {baseline,vestigekv} --port 30000 \
#           --model moonshotai/Kimi-Linear-48B-A3B-Instruct --n 50 \
#           --tasks niah_multikey_3,niah_single_3 --lengths 32768,65536 \
#           --out results/kimi/ruler --tag mk3-copyfidelity-n50
#       READ IT AS A RATE COMPARISON, not a quality number: 100 items per (task, arm).
#       The prediction under test is near-miss rate on multikey_3 > on single_3 in the
#       compressed arm, with neither showing near misses in the dense arm.
#   lb2gen-{dense,vk} -> LongBench v2 that actually reaches the compressed path. The existing
#       lb2-{baseline,vestigekv} runs score the four choices from ONE token, whose logits come
#       from the prefill's last position; vestigekv's forward_extend is the unmodified base
#       kernel over the full pool (compression only changes forward_decode), so that protocol
#       returns BIT-IDENTICAL logprobs on both arms by construction -- verified, 300/300
#       questions, same floats -- and measures nothing about the cache policy. Those runs are
#       kept: they are an exactness check on real documents and they price the compression
#       events at +1.3% of prefill. They are not a retrieval result and the paper no longer
#       calls them one.
#       These two answer by GENERATING 1024 tokens, so 1024 decode steps run on the compressed
#       path. NOT 128: utils.py already records that measurement -- this model reasons before
#       answering and reaches the answer line in 1% of 300 questions inside 128 tokens, both
#       arms at the 0.25 floor -- and 1024 is also far past the ~18 calibration queries the
#       recall index needs to leave the Z_MAX clamp. The letter is parsed by an explicit
#       verdict, then a parenthesised choice, then the last standalone letter; misses are
#       counted so a protocol that stops parsing is visible.
#       LAUNCH COMMANDS, exactly as the runner issues them (audited 2026-09-18):
#         # server -- QUALITY line, RADIX unset so --disable-radix-cache is ON
#         CTX=135168 MAX_REQS=1 MAMBA_SLOTS=8 CHUNK=4096 GRAPH_BS=1 \
#           [ENGINE=$HOME/vestigekv-wt/engine-fused] bash mexp/kimi/{baseline,vestigekv}.sh
#         # client
#         python mexp/kimi/run_longbench2.py --arm {baseline,vestigekv} --port 30000 \
#           --model moonshotai/Kimi-Linear-48B-A3B-Instruct \
#           --out results/kimi/longbench2 --max-tokens 1024 --tag lb2gen-{dense,vk}
#       SEEDS: server --random-seed 0 (common.sh default); the client is temperature 0 and
#       serial, so the run is deterministic. The answer is parsed as the first A/B/C/D in the
#       generated text; questions where none appears are counted in choices_absent_from_topk.
#   litspeed-{64,128,256}k-{dense,vk} -> the speedup on REAL DOCUMENTS with a real decode,
#       which the paper otherwise claims only from a synthetic stream of random tokens. Each job
#       continues full-length published novels (truncated to exactly --input-len, a
#       different window per request so the radix cache cannot serve the second from the first)
#       for 4096 tokens with ignore_eos, at bs=1 under the paper's serving config
#       (MAX_REQS=1 MAMBA_SLOTS=8 CHUNK=4096 GRAPH_BS=1 RADIX=on; vk on ENGINE=engine-fused,
#       3113f89790). Matched dense and vestigekv arms at each context, so the ratio is a real
#       speedup rather than a one-armed rate:
#       LAUNCH COMMANDS, exactly as the runner issues them (audited 2026-09-18 against
#       queue_runner.py run_client()/ServerHandle.ensure() and mexp/kimi/common.sh):
#         # server, dense arm            CTX: 64k->73728, 128k->135168, 256k->270336
#         CTX=<ctx> MAX_REQS=1 MAMBA_SLOTS=8 CHUNK=4096 GRAPH_BS=1 RADIX=on \
#           bash mexp/kimi/baseline.sh
#         # server, vestigekv arm
#         CTX=<ctx> MAX_REQS=1 MAMBA_SLOTS=8 CHUNK=4096 GRAPH_BS=1 RADIX=on \
#           ENGINE=$HOME/vestigekv-wt/engine-fused bash mexp/kimi/vestigekv.sh
#         # client, both arms            input-len/num-prompts: 65536/8, 131072/6, 262144/4
#         python mexp/kimi/continue_text.py --port 30000 \
#           --model moonshotai/Kimi-Linear-48B-A3B-Instruct \
#           --input-len <N> --output-len 4096 --num-prompts <K>
#       RADIX=on is not a tuning choice: mexp/kimi/common.sh documents it as the production
#       default and the protocol the paper's serving numbers were taken under, and it is what
#       drops --disable-radix-cache -- this is the PERFORMANCE line; every quality-line job
#       leaves RADIX unset, which is off, and 0 of the 126 queued jobs violates that (audited
#       2026-09-18). ENGINE is consumed by that same file
#       (PYTHONPATH=${ENGINE:-$ROOT/engine}/python), so the vk arm runs the tree the serving
#       figure used. SEEDS: the server is --random-seed 0 (common.sh default, not overridden
#       by any job); continue_text.py takes no seed because it has no random source -- the
#       novels are sorted longest-first and each request's window is offset by a fixed
#       (i // n_docs) * 997 -- and the client sends ignore_eos, temperature 0, max_tokens 4096,
#       so the run is deterministic end to end.
#       Prompt counts are set by how many novels are long enough at the *8 chars/token filter:
#       PROVENANCE: the text is ordinary long-form fiction -- The Count of Monte Cristo, Don
#       Quixote and Moby-Dick are identifiable among the eight used at 64k. They are read
#       from the Literary and Detective sub-domains of THUDM/LongBench-v2 because that is a
#       convenient packaged source of very long prose, NOT because anything here is a
#       LongBench measurement: no question, choice or score of that benchmark is used, only
#       its `context` field. The paper says 'published novels' for that reason.
#       19 documents clear 64k, 6 clear 128k, 4 clear 256k. The script fails loudly rather than
#       silently shortening if none does.
#       WHY REAL TEXT IS THE CONSERVATIVE CASE, not the flattering one: fallback is HIGHER on
#       novels than on random tokens at matched context and decode length (0.483 vs 0.407,
#       pre-registration 6), so a speedup that survives here is not an artifact of synthetic
#       input. This pairs with the LongBench-v2 quality run, which at max_tokens=1 prices the
#       warm-up with nothing to amortise it over (+1.3%); these price the other end, where a
#       4096-token decode has something to amortise it over.
#   ruler-{baseline,vestigekv}-long-{mid,max} -> RULER above 64k, the evidence gap this study
#       otherwise ships with: the paper claims 1.53x at 508k and a speedup at 256k while its
#       capability evidence stops at a 128k needle and 64k RULER. ONLY -mid RUNS
#       (131072,262144). -max was queued at 524288 and dropped, as 1048576 had been before it.
#       The reason is measured, not estimated: 136 s/question at this shape, so the 512k pair is
#       another ~8h on top of -mid's ~8h, while n=5 puts 2 sigma on the arm mean at ~0.053
#       against a gap measured at 0.020 -- it can bound collapse and nothing finer, and at 512k
#       the dense control is unlikely to be strong enough for even that to inform. 256k stays
#       because it is where the headline speedup is claimed. (My original sizing said 1.4h for
#       the -mid pair; it extrapolated prefill tokens linearly and ignored that attention grows
#       superlinearly with context.) Served with CTX=1064960 MAX_REQS=2 GRAPH_BS=2 and
#       SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1; the model declares model_max_length=1048576,
#       so these sit inside its window rather than testing extrapolation.
#       LAUNCH COMMANDS, exactly as the runner issues them (audited 2026-09-18):
#         # server -- QUALITY line, so RADIX stays unset i.e. --disable-radix-cache is ON
#         CTX=1064960 MAX_REQS=2 MAMBA_SLOTS=2 CHUNK=4096 GRAPH_BS=2 \
#           SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 VK_JOB=<job> \
#           bash mexp/kimi/{baseline,vestigekv}.sh
#         # client                       -mid: --lengths 131072,262144   -max: --lengths 524288
#         python mexp/glm53/run_ruler.py --arm {baseline,vestigekv} --port 30000 \
#           --model moonshotai/Kimi-Linear-48B-A3B-Instruct --n 5 \
#           --lengths <lengths> --out results/kimi/ruler --tag <job>
#       SEEDS: server --random-seed 0 (common.sh default); run_ruler.py --seed 0 sets the
#       lm-eval random, numpy and torch seeds together, and requests are serial and greedy.
#       READ THESE AT n=5, WHICH IS WHAT THEY ARE FOR. 13 tasks x 2 lengths x 5 = 130 prompts per
#       arm puts 2 sigma on the arm mean at about 0.053, and the gap measured at 4k-64k is 0.020:
#       these bound COLLAPSE, they do not measure the gap. A result of "vestigekv tracks dense"
#       here means "it does not fall apart at 256k", and the sample size must be stated wherever
#       the number appears so it is not read against the n=50 table.
#       DEVIATION, recorded 2026-09-18, after the run. The band above treats the two arms as
#       independent; they are not, since both answer the same items under seed 0. A PAIRED
#       reading of the retained samples_*.json (exact sign-flip permutation, macros
#       \rulerKLPaired{N,Delta,TwoSigma,P} and \rulerKLDisagree*) gives 2 sigma 0.035 on the
#       measured -0.046, p=0.008, with all 8 disagreeing items against vestigekv. This analysis
#       was NOT pre-registered and was chosen after seeing the data; it is reported because it
#       cuts AGAINST this method -- it upgrades the 128k-256k gap from "unresolved" to real --
#       and the paper prints the pre-registered reading first and labels the paired one post-hoc.
#       The n=50 grid cannot be re-read this way: its samples_*.json were deleted in the results
#       reduction (results/README.md, "What was removed entirely"), and its vestigekv arm's
#       record predates the seed field, so item identity across its arms is not provable from
#       the records either. Its bands stay unpaired, hence conservative in the direction that
#       flatters the localisation claim. DO NOT delete samples_*.json for a compared arm again.
#   stream-vestigekv-256k-origin -> the same timed 4k->256k stream served from origin/vestigekv
#       (the paper's backend, ENGINE=~/vestigekv-wt/engine-origin, a detached worktree at
#       281abac) on this box and this protocol: the per-step cost of every commit since (fence,
#       int32 tables, int64 CSR, kernel tiles, margin argument, per-request basis) read as the
#       server-side latency ratio against stream-vestigekv-256k (the current tree) at 64k/128k/256k;
#       the "--log-level info" server arg is sglang's default and only makes the runner tag the
#       output file. A gap above the window spread is bisected with the same job at the
#       intermediate commits.
#   prod-stream-vestigekv-256k-{fused-base,fencestub} -> the probe that splits arming a fence
#       from running it (engine branch vestigekv-fused-fallback). Both arms are that branch
#       under the production protocol; the second sets SGLANG_DEBUG_VESTIGEKV_FENCE_STUB=1,
#       which compiles the gather's dense branch away while the rest of the fence stays armed.
#       If the gap to the disarmed arm collapses, the cost is the branch existing (register and
#       shared-memory pressure); if it stays, the cost is elsewhere, and the next suspect is the
#       eager path's per-step page-table materialization in _dense_rows. The stub arm's output
#       is wrong on the steps that fence: it is a timing probe, never a quality arm.
#   td-needle, td-replay-64k, td-stream-256k, td-ruler-64k, td-ruler-64k-off, td-profile-256k -> the tier-decode router
#       (engine branch vestigekv-fused-fallback, --enable-vestigekv-tier-decode): stage 1 reads
#       a lane's rows from the kept table and the fetch buffer, or from the page table when the
#       lane is fenced, so the per-step CSR copy goes away while the overflow path stays. The
#       fence stub says 4.33 ms/token at 256k is what that is worth; td-stream-256k is the same
#       production protocol and that is its gate. The first two are the correctness smoke (the
#       head needle, the three saved 64k replay prompts) and td-ruler-64k is the 13 tasks at
#       4k-64k, which must reproduce the default arm's answers: the row sets are unchanged, so a
#       difference is a bug, not a quality result. All four run with
#       ENGINE=/home/user/vestigekv-wt/engine-fused.
#       td-stream-256k-nodense adds --disable-vestigekv-recall-overflow-fallback, which turns the
#       fenced branch off: an overflowed scan then truncates to its fired rows, which is exactly
#       what origin/vestigekv does. It is the like-for-like comparison against
#       prod-stream-vestigekv-256k-origin, and it answers whether the forked stage 1 itself costs
#       anything, with the overflow work removed from both sides. Not a quality arm: truncating
#       is what pre-registration 3's C1 measured at -0.057 of the RULER mean.
#   stats-stream-64kprefill-long -> the one variable stats-stream-64k-x130 left confounded.
#       That control holds the context, the generated length and the request count against
#       RULER and changes only the text, which rules the text out (random continuation falls
#       back 0.41 against RULER's 0.36). But it still differs from the long stream in TWO ways:
#       its 64k arrived by prefill rather than by decoding up to it, and its calibration is
#       amortised over 14 steps rather than 258050. This job keeps the 64k prefill and decodes
#       4096 steps, so only the amortisation changes. Near 0.0002 puts the cause on the step
#       count; near 0.3-0.4 puts it on the context's provenance, and then "the speedup is a
#       long-decode number" needs a second qualifier in the paper.
#   stepattr-mk3, stepattr-s1 -> the blind spot: what happens at ANSWER steps.
#       Every offline study here runs on CALIBRATED tiers sampled at the eight decode steps
#       right after the build, while 81% of RULER's scans run on a PROVISIONAL tier at answer
#       positions. So "the certificate misses nothing" (99.9% held out, no decay to k=8),
#       "coverage rises with k" (0.620 -> 0.965) and "fired rows are 0 to 13" all describe the
#       19% of steps RULER barely enters -- and the third was already caught wrong by two
#       orders of magnitude (serving fires 90 to 230).
#       SGLANG_DEBUG_VESTIGEKV_STEPDUMP=1 writes one record per (step, layer, lane) to
#       results/kimi/stepattr/: dense coverage of the attended set, its per-head minimum,
#       whether dense's argmax row was attended at all, entropy, rows beating max1, plus the
#       tier's state (provisional, zp, built_at) so the 81% separates from the 19%.
#       Two jobs, one variable: niah_multikey_3 (0.760 at 64k) against niah_single_1 (1.000),
#       n=10 at 65536 only. Same answer length family, same haystack, different key count.
#       PREDICTION: on multikey_3's failing steps, coverage or top1_attended is materially
#       worse than on single_1's, and the difference sits on PROVISIONAL records.
#       FALSIFIER: the two tasks' records are indistinguishable. That would say the attended
#       set is not where multi-key is lost even at answer positions, and the remaining
#       candidate is the arithmetic of attending rather than the choice of them -- which the
#       answer-length test already argues against (r = -0.096, and niah_multikey_3 and
#       niah_single_3 emit the same 37 characters with gaps of +0.140 and +0.000).
#   fence-mk1-ruler, fence-mk4-ruler -> the retreat: fence multi-key instead of ranking it.
#       Eight recall-side mechanisms are ruled out and the gap survives all of them, so this
#       stops trying to recover the row and instead notices the case and attends densely. A
#       fenced lane IS dense, so the fence is exactly right where the certificate is weakest;
#       it costs fallback and nothing else. The detector is free: on the Kimi dumps the
#       fired-row count rises monotonically with how many archived rows the query needs --
#       median 0, 1, 2, 4, 13 for 0, 1, 2, 3, 4+ -- and per lane, fencing above ONE fired row
#       catches 100% of the lanes where some head needs two or more, fencing 29.7% of lanes
#       against the 0.305 RULER already pays.
#       Two arms: --vestigekv-multikey-fence-rows 1 (catch everything) and 4 (59.5% of
#       multi-key lanes, 13.5% of lanes). A0 at n=10 is multikey_2 0.920, multikey_3 0.860,
#       qa_hotpot 0.680, 65-cell mean 0.9179.
#       PREDICTION: multikey_2 and multikey_3 move toward dense (1.000) and the 65-cell mean
#       does not fall; fallback at 64k rises from 0.305 toward 0.5 at fence 1.
#       FALSIFIERS, either ending the retreat: (1) multi-key does NOT improve even though the
#       fenced lanes are attending densely -- which would mean the failing steps are not the
#       ones firing several rows, and the detector is aimed at the wrong thing; (2) any
#       currently-perfect task falls, i.e. the fence costs accuracy somewhere it was fine.
#       NOTE this is a SHORT-ANSWER knob: a long decode fires few rows per scan and would pay
#       far more fallback for far less, so it defaults off and the paper must say so.
#   lit-continue-64k-short, lit-continue-64k-long -> continue a NOVEL, to separate natural
#       text from decode length. mexp/kimi/continue_text.py, new client "continue": real
#       prose from LongBench v2's Literary and Detective sub-domains (Journey to the West,
#       The Count of Monte Cristo, Don Quixote -- 18 documents over 512k characters),
#       truncated to exactly 64k tokens, continued with ignore_eos so every request
#       generates its full length.
#       WHY: pre-registration 6's "naturalness" ordering -- self-continuation 0.003, real
#       documents 0.241, needles 0.360, random 0.407 -- is ALSO an ordering by decode length
#       (~250k, 1, 14, 14 tokens). The two axes were never separated, and the decode count
#       moves the rate 180x where the text type moves it 35%, so that ordering is mostly the
#       confound. These two hold the text natural and vary only the length; x130 (random,
#       same context, same counts) holds the length and varies only the text. The four are a
#       2x2 and the paper's applicability claim rests on which axis wins.
#       PREDICTION: short 0.25-0.40 (natural text shaves something off random's 0.41 but the
#       step count dominates) and long 0.002-0.01, with the short run ~80% provisional. A
#       novel is the most favourable case a spectral prior can get, so if it still falls back
#       like RULER at 14 steps, "the prior fails on non-natural text" is not the explanation
#       for the fallback rate and pre-registration 6's amendment 1 has to be rewritten.
#       FALSIFIER of the amortisation story: short comes in near 0.01, i.e. natural text
#       alone fixes it without any extra decode steps.
#   idxstate-stream-64k-short -> the third corner of the amortisation table, with buckets.
#       stats-stream-64k-x130 already showed that 64k prefill + 14 decode steps of RANDOM
#       text falls back on 0.407/0.439 -- MORE than RULER's 0.305 -- so the short-decode
#       regime reproduces the gap with no needle question anywhere. What that run predates is
#       the index-state instrument, so it cannot say WHY. This is the same job (input 65536,
#       output 14, 130 prompts) on ENGINE=engine-fused with the buckets on.
#       PREDICTION before running: provisional serves ~80% of scans, matching RULER's 81%
#       against the long stream's 0.6%, and the per-state overflow rates match both
#       (prov ~0.32). FALSIFIER: a fresh-dominated mix here would mean the short-decode
#       fallback has a second cause the index state does not explain, and the claim "the
#       fallback rate is the state mix times the per-state rate" is wrong.
#   omit-blend-ruler-n10 -> the omitted-mass arm, the one lever the six refutations left.
#       Six independent measurements now say more rows cannot help multi-key: the margin
#       sweep 0->3 does not move it, A2's corrected target nets zero, A4 has nothing to
#       recover, and held-out recall is 99.9% with no decay from k=1 to k=8. All six ask the
#       same question -- is the row that beats max1 fired -- and all six answer yes. None of
#       them looks at the softmax SCALE. Measured on the Kimi dumps: the attended set is ~6%
#       of rows and captures 57% of the dense softmax mass (min 24%), so every retained weight
#       is inflated by Z/Zv, mean 2.9x and up to 8.7x. Offline, correcting the denominator
#       from the certified bound the scan already computes AND carrying the omitted mass at
#       that set's MASS-WEIGHTED centroid cuts the attention-output error from 0.471 to
#       0.195; the plain archive mean only reaches 0.367, and the denominator alone makes it
#       WORSE (0.652), because that subtracts the mass without returning its value. An oracle
#       with the true mass and true values reproduces dense exactly, which is what says the
#       decomposition is right. The weighted centroid is not a tweak: the correction is one
#       turn of the online-softmax recurrence O_n = lerp(O_{n-1}, v_n, sigmoid(s_n - lse)),
#       and a synthetic row standing for the omitted set is exact only if it carries that
#       set's weighted centroid.
#       THREE JOBS. The first two measured NOTHING and their numbers are void, not null:
#         omit-blend-ruler-n10, omit-blend-exact-n10 -- decode replays a CUDA graph, so the
#           router's python runs once at CAPTURE; the blend was installed per step from
#           _recall_step, i.e. after capture, so its ops were never recorded and every replay
#           ran the uncompensated path. The tell was two compensators that differ by a lot
#           offline (0.367 vs 0.195) returning BIT-IDENTICAL answers on all 13 tasks over 650
#           questions. Do not quote either; VOID.
#         omit-blend-exact-n10-v2 -- the registered arm, commit 90df7a5: buffers allocated
#           before capture and installed where rows_for_layer is, so the blend is IN the
#           graph. Two log lines (VKBLEND active, VKBLEND fill) must appear in the server log
#           or the run is void again and must not be read as a result.
#         omit-blend-ruler-n10  the ARITHMETIC-mean variant (offline 0.367). It launched at
#           11:06 from a working tree rewritten at 11:10, so it has NO commit. Indicative
#           only; its numbers may not be quoted and do not go in the paper.
#         omit-blend-exact-n10  the registered arm: the mass-weighted centroid (offline
#           0.195), commit 974e266. This is the one that decides the line.
#       Testing the exact form rather than a cheap approximation is deliberate: if the BEST
#       available compensation does not move accuracy the line is dead, whereas a weak
#       variant failing proves nothing. Cheap approximations already ruled out offline, so
#       they are not retried: the rank-64 sketch is a poor basis for the VALUE (0.531, worse
#       than doing nothing); bucketing the archive on its top principal direction saturates
#       at 0.311; firing the top 2048 omitted rows exactly captures only 28.7% of the mass
#       (a long flat tail, not a head); and the NoPE dividend -- phi_j = (side, csk, rho) and
#       v_j are static, so a tail summary could be built once at block close, which RoPE
#       would forbid -- is structurally real but its FAVOR+ random-feature realisation
#       underflows at this score scale (exp(-|u|^2/2) -> 0) and returns the uncompensated
#       output at D = 128, 512 and 2048 alike.
#       This runs that arm end to end: SGLANG_DEBUG_VESTIGEKV_OMITTED_BLEND=1, 13 tasks,
#       4k-64k, n=10, against the delivered A0 at the same n (multikey_2 0.920,
#       multikey_3 0.860, qa_hotpot 0.680, 65-cell mean 0.9179). `probe: true` runs the head
#       needle first so a catastrophic sign or scale error costs one minute, not twenty.
#       PREDICTION, before running: multikey_2 and multikey_3 rise. FALSIFIERS, either of
#       which ends the arm: (1) the targeted tasks do not move -- output error was a weak
#       proxy and 20% of it buys nothing; (2) ANY currently-perfect task falls, which says
#       the blend trades the tasks that work for the ones that do not. Note the arm rewrites
#       every attention output, so collateral damage is the real risk and all 13 tasks run,
#       not just the three that are failing.
#       RISK: the arm materialises the [H, archive] score matrix and the side/csk/rho views
#       per lane per layer per step -- the three the steady-state footprint deliberately does
#       not hold. ~430 MB resident at 64k x 4 lanes against ~9 GB free. An OOM is a cost of
#       the probe, not a result.
#   mexp/kimi/cert_holdout_offline.py -> does the certificate hold OUT OF SAMPLE?
#       ent_margin_offline.py and z_topk_offline.py both score the fitted zp on the queries
#       zp was fitted on. A conformal quantile recovers its target in sample BY CONSTRUCTION,
#       so "1081 of 1081 rows recovered" was never evidence that the recall step loses nothing
#       -- and the ruling that stopped arms A2 and A4 rests on it. This refits zp on half the
#       calibration points with the delivered rule and scores the other half, reporting both
#       row recall (of the archived rows whose true score beats the kept maximum, how many
#       fire) and QUERY recall (of the queries needing at least one, how many get ALL of them,
#       broken down by how many they need). The second is the quantity a k-key question is
#       made of; a marginal per-row guarantee does not deliver it.
#         python mexp/kimi/cert_holdout_offline.py --split random   (and --split pos)
#       Offline, no GPU, reads results/kimi/caldump. PREDICTION before running: out-of-sample
#       row recall lands near the 0.90 target rather than the in-sample 1.00, and query recall
#       falls off with k. If instead out-of-sample recall is also ~1.00, the certificate really
#       is not where the rows go and the multi-key gap is tier-1's keep decision, not recall.
#   idxstate-ruler-64k, idxstate-stream-64kprefill -> WHICH index the misses are under.
#       The offline scan says a CALIBRATED certificate fires every archived row that beats the
#       kept maximum (1081 of 1081 on the Kimi snapshots), so the rows RULER loses on multi-key
#       are lost on steps some other index served. Until now the stats line pooled every scan
#       into one fired-row histogram, so that could not be checked. The delivered tree now
#       splits scans, fired rows and overflows three ways by the state of the index that served
#       them -- PROVISIONAL (no tier, or z clamped to Z_MAX because calibration has not met
#       min_hard), FRESH (fitted, context not outgrown), STALE (fitted past
#       INDEX_STALE_FACTOR x the length at the fit) -- and reports them as
#       idx(scans/fetch/ovf)[prov=.. fresh=.. stale=..].
#       These two run that instrument on the two ends of the amortisation axis that
#       stats-stream-64kprefill-long just isolated: the same 64k prefill, 14 decode steps
#       (RULER) against 4096 (the stream), where fallback measures 0.36 and 0.0045. Both on
#       ENGINE=/home/user/vestigekv-wt/engine-fused, which is the only tree carrying the
#       instrument.
#       PREDICTION, recorded before running: if the amortisation reading is right, RULER's
#       scans are mostly prov and its overflows are concentrated there, while the long stream's
#       are mostly fresh. The falsifier is sharp -- RULER showing a fresh-dominated mix, or its
#       per-bucket overflow rate being flat across the three, says the misses are NOT an
#       index-state effect and the multi-key gap needs a different explanation.
#   stats-stream-64k-x130 -> the controlled counterpart of stats-vestigekv-ruler-64k: the same
#       engine tree, the same env (CTX 73728, radix off, stats on), the same context (64k
#       prefill), the same generated length (14 tokens) and the same number of requests (130,
#       matching RULER's 13 tasks x 10 samples and its 1850 steps). The ONLY variable left is
#       what the text is -- random continuation instead of a needle question. RULER measures
#       fallback 0.360 at 64k while the long stream measures 0.00014 at the same context, and
#       this says how much of that 2100x is the question rather than the short-answer regime.
#         (queue job; the stream client takes args.num_prompts for this)
#   origin-stats-64k-eager, current-stats-64k-eager -> how many rows each tree's certificate
#       fires, measured where origin can report it. origin-stats-stream-256k produced no VKSTATS
#       because origin's in-graph branch returns before its stats branch, so the instrument
#       never runs under the production protocol; these two disable the cuda graph and read
#       fetched=<n>/call at 64k. Their timings are meaningless and are not read.
#   DELIVERED IMPLEMENTATION (engine branch vestigekv-fused-fallback): the decode step reads a
#       lane's rows from the tiers, and a fenced lane whose page table is one contiguous run
#       computes its row ids as base+offset. Both are unconditional -- the flags that used to
#       select them are gone, and the alternatives live in that branch's history. Affinity is
#       DETECTED, not asserted: _verify_affine re-checks the table when the batch composition
#       changes and the pack's prep kernel clears the flag on any step whose row does not land
#       at base+seq-1. 256k: 4.369 ms/token, 1.276x over dense, against the CSR design's 1.169x.
#   td-stream-512k -> the delivered implementation, 4k prefill decoded to 512k (CTX 528384,
#       520192 output tokens, ~50 min). One run answers two questions: passing 256k it must
#       reproduce td-stream-256k-affine's 4.369 ms/token, which confirms end to end that the
#       row-loop refactor generates equivalent code (the disassembly already says so: same
#       instruction histogram, 3024 each, same 237/0 registers, same 8 LDGSTS); and its 508k
#       point is the paper's long-context number on the delivered code.
#       The dense arm is NOT re-run: results/latency_stream_serverlog_dense_instruct.log from
#       the 2026-09-15 campaign gives 4.360 / 4.786 / 5.590 / 7.120 ms/token at 64k / 128k /
#       256k / 508k under the same server-side metric, and its first three reproduce this
#       line's own dense baseline to three decimals (4.360 / 4.783 / 5.576), so the protocols
#       are comparable and 7.120 is the 508k reference.
#   td-ruler-64k-final -> the 13 tasks x 4k-64k on that delivered default, the arm the paper's
#       quality numbers should cite. td-ruler-64k / td-ruler-64k-off are its matched pair for
#       the tier path alone (same tree, tier-decode on and off), from before the flags went.
#   td-stream-256k-affine -> the same 256k stream with --enable-vestigekv-affine-page-table:
#       a fenced lane computes its row ids as base+offset rather than loading them, which
#       makes the K address affine and gives the loop its own async-copy pipeline (LDGSTS 4
#       -> 8, checked with mexp/kimi/fence_disasm.py). It is an ASSERTION about the allocator,
#       not a check: SGLANG_DEBUG_VESTIGEKV_STATS reports pagetable_affine, measured 1.0000
#       over 258050 steps here, and a fragmented table would silently attend wrong rows.
#   td-stream-256k-nospill -> the 256k retest after the fenced arm stopped spilling (255 regs
#       and 40B of stack became 200 and none; see mexp/kimi/fence_disasm.py and the engine rule
#       .claude/rules/disassemble-check-for-spills.md).
#   origin-stats-stream-256k -> origin/vestigekv on the production 256k stream with
#       SGLANG_DEBUG_VESTIGEKV_STATS=1. Origin's stats line predates the fetch percentiles and
#       the fallback rate, but it carries fetched=<n>/call, the mean rows a scan fires, and that
#       is the question: origin clamps to the same 4096 buffer and truncates silently, so
#       whether it "never falls back" is a claim about how much its certificate fires, not about
#       its capacity. The calibration differs -- origin solves z against the kept maximum, the
#       current tree against the archived row's own true score -- so the current tree fires more
#       by construction, and this measures by how much.
#       td-profile-256k adds the per-kernel view
#       (200 profiled steps at 256k, diffed against profile-vestigekv-perf-256k with
#       mexp/kimi/kernel_diff.py): it is what shows the gather kernel gone from the step rather
#       than only a faster total.
#   Findings and the method of this audit: mexp/kimi/perf_audit_origin_vs_current.md.
#   mexp/kimi/bench_dense_mla.py times the experimental page-table dense MLA decode
#       (engine branch vestigekv-fused-fallback, module vestigekv/dense_mla.py) against a torch
#       reference and against the CSR copy it would remove, at a fenced lane's row count. Run
#       with ENGINE=~/vestigekv-wt/engine-fused and only when no timed job is measuring.
#   mexp/kimi/bench_gather.py reads N rows of 576 bf16 through an index list, contiguous
#       against scattered (one row every 32, the attended tier's spacing) and against random,
#       over a grid sweep: it says how much of the attended tier's ~88 GB/s is locality and
#       how much is parallelism, i.e. whether a contiguous arena for the kept rows is worth
#       building. Standalone GPU script, no server; run it when no timed job is measuring.
#   mexp/kimi/rank_skew.py <trace prefix>... splits each rank's collective time (a sink that
#       absorbs the wait for the slower rank) from its compute, which is what a code change
#       actually costs; it is how the 256k fence cost was found.
#   prod-stream-{baseline,vestigekv,vestigekv-origin,vestigekv-nofence}-256k -> the same timed
#       4k->256k stream under the PRODUCTION protocol (RADIX=on in common.sh keeps the prefix
#       cache, as the paper's serving numbers were taken; the quality line stays radix-off).
#       Four arms separate protocol from code: dense, the current tree, origin/vestigekv, and
#       the current tree with --disable-vestigekv-recall-overflow-fallback (origin's truncate
#       behaviour on overflow). MAMBA_SLOTS=8, not the radix-off streams' 1: with the prefix
#       cache on, sglang sizes the hybrid state cache at 5 mamba slots per request and refuses
#       to start below that ("state cache is too small to serve any requests").
#   profile-vestigekv-origin, profile-vestigekv-current -> kernel-level decode profiles of the two
#       trees (origin/vestigekv via ENGINE, and the current tree) on the same stream protocol:
#       mexp/kimi/profile_stream.py drives one 4k-in, ignore-eos request and, when the server's
#       decode log reaches 128k and 256k, runs the built-in torch profiler for 200 forward steps
#       (POST /start_profile with num_steps; CUPTI records the kernels inside the CUDA-graph
#       replay); traces in results/kimi/profile/<job>/ctx<N>k-TP-0.trace.json.gz. The diff:
#       python mexp/kimi/kernel_diff.py results/kimi/profile/profile-vestigekv-origin/ctx256k-TP-0.trace.json.gz \
#           results/kimi/profile/profile-vestigekv-current/ctx256k-TP-0.trace.json.gz --label-a origin --label-b current
#       gives per-kernel GPU us/step for both trees sorted by the difference, which is what
#       names the commit behind a latency gap (then bisected with the same job at that commit).
#   mexp/kimi/make_lb2_numbers.py regenerates the paper's LongBench v2 macros from
#       results/kimi/longbench2/results_{baseline,vestigekv}.json:
#       python mexp/kimi/make_lb2_numbers.py [--out ~/vestigekv_paper/lb2_numbers.tex]
#       Overall and per domain / difficulty / length bucket per arm, the signed paired
#       difference, and the questions the arms answer differently in each direction. An arm
#       that has not run emits \PENDING, which errors at build time rather than printing a
#       stale number.
#   mexp/kimi/fence_disasm.py compiles the forked decode stage 1 at both settings of the FENCE
#       constexpr and disassembles each, so "the two differ by one branch" is checked rather
#       than assumed:
#         python mexp/kimi/fence_disasm.py [--out <dir>]
#       Prints registers, spills and instruction counts for both and writes ptx.diff/sass.diff.
#       It found the fenced build at 255 registers with 40 bytes of stack against 200 and none,
#       which is a cost every step pays because register allocation is compile-time.
#   caldump-kimi-64k -> step 0 of pre-registration 6: one short RULER job (niah_multikey_3 at
#       64k, n=2) with SGLANG_DEBUG_VESTIGEKV_DUMP_DIR set, so the calibration snapshots exist
#       for Kimi's geometry. z_topk_offline.py then reads them and the table picks the k that
#       arm A2 uses. The GLM snapshots cannot: rank 128 and no sidecar against Kimi's rank 64
#       and a 64-dim one.
#   mexp/kimi/z_topk_offline.py asks what the conformal z has to be to certify a query's
#       top-k archived rows rather than its single best one, on the same dumped snapshots
#       bucket_offline.py reads:
#         python mexp/kimi/z_topk_offline.py --dir results/glm53/caldump --ks 1,3
#       The calibration solves z_req = (true_best - idxs_best)/cert_best, so its guarantee is
#       marginal and about ONE row, while a multi-key question needs k at once and a marginal
#       guarantee does not compose (tau^k). Measured on 8 GLM-5.3 snapshots, certifying the
#       top-3 instead of the top-1 costs a few percent of z and 10-33% more fired rows at p90
#       -- cheap against the 0.14 the method loses on niah_multikey_3. GLM geometry, not
#       Kimi's: the structure transfers, the numbers do not, and Kimi snapshots are not dumped.
#       It also shows z at 6.98-8.31 on the 64k snapshots against Z_MAX 8.0, so that clamp is
#       already binding and weakens the guarantee before k enters.
#   mexp/kimi/ruler_diff.py compares two RULER arms question by question:
#       python mexp/kimi/ruler_diff.py results/kimi/ruler/samples_<a>.json results/kimi/ruler/samples_<b>.json
#       For arms that must be identical (a stats-on twin, or a change that moves no row such as
#       --enable-vestigekv-tier-decode) any differing answer is a bug and the tool names the task,
#       the length and the document; exit status is 1 when they disagree, so a smoke job can gate
#       on it. The control has to be the SAME engine tree with the flag off (td-ruler-64k-off),
#       not the default arm's run on engine/: the two trees differ by eight commits and a
#       difference between them would not be the flag's.
#       A job whose only difference is a server flag needs `args.lengths` or `args.tasks` as well,
#       or the ruler client writes UNTAGGED and overwrites the arm's canonical results file. lm-eval scores only one length per record, so the per-question win counts cover the
#       scored cells alone while the answer comparison covers all 65.
#   PRIORITY (owner, 2026-09-17, supersedes the performance-first rule below): the RESEARCH
#       line runs first and keeps running until the multi-key gap is closed or shown to be
#       unclosable; then the QUALITY line; then the PERFORMANCE line. Research is whatever
#       decides an algorithm change -- today the truncation pair, the regime run, and
#       pre-registration 6's arms. Quality is the paper's numbers (RULER n=50, LongBench, the
#       per-task fallback column, the 1M pair). Performance is the timed streams.
#       A job of a higher line interrupts a running job of a lower one: its state line is
#       dropped from results/kimi/queue_state.jsonl so it reruns. Within the performance line
#       the timing job still goes first, ahead of its own correctness smokes. In-flight work of
#       a lower line is not a reason to queue a higher one behind it.
#   The order of the whole line is fixed: every performance job, then the pre-registration 4
#       candidate arms and their smoke, then the default is committed, and only then the
#       quality jobs -- which all run under that default. Draft rule:
#       mexp/kimi/prereg4_new_default.md (frozen once the performance line's numbers are in).
#   p4-{m0,m2,m3,lse46} -> the pre-registration 4 candidate arms (mexp/kimi/prereg4_new_default.md,
#       frozen with the performance line's numbers): the 5 targeted tasks at 16k/32k/64k, 10 per
#       cell, stats on, all four on the same package (prefill calibration, rebuild trigger 0.05,
#       capacity 16384) and differing only in the margin. The queue file drives the whole line
#       unattended from here; jobs that need the chosen default carry skip:true and are unskipped
#       in one edit once a candidate passes its gates.
#   pc-A{1,2,3}-{ruler,stats64k,stream256k-stats,stream256k} -> pre-registration 2
#       (mexp/kimi/prereg2_prefill_calibration.md): arms A1 = --enable-vestigekv-prefill-calibration,
#       A2 = A1 + --vestigekv-recall-margin 2, A3 = A1 + margin 4.6 with --vestigekv-recall-threshold
#       lse; per arm the 13-task 4k-64k RULER (10/cell), the 64k stats run, and the stats-on and
#       timed 4k->256k streams, read against the A0 (default) and dense records already on file.
#       The default chosen by its rule is committed before ruler-vestigekv-long and the n=50 pair run.
#       Amendment before any pc-* run (prereg2 file): all arms run on engine commit "prefill-time
#       calibrated builds at the first closed block, then at every doubling"; the pc-* jobs run
#       right after longbench2-baseline, and every vestigekv-arm quality job of prereg 3 (LongBench
#       v2 arms, nofallback, fb-*, n=50, ruler-vestigekv-long) is skip:true until the default is
#       chosen, then unskipped and run under it: results taken while decode starts on the
#       provisional index do not describe the method as served.
#   ruler-baseline-long, ruler-vestigekv-long -> the 13 tasks x {128k,256k,512k,1M}, 5 samples/cell,
#       with CTX=1064960 (1M + 16k; SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 because the
#       model's derived context is exactly 1048576), --max-running-requests 2, two mamba slots,
#       --cuda-graph-max-bs-decode 2 (lm-eval's RULER generator is seeded 0 here too)
# step 0 of the line: the RAM weights daemon holds the Instruct checkpoint in the page cache
# and prints its WEIGHT-FP (same tool as the Base line's step 0; results/kimi/weight_daemon_ram.log):
nohup python tools/weight_daemon.py --model moonshotai/Kimi-Linear-48B-A3B-Instruct > results/kimi/weight_daemon_ram.log 2>&1 &
# GPU weight-cache daemon (one process per GPU holding the TP=2 bf16 shards; a server launched
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

### Experiments are scripts, not typed commands

`mexp/exp/` holds one shell script per registered experiment, and
`bash mexp/exp/audit.sh` is run before any of them. The rule exists because
hand-typed runs went wrong twice in ways a script could have caught: an OOM
from `--gpu-expert-layers 99`, and a wrong protocol from `--rhos 0.03125` where
that flag wants a compression RATIO. The audit refuses on exactly those, plus a
missing guard, an `--out` that already exists, and any script this README does
not mention. `_lib.sh` carries the guards: refuse to start while a server, the
queue runner or another harness run holds the GPU; refuse to overwrite a record;
refuse a model directory that is not a plain dir.

**Control the variables, and say which.** Every defect this project has shipped
in a comparison was a loose variable, never a wrong number: two arms of one
table row running different operators (a drop policy against a half-budget
merge); an arm measured at a detector bandwidth nobody deploys; a control whose
engine tree no surviving record names; rows of one table at different trial
counts. None of those is visible in the numbers afterwards, and all of them
were decided before the run.

So every script declares, and `audit.sh` refuses without it:

```
# CONTROL:
#   varies:  the one thing this experiment changes between arms
#   fixed:   seed, trial count, lengths, tasks, checkpoint, engine tree, and
#            every launch knob -- named, not implied
```

Three rules follow from it. Arms that will be printed in one table run from
**one script in one sitting**, because that is the only way to be sure they
differ in what the script says they differ in. An arm measured on a different
engine tree is a different experiment, and the tree is part of `fixed:`. And
`--seed` is passed explicitly even where the runner defaults it, because a
default is not a declaration -- the audit greps for it with comments stripped,
having once been satisfied by a script whose only `--seed` was a sentence in
its own header.

| script | what it registers |
|---|---|
| `mexp/exp/kvzip-needle-8k.sh` | head-to-head against KVzip, the query-independent baseline the PAT review asked for three times, at 32x and 128x -- where THIS paper operates. Our transcription of the authors' scoring rule in our harness, at a matched budget |
| `mexp/exp/kvzip-inrange-8k.sh` | the same three arms at 2x, 4x, 8x -- where KVZIP operates, its README claiming 3-4x with minimal degradation. Reporting only the first script would repeat the criticism the review made of the H2O table: a method evaluated outside its stated range collapses by construction and says nothing. Also carries the timing reference, whose limits its header states |
| `mexp/exp/kvzip-selector-kappa16.sh` | the selector-only arm of that table at the DEPLOYED detector bandwidth. The two scripts above ran it as `digk64`, whose `k64` is the bandwidth kappa=64, not the shipped kappa=16; its row therefore disagreed with the Section 6.1 selector numbers (`dig_r64`, kappa=16) for the same quantity. This run is the one the table prints |
| `mexp/exp/kvzip-random-floor.sh` | the floor row of the same table: `imp_random` keeps top-m by a random score at the same budget, so the other rows have a scale. It exists to answer the cheapest attack on a transcribed baseline -- "your port is broken" -- because a broken port cannot score 24/24 at 2x on a metric where blind selection scores 0. Not a baseline anyone proposed |
| `mexp/exp/ruler-fallback-ablation.sh` | what the overflow fallback contributes, measured by deleting it: three arms back to back on engine-fused (the tree the n=50 RULER arms were measured on), differing only in `SGLANG_DEBUG_VESTIGEKV_NO_OVERFLOW_FALLBACK` and the truncation policy. Deleting it is an ablation, not a configuration -- the buffer sits behind an uncapped admission rule, so an overflow is the certificate reporting that a step cannot be served sparsely, and the fallback is the answer to that report |

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
