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
