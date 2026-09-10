# Experiment catalogue

Reproduction code lives in two places: the HF-forward algorithm harness
(`harness/e2e.py`, `--ops` selects the measurement) for the selection and
recall claims (needle, baselines, RoPE-collapse, ablations, bits-per-byte),
and the serving stack (`mexp/`, `engine/`) for the latency, throughput, and
serving-path quality numbers (`results/`).


Every experiment's skill file (under `skills/`) carries the exact reproduce
command, the frozen decision bar, and the verdict; the dated
pre-registration files ship with the paper's supplementary materials.

<!-- INDEX_START -->

The delivered method is **sidecar-residual exact-keep eviction** (tier-0) plus an
optional **GPU-resident archive + trigger** (tier-1/2). No training, no
quantization anywhere (barred by rule, `PREREG27`). Almost every Phase-A quality
number comes from one script, `e2e.py`, driven by its `--ops` registry.

### Phase A — core compressor (HF harness)

*A.0 — premise and operator foundation*

| experiment | prereg | verdict | skill |
|---|---|---|---|
| Premise: does NoPE make MLA more compressible? (merge vs evict) | PREREGISTRATION | reverse-null fired — merge class (fft/svd/avgpool) 0.00; only exact keep/evict works | [skill](skills/prereg00-premise-merge-vs-evict.md) |
| Sparse impulses + low-pass residual as cache format | PREREG2 | low-pass adds nothing on top of selection → pure exact-keep eviction | [skill](skills/prereg02-hybrid-impulse-lowpass.md) |
| Decouple detector bandwidth kappa from storage budget m | PREREG3 | renorm-fill rejected e2e; kappa=16 (<=16k) / 64 (>=32k) law | [skill](skills/prereg03-decouple-bandwidth-budget.md) |
| Branch-asymmetric anomalous cache (content vs sidecar split) | PREREG4 | rank-1 + uncertainty-boundary refuted → dual-tier design (archive off attention path) | [skill](skills/prereg04-branch-asymmetric-cache.md) |

*A.1 — mechanism autopsy ("what actually makes Kimi compressible?")*

| experiment | prereg | verdict | skill |
|---|---|---|---|
| Key-space cluster merge, de-rotation, Granite union | PREREG6 | NoPE-exclusivity premise FAILED 3/3 mechanistic tests — gap is real but not positional encoding | [skill](skills/prereg06-mechanism-cluster-derot-granite.md) |
| Selector alignment, QK read-out rank, feature-axis proj | PREREG7 | read-out full-rank 512 → feature-axis compression 0.00; covering rejected | [skill](skills/prereg07-readout-rank-selector-align.md) |
| Dedup regime + sidecar long-context validation (32k/65k) | PREREG8 | dedup theorem has no application regime; sidecar selector PASS zero gap at 8k/32k/65k | [skill](skills/prereg08-sidecar-longcontext-dedup.md) |

*A.2 — champion selector, baselines, streaming, bpb*

| experiment | prereg | verdict | skill |
|---|---|---|---|
| Learned static selector (MLP + hinge) | PREREG9 | rejected — offline +0.406 → e2e -0.71 (10th offline/e2e inversion) | [skill](skills/prereg09-learned-static-selector.md) |
| Baselines: H2O / SnapKV / StreamingLLM | PREREG17 | baselines structurally collapse (0.00); VestigeKV holds 1.00/0.88/0.67 | [skill](skills/prereg17-baselines-h2o-snapkv-streaming.md) |
| LM-loss / bits-per-byte characterization | PREREG21 | effectively lossless — ~0.0006 bpb at 32x | [skill](skills/prereg21-bpb-lm-loss.md) |
| Streaming / iterated compression | PREREG23 | tax-free (0.00 gap); constant-m is the serving default | [skill](skills/prereg23-streaming-iterated-compression.md) |
| Table gap-fill + n-boost | PREREG26 | pure measurement (no bar) | [skill](skills/prereg26-gapfill-nboost.md) |
| batch>1 quality | PREREG_batch | no degradation from batching | [skill](skills/prereg-batch-quality.md) |

*A.3 — archive trigger line (tier-2 recall)*

| experiment | prereg | verdict | skill |
|---|---|---|---|
| Trigger-recall, sidecar only | PREREG10 | rejected — recall@4 0.489 | [skill](skills/prereg10-trigger-recall-sidecar.md) |
| Trigger v2 + rank-r query-subspace sketch | PREREG11 | partial — r=32 held 0.818 | [skill](skills/prereg11-trigger-v2-sketch.md) |
| Confirmatory r=64 on fresh data | PREREG12 | stays partial — fresh 0.812 | [skill](skills/prereg12-trigger-confirmatory-r64.md) |
| Certified Cauchy-Schwarz trigger | PREREG13 | rejected — sound but fires 99.5% | [skill](skills/prereg13-certified-cauchy-schwarz-trigger.md) |
| Gaussian-tail + conformal trigger | PREREG14 | partial — conformal 94%@1.9% in-dist, OOD fails | [skill](skills/prereg14-gaussian-conformal-trigger.md) |
| ArkVale page digests, NoPE vs RoPE | PREREG15 | real finding = NoPE layer-uniformity (0.672 vs RoPE 0.024) | [skill](skills/prereg15-arkvale-page-digests.md) |
| Query-only (SVM) trigger | PREREG16 | not viable — dominated 7-20x by the index scan | [skill](skills/prereg16-query-only-svm-trigger.md) |
| Entropy trigger + cascade gate | PREREG18 | viable in-dist (21%@0.93); OOD transfer fails | [skill](skills/prereg18-entropy-cascade-gate.md) |
| Self-calibration + e2e recovery (two-tier) | PREREG19 | recovery 1/128 0.67 → 1.00; the shipped tier-2 | [skill](skills/prereg19-selfcal-e2e-recovery-twotier.md) |
| Index rank vs trigger-rate sweep | PREREG20 | knob pays — sweet spot r=128-192 | [skill](skills/prereg20-index-rank-sweep.md) |
| LSE-compatible gate | PREREG24 | PASS — indistinguishable from entropy gate | [skill](skills/prereg24-lse-compatible-gate.md) |

### Phase B — mini-sglang port validation (native stack == HF harness)

| experiment | prereg | verdict | skill |
|---|---|---|---|
| M3 parity gate (native vs HF) | none | bar max\|dNLL\|<=2e-3, median<=5e-4 | [skill](skills/m3-parity.md) |
| M4 tier-1 needle in the deploy loop | none | VestigeKV tier-1 in KimiEngine, L=8192 | [skill](skills/m4-needle.md) |
| M5 GPU-resident tier-2 decode (8k + 32k) | none | tier-2 recovery in the decode loop | [skill](skills/m5-tier2-decode.md) |
| M6 deployment-path bpb | none | bits-per-byte on the production cache form | [skill](skills/m6-bpb.md) |
| M7 MAUVE generation quality | none | MAUVE over 16 contexts, 3 text sets | [skill](skills/m7-mauve.md) |
| V4-M1 native-vs-HF parity (fp32 referee) | none | KimiKDA vs HF KimiDeltaAttention | [skill](skills/v4m1-native-parity.md) |
| V4-M2 full-stack needle (native) | none | full stack reproduces the validated engine | [skill](skills/v4m2-fullstack-needle.md) |
| V4-M3 single-machine serving smoke | none | native server smoke | [skill](skills/v4m3-serving-smoke.md) |
| V4-M4 30-min stress | none | fairness / leaks / quality under load | [skill](skills/v4m4-stress.md) |

### Phase C — dual-machine serving (local 0-19 + remote 5090 20-26, gloo/CPU)

| experiment | prereg | verdict | skill |
|---|---|---|---|
| Bandwidth microbenchmark (T up to 524k) | PREREG30 | the ~4x byte account is realizable | [skill](skills/prereg30-bwbench.md) |
| 512k decode wall-clock (stock vs vestige) | PREREG31 | 4.21x speedup (0.472 → 0.112 s/step); 99.8% trigger @512k | [skill](skills/prereg31-decode512k.md) |
| 512k end-to-end quality | PREREG32 | PASS 5/5; 99.8% trigger but 0.024% fetch — no degeneration | [skill](skills/prereg32-quality512k.md) |

### Phase D — V5 wall-clock (single-GPU, all-GPU)

| experiment | prereg | verdict | skill |
|---|---|---|---|
| Step profiling | none | MoE GEMM dominates the 82 ms 8k step | [skill](skills/v5-profile-step.md) |
| M1 SDPA-fused attention gate | none | fused attn preserves recovery + faster | [skill](skills/v5m1-fused-attn.md) |
| M2 MoE decode fast-path gate | none | gather+bmm preserves recovery; step 82 → 40 ms | [skill](skills/v5m2-moe-fastpath.md) |
| M3 batch-scaling projection | none | ratio rises with batch from measured components | [skill](skills/v5m3-batch-projection.md) |
| Cost of removing the topj/512 cap (LAST) | PREREG33 | cap removal nearly free @batch=1, <=32k; keep cap as the bounded-fetch guarantee | [skill](skills/prereg33-capcost.md) |

### Phase E — closed / negative routes (ICBINB material)

| experiment | prereg | verdict | skill |
|---|---|---|---|
| Intra-index cascade (sidecar shortlist) | PREREG25 | refuted — recall 0.10-0.79; ICBINB Class C | [skill](skills/prereg25-intra-index-cascade.md) |
| Cascade hard-target recall | PREREG27 | refuted both legs → lever closed; no-quant rule frozen here | [skill](skills/prereg27-cascade-hard-recall.md) |
| Per-layer self-calibrated cascade enable | PREREG28 | PASS but value decays with length; ships default-off | [skill](skills/prereg28-perlayer-cascade-enable.md) |
| Hierarchical page index | PREREG29 | refuted — admissible but vacuous; ICBINB Class D | [skill](skills/prereg29-hierarchical-page-index.md) |
| RoPE / negative-mechanism control arms | none | the exclusivity controls anchoring the ICBINB taxonomy | [skill](skills/negative-control-arms-rope.md) |

<!-- INDEX_END -->

## Tier-1 ablation on the standard sglang stack (the serving floor)

> **Protocol rule (2026-09-06): all future experiment data is measured WITH
> the recall tier; tier-1-only runs are ablations.** Everything below is the
> tier-1 floor -- superseded as headline data once the port wires recall.

The wall-clock numbers above (PREREG31: **4.21×** at 512k) come from the
`mini-sglang` native decode microbenchmark, where MLA attention dominates the
step. We also ported VestigeKV into **standard sglang** (`vestige_mla` attention
backend; see the fork's `VESTIGEKV_PORT.md`) and re-measured end-to-end on the
production hybrid-MoE serving stack. All numbers here are produced by sglang's
own `python -m sglang.bench_one_batch_server` (output throughput, `--output-len
32`, `--skip-warmup`); the orchestration scripts only launch the server and flip
the A/B arm. The one exception is the 524k point, taken with a small custom
streaming client (`mexp/probe524.py`) because the bench client times out on the
~13-min 64-chunk prefill — flagged here per the "data via sglang's bench module"
rule.

**Config (both arms identical):** dual-machine pipeline parallel (RTX PRO 6000
+ RTX 5090, `SGLANG_PP_LAYER_PARTITION`), CUDA graph **on for both arms**,
bf16, no quantization, VestigeKV **uncapped** (tier-1 sidecar eviction ρ=1/32,
no topj cap). FULL = attend all rows (== baseline); VESTIGE = attend the kept
set. Arms switch on a running server via the `/tmp/vestigekv_full` flag, so both
replay the identical graph and code path.

### The recall fetch cap (topj): **default is UNCAPPED — a cap must be set explicitly**

> **Fool-proof by design: the serving default fetches the FULL
> fired recall set (`SGLANG_VESTIGEKV_TOPJ=-1`). There is no silent partial
> default — if you want a bounded fetch, YOU state the bound.** This is
> deliberate: the cap is the one knob that trades recall completeness for a
> worst-case latency bound, and the engineer setting it should know what it
> bounds (fetch ≤ topj × num_heads rows / step / layer) and why — see
> PREREG33/34 and the paper's "recommended configuration".

**What the cap does.** Every decode step, the tier-2 index scans the evicted
rows and *fires* the ones whose certificate says the current query might need
them; the fired rows are fetched back into that step's attention. Uncapped,
the fired set is whatever the certificates produce — usually small, but with
**no upper bound** (observed worst case: 3,983 rows in one step, PREREG30
telemetry). `topj` caps the fetch to the **top-j scored rows per head**,
giving the hard guarantee *fetch ≤ topj × num_heads rows / step / layer*.
That bound is what makes the method non-degenerate by construction: no
adversarial context can push a step back toward attending the whole cache,
and in the host-offload variant it bounds per-step PCIe traffic.

**Recommended cap: `topj = 16`, with the evidence:**

| leg | capped (16) vs uncapped | source |
|---|---|---|
| worst-case fire / step | **296 vs 3,983 rows (13×)** — the bound the cap buys | PREREG30/33 |
| needle recovery | 4/4 vs 4/4 — identical (the needle row always sits in the top-16) | PREREG33 |
| generation | **bit-identical text**; MAUVE unchanged | PREREG34 |
| likelihood (bpb vs engine) | +0.145% vs +0.102% — cap costs **0.04 pp**, inside the ±0.001-bpb neutral band | PREREG34 |
| wall clock (batch=1, ≤32k) | 8k: 40.5→40.1 ms (−1%); 32k: 40.3→41.8 ms (+4%) — cap ≈ free | PREREG33 |

Note a property classic sparse attention does not have: with recall on,
per-step attention volume is *query-dependent and variable* (0 to the
observed 3,983-row worst case). The cap is therefore the latency-determinism
switch, not just an optimization -- capped, per-step work is hard-bounded;
uncapped, the *theoretical worst case is degeneration to naive MLA plus the
scan overhead* for that step (quality unharmed -- the extra rows are exact;
steps independent). Measured worst fire is 3,983 rows, an observation, not a
bound; an uncapped SLO must quote naive-MLA-plus-scan as its per-step worst
case.

So j=16 sits exactly at the knee: large enough that every measured retrieval
target lands inside it (quality legs all tie), small enough that the fetch
bound tightens 13×. Honest scoping from PREREG33: at batch=1/short context
the cap does **not** buy wall clock (attention is a small step fraction
there); its value is the worst-case bound, which matters precisely in the
long-context + large-batch regime where attention dominates.

```bash
# production launch with the recommended bounded-fetch cap (explicit opt-in):
SGLANG_VESTIGEKV_TOPJ=16 MODEL_PATH=<kimi-ckpt> \
    mexp/launch_vestige_server.sh <node_rank> <dist_addr:port>

# research / quality-ceiling runs (the default; equivalent to omitting it):
MODEL_PATH=<kimi-ckpt> mexp/launch_vestige_server.sh <node_rank> <dist_addr:port>
```

**All recommended values live in one file: [`config/recommended.json`](config/recommended.json)** —
each entry carries `value` / `why` / `evidence` / `class`. `operational` entries
(partition, backends, prefill chunking, NCCL tuning) are read by
`mexp/launch_vestige_server.sh` directly, so no magic number appears in a launch
line; `algorithm` entries (ρ, **topj**, r, sinks, recent window) are a *record*,
never auto-applied — the engineer types them, per the fool-proof rule above.
`tools/check_recommended_config.py` fails CI-style whenever the JSON and the
sources drift (and aborts loudly if it cannot resolve a check subject).

All benchmark numbers in this README are the uncapped default (`-1`), so a
capped deployment can only fetch less, never more, than what is measured here.

### Speedup vs context length (batch=1 floor, uncapped)

| context S | FULL tok/s | VESTIGE tok/s | speedup |
|---|---|---|---|
| 4,096   | 115.9 | 133.0 | 1.15× (small-S warmup noise) |
| 8,192   | 136.8 | 138.1 | 1.01× |
| 16,384  | 133.5 | 139.8 | 1.05× |
| 32,768  | 131.4 | 136.1 | 1.04× |
| 65,536  | 127.4 | 134.9 | 1.06× |
| 131,072 | 114.2 | 133.7 | 1.17× |
| 262,144 | 101.4 | 130.2 | **1.28×** |

The speedup rises monotonically in log(S). (At 524,288 the FULL arm OOMs the
32 GB node while the VESTIGE arm keeps serving — an observed capacity
asymmetry, recorded in `out/curve_ab.csv` but excluded from the curve: its
throughput was measured off the standard bench path, see `mexp/probe524.py`.)

### Tier-1 ablation: speedup vs batch (S=64k)

The second, orthogonal axis of the same Amdahl account: batch amortizes the
fixed MoE/KDA/PP cost while the attention read scales with batch, so the
attention fraction — and the end-to-end speedup — rises with batch. Same
protocol (interleaved arms, `bench_one_batch_server`); round 1 warms each
CUDA-graph bucket and is discarded, round 2 reported. Partition 23,4; the KV
pool (1,206,077 tokens) admits at most batch 16 at S=64k (16×65,536=1,048,576).

| batch | FULL tok/s | VESTIGE tok/s | speedup | Amdahl bound |
|---|---|---|---|---|
| 1  | 123.7 | 138.1 | 1.12× | 1.09× |
| 2  | 128.4 | 140.1 | 1.09× | 1.12× |
| 4  | 203.5 | 233.1 | 1.15× | 1.19× |
| 8  | 288.2 | 347.0 | **1.20×** | 1.27× |
| 16 | 391.6 | 490.7 | **1.25×** | 1.35× |

Final protocol: both ranks code-verified, a full proof round with the row-set
invariant enabled (0 violations on both ranks), one bucket-warmup round
discarded, report round quoted. Data: `out/batch64k_final.csv` (earlier
`batch64k_ab.csv` runs predate the node1 code-sync fix and are superseded).

**Why 1.28× here vs 4.21× native — Amdahl, not a regression.** Deriving the
decode-step cost from the FULL-arm throughput gives a context-independent
`7.23 ms` (20 KDA layers + MoE + cross-machine PP) plus `~10 ns/attended-row`.
Only 7 of Kimi Linear's 27 layers are MLA, so attention is 1% of the step at 8k
and 27% at 262k; the end-to-end speedup is Amdahl-capped by that fraction. A 4×
end-to-end decode on this stack would need attention to be ~77% of the step
(~2.5M-row context), which the hardware cannot hold — so the production ceiling
is memory (512k OOM), and the 4.21× native figure is the attention-dominated
microbenchmark regime. Both are true; they measure different things.

### Best PP layer split

Sweeping `SGLANG_PP_LAYER_PARTITION` (S=131072, batch=1) shows decode throughput
is nearly invariant to the split (FULL 101–106 tok/s) — the two GB202 cards are
near-equal speed — but the KV pool (min across ranks) is very sensitive:

| split (node0,node1) | pool tokens | speedup | 256k max batch |
|---|---|---|---|
| 24,3 | 804,452 | 1.33× | 3 |
| **23,4** | **1,206,077** | 1.32× | **4** |
| 22,5 | 862,851 | 1.30× | 3 |
| 21,6 | 525,137 | 1.28× | 2 |

`23,4` gives the largest pool (2.3× the memory-driven `21,6`), so it is the
partition to use when batch or context headroom matters.

## Script inventory (current)

| where | what |
|---|---|
| `mexp/` | active, sglang-only: data collection + what sglang lacks (see `mexp/README.md`) |
| `tools/` | health check, config consistency check, `serving/` dual-machine launch harness |
| `archive/probes/` | frozen pre-port HF-harness probes (`e2e.py` and friends; provenance of Phase-A numbers) |
| `archive/gates/` | frozen mini-sglang-era gate scripts (m3–m8, v4*, v5*; dependency retired) |

## Pre-registration discipline

`PREREG2.md`–`PREREG33.md`: each states the comparison, the decision rule, and
what every outcome means — frozen before the data. Retracted predictions are
kept, not deleted. See `DELIVERABLE.md` for the consolidated results and the
refusal table (closed routes), and `icbinb/` for the negative-results corpus.
