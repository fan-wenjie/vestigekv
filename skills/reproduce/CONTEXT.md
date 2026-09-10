# VestigeKV — session handoff / context restore

Read this first. It lets a fresh assistant session continue the project
(paper writing, code experiments) on a new machine without re-deriving what
is already settled. Pair it with REPRODUCE.md (how to run) and RULES.md
(standing constraints).

## What the project is

Training-free NoPE-MLA KV-cache compression for Kimi Linear 48B, ported
into sglang as attention backend `vestigekv_mla`. Target: ICLR 2027
submission + an upstream sglang PR. Two tiers, both GPU-resident (the
advantage is decode SPEED, not memory): tier-1 sidecar-residual eviction
(frozen at prefill) into an archive; tier-2 certified per-step recall
(rank-64 sketch + conformal z-certificate + entropy gate) that keeps the
archive reachable so no row is ever dropped. Cache rows stay bit-exact
bf16; only the 16-bit scan index metadata is reduced-precision.

## Repository layout (this repo = the experiment repo)

- `engine/` — vendored sglang submodule; the `vestigekv_mla` backend lives
  in `engine/python/sglang/srt/layers/attention/vestigekv*`. The submodule
  gitlink points at the single-commit PR branch.
- `mexp/` — serving-path experiment drivers (throughput/latency plots,
  MAUVE via HTTP, quality runner). Low-coupled to sglang (text in, scores
  out).
- `harness/` — HF-forward research harness (e2e.py + extract.py) for the
  ALGORITHM claims (needle, baselines, RoPE-collapse, ablations, bpb).
- `tools/` — weight daemon, health check, config checker.
- `results/` — the published headline data (jsonl + figures + manifest).
- `skills/` — experiment catalogue (prereg*/m*) + this reproduce package.
- `docs/` — INFRA_OPTIMIZATIONS.md, whitepaper/.
- The paper source is NOT in this repo (kept in a separate materials dir);
  it `\input`s numbers.tex whose macros trace to results/ and harness runs.

## Current status (as of this handoff)

- **PR branch** `vestigekv` = single squashed commit `09c39628` (tree ==
  `vestigekv-dev`). Contains the 7-kernel fused in-graph recall, per-batch-class
  grid sizing, operand reuse across the calibration ladder, the NoPE-precondition
  guard, the provisional-index eigh removal, the projection-form fused sigma
  kernel (+ radix histogram), the fused tier-2 operand builder, and the
  prefill sigma spec fix, and the in-kernel strided sigma read (no [T,576]
  gather intermediate). All bit-parity gated; registered kernel tests included.
  The last of these was verified at kernel level only (bit-parity +
  microbenchmark + the CPU unit suite, 87 vestigekv tests): the two-node
  end-to-end greedy gate could not run because node1 was already down.
- One optimization was REVERTED (a host `kept_len` mirror) after its STATS
  assertion caught a drift at a block close — see RULES.md "block-close safety".
- **Paper**: main text 8.92 pages (ICLR initial-submission limit is 9), 0
  errors / 0 undefined / 0 overfull. Abstract ~285 words. A simulated peer
  review was applied in full at the text level (consistency, grammar,
  de-AI-isms); its three "Major" items requiring new experiments were not
  run except the second-checkpoint one, which WAS done (below).
- **Instruct (post-training) result — completed**: the mechanism survives
  post-training. Static anatomy branch/content row-norm peak 3.482 (Base)
  -> 3.490 (Instruct), per-layer differences under 1%; deployed two-tier
  needle retrieval 1.00 at both 32x and 128x on both checkpoints;
  recent-only floor 0.00 on both. Recall LOAD does not grow: median
  fetch/query 39.9 -> 30.9 rows, extra attended fraction 0.97% -> 0.75%,
  at identical z_p and gate rate. Written into the abstract, Limitations
  and Appendix `app:sft`; data in `results/harness_*`.

## Known open items

- **Instruct serving measurement (blocked)**: throughput and the latency
  curve for the Instruct checkpoint were never run, because node1 became
  unreachable (its SSH service on the mapped port disappeared; the host
  answers on other ports, so it is a platform-side container/port change).
  The mechanistic argument is already in the paper (recall load falls), but
  the direct serving numbers are outstanding. On a fresh two-node setup,
  run the README metric-1 and metric-2 commands against the Instruct
  checkpoint and compare with the Base numbers in `results/`.
- Deployment-only engineering threads (neither changes any paper number):
  the prologue-merge kernel is under-occupied at bs=1 (~30us), and the
  tier-2 build could become a single on-GPU persistent kernel (removing
  host orchestration and scalar readbacks).
- Not run (simulated-review suggestions): a standard long-context suite
  (RULER/LongBench subset), and multi-launch statistics for the serving
  curve (currently one launch per arm, with the +/-15% fixed-term variance
  handled by reading Delta-vs-4k).

## Key attributions (measured, in DEFECTS.md)

- bs=1@64k throughput ~0.98x is the steady-state attention share, not the
  startup transient (a 4k-decode request runs ~33s; the transient is tens
  of ms). Confirmed by per-step itls (uniform gap, no periodic block-close
  spikes) and by nsys (vk attention kernel 13us vs stock 101us — 7.7x
  FASTER; the recall machinery scan+prologue ~114us roughly equals the 89us
  attention saving at 64k).
- Steady state has NO device->host sync: the kept_len append is an in-graph
  kernel store/load (GPU->GPU), the kv_indptr prefix-sum is a device cumsum,
  the slot list is cached. The 22% GPU idle at bs=1 is the PP=2 pipeline
  bubble (stock has it too), not a vestigekv D2H.

## How to continue

- Paper edits: the materials dir holds main.tex + sec_eng_serving.tex +
  numbers.tex; rebuild the two zips (anon + named) with the packaging
  recipe. Every data macro must trace to a results/ file or a harness run.
- New experiments: write the command into README FIRST, audit it, then run
  it verbatim (this is a hard project rule — see RULES.md). Data must come
  from the PR-equivalent tree (the launch guard enforces this).
- Health before any launch: `bash tools/health_check.sh`; only act on FAIL.
