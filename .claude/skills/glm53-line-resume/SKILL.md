---
name: glm53-line-resume
description: Resume the GLM-5.3-Flash DSA-selector line (VestigeKV). Load before touching the glm53 queue, the vk_select kernel, or any claim about the 1.0058x arm — it carries the settled numbers, the refutations, and the three checkouts.
---

# Resuming the GLM-5.3 line

Read this before running a job, quoting a number, or editing the selector. The
authoritative record is `README.md` in this repo (the launch register); this file
is the index and the don't-redo list.

## The three checkouts — do not mix them up

| path | branch | role |
|---|---|---|
| `/home/user/vestigekv-glm` | `glm53-line` | register (`README.md`), queue (`mexp/glm53/queue.jsonl`), results. THIS repo. |
| `/home/user/vestigekv-wt/engine-glm521` | `glm53-v0521` | **the served engine**. Every glm53 job runs this tree. |
| `/home/user/vestigekv-wt/engine-glm521-pr` | `glm53-pr` | PR hygiene only. Never served. |

`/home/user/vestigekv` + `/home/user/vestigekv/engine` are the **Kimi** line. Its
queue has been drained since 2026-10-02; its `results/kimi/` holds ~140 untracked
jsonl files that are not ours to commit.

## Where the line actually stands

- **Latency crossed.** The forked selector (`SGLANG_DSA_VK_SELECT=1`, q=0) is
  **1.0058x** end-to-end vs DSA at 4k/124k: tput 92.421 / 92.414 / 92.426 / 92.318
  against a floor of 91.8617 +- 0.0322 (n=6), 16.5 sigma. Kernel is 2.1x DSA's
  rank top-k, flat in context, holding through bs=8.
- **Quality did not.** `ruler_cwe` @64k: vk 0.758/0.702/0.712/0.784/0.706 against
  DSA 0.960/0.956/0.920/0.954 — gap **0.215 +- 0.043** (t=11.4), complete
  separation. Every `niah_*` ties DSA at 1.000.
- **So `glm-line-order-and-paper-rule`'s crossover is NOT established.** The
  defensible claim is a speed/quality trade with a workload dependence.
- **Cause, measured not inferred.** The arm finds the right words and then repeats
  itself: at 64k it duplicates items in 26/50 cwe docs vs DSA's 9/50 and derails in
  18/50 vs 5/50. Recency telemetry (`SGLANG_DSA_VK_SELECT_TELEMETRY=1`) measured the
  deficit `dsa_frac - sel_frac` on recent groups at **+0.165 @64k vs +0.089 @4k**,
  worst layer 7 selecting the newest group 42.3% where an exact top-k takes 76.9%.

## Do not redo these

- **W=16 recency window: REFUTED.** Implemented and correct (`SGLANG_DSA_VK_SELECT_RECENT_W`,
  default 0). cwe 0.738/0.694/0.668, diff vs W=0 **-0.032, t=-1.24** — nothing.
  Do not retry at larger W: W=25 reaches 91.6% of a deficit W=16 already took
  73.9% of, against a measured response of zero.
- **Nine structural predictions refuted** (register has the list). The only two
  that ever worked were deletions of work: skipping DSA's top-k, and removing a
  redundant scan of the global score row.
- **Norm / low-rank on NoPE GQA (Nemotron-3-Nano): closed.** Centered effective
  rank 63-93 of 128; sound prune rate **0.0000%**; norm-as-selector at or below
  random. `sigma_conditions.py` already carried the sqrt(d) argument — search it
  before re-deriving.
- **Phase probe: bounded null, no range.** niah is 1.000 on both arms, and decoys
  do not help (1.0000 at 4/16/64/256). Any niah-family phase gap is under ~10
  points. A probe with range must target **aggregation**, not lookup.

## Hard-won operational rules

- **Never edit the served tree while a runner lives.** Violated twice; once it put
  two TP ranks on different JIT kernel source and voided a run.
- **A quality draw is the only detector after a selector change.** Both the
  trash-slot incident and the W=16 `+inf` latch reproduced every counter with
  `errs=0` and no traceback. Always ride a `niah_multikey_3` canary along.
- **Never `pkill -f` a pattern that appears in your own command** — it kills your
  shell. Read `/proc/<pid>/cmdline`, skip `shell-snapshots`.
- **SM120:** attention backend must be `triton` (no trtllm/flashmla FMHA kernel).
  JIT needs `CUDA_HOME=/usr/local/cuda` or nvcc can't find its toolchain.
- **TP=2 hangs at parallel init** for Nemotron-H on this box; TP=1 serves fine.
- `sgl_kernel/top_k.py` asserts `topk == 2048` at all three entry points, so the
  index_topk budget axis is closed on this kernel package.

## How to run anything

Copy an existing row from `mexp/glm53/queue.jsonl` (each client's args fail
differently, two fail silently), append, verify the runner's own view, register it
in `README.md` with the acceptance **stated before the data**, then:

```bash
cd /home/user/vestigekv-glm && nohup .../python mexp/glm53/queue_runner.py > results/glm53/runner_X.log 2>&1 &
```

Commit identity is always `-c user.name=fan-wenjie -c user.email=fanwj@mail.ustc.edu.cn`.

## Open decisions (mine to propose, not to take)

1. **Aggregation phase probe** — align every occurrence of a cwe target word to one
   phase, 4 phases x 2 arms. Two probes already failed the same way (built in the
   saturated family), so this design wants review before it runs.
2. **Keep or revert the W=16 kernel change.** Default-0 and harmless, but now dead
   code with two tests guarding it. Worth keeping only if the length-segmented
   design will reuse the machinery.
3. **Writing the line up as the trade** it measured, with the unmeasured phase
   question and the fixed-window limitation stated as limitations.
