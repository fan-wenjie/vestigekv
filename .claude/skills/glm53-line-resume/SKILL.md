---
name: glm53-line-resume
description: Resume the GLM-5.3-Flash DSA-selector line (VestigeKV) on this or another machine. Load before touching the glm53 queue, the vk_select kernel, or any claim about the 1.0058x arm — it carries the pinned commits, the settled numbers, the refutations, and the preflight.
---

# Resuming the GLM-5.3 line

Self-contained package: pinned commits, settled numbers, the don't-redo list, and
scripts to rebuild the checkouts on a fresh box. The authoritative detail is
`README.md` in this repo (the launch register, ~6k lines); this package is the index
over it so a cold session does not re-derive what is already measured.

## Read in this order

| | |
|---|---|
| **`reference/state.md`** | where the line stands, and the six things not to redo. **Read first** — it is the one that prevents repeated work. |
| `reference/environment.md` | machine spec, the three repos with exact SHAs, weights, the two env vars that are not optional |
| `reference/runbook.md` | served config, the selector's knobs, how to queue/verify/launch, how to verify a number before quoting it |
| `reference/pitfalls.md` | the rules that each cost something concrete |
| `scripts/bootstrap.sh` | rebuild the checkouts at the pinned commits (asks before acting, never clobbers) |
| `scripts/verify_env.sh` | preflight; run before trusting a measurement |

## On a new machine

```bash
bash scripts/bootstrap.sh          # checkouts at the pinned SHAs
# then reference/environment.md: conda env + hf download (~302 G for all three models)
bash scripts/verify_env.sh         # must print PREFLIGHT OK
```

## The 60-second version

Three checkouts, and mixing them up is the expensive mistake:

    /home/user/vestigekv-glm              glm53-line    register, queue, results
    /home/user/vestigekv-wt/engine-glm521 glm53-v0521   THE SERVED ENGINE
    /home/user/vestigekv-wt/engine-glm521-pr glm53-pr   PR hygiene, never served

`/home/user/vestigekv` + `/home/user/vestigekv/engine` are the **Kimi** line, drained
since 2026-10-02; its `results/kimi/` holds ~140 untracked jsonl that are not ours to
commit.

**Latency crossed at 1.0058x (16.5 sigma). Quality did not: cwe @64k is 0.215 ± 0.043
worse than DSA with complete separation.** So the paper rule's crossover is NOT
established and the defensible claim is a speed/quality trade. The cause is measured:
the arm finds the right words then repeats itself, because its own output is starved
by a prompt-calibrated threshold.

**Refuted, do not retry:** the W=16 recency window (t = −1.24), nine structural
optimisations, norm/low-rank screening on NoPE GQA (0.0000% prune), and the phase
probe (bounded null — niah is saturated, so it has no range).

**Two things worked, ever, and both were deletions of work.**

## Three open decisions

They are in `reference/state.md` with their arguments. None should be taken without
the user: the aggregation phase probe (two probes already failed the same way), keep
or revert the W=16 kernel change, and whether to write the line up as the trade it
measured.
