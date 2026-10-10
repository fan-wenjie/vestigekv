# Reconstructing the machine

## What this was measured on

    GPU        2x NVIDIA RTX PRO 6000 Blackwell Workstation, 97887 MiB each
    arch       compute capability 12.0 -> SM120
    CUDA       /usr/local/cuda, release 13.0 V13.0.88
    python     3.12.14 at /home/user/.conda/envs/sglang-dev/bin/python
    torch      2.13.0+cu130 (cuda 13.0)

**SM120 is not incidental.** trtllm and flashmla have no FMHA kernel for it, so the
attention backend must be `triton`; five separate box patches are required and were
re-verified against v0.5.20 (register ~line 1090). A different arch invalidates the
latency numbers and possibly the patches.

## Repos and the exact commits

| repo | branch | commit | remote |
|---|---|---|---|
| vestigekv (register+queue) | `glm53-line` | `6d3ce06a55b0001ae54c29f9113e52492a8c1c1b` | `github.com/fan-wenjie/vestigekv.git` |
| sglang (SERVED engine) | `glm53-v0521` | `d3ac113872157d2903e8610fedda67c560493af2` | `github.com/fan-wenjie/sglang.git` |
| sglang (PR hygiene) | `glm53-pr` | `5cfc7ac455728f8e92a5296a1436967a48a2f94f` | same |

**The engine checkouts are git WORKTREES of the `sglang` submodule**, not clones:

    /home/user/vestigekv-wt/engine-glm521/.git
      -> gitdir: /home/user/vestigekv/.git/modules/sglang/worktrees/engine-glm521

`scripts/bootstrap.sh` reproduces that layout. Cloning sglang separately also works
but then `ENGINE=` in every queue row must be repointed.

## Weights (plain `hf download` into the default HF cache)

    nvidia/GLM-5.3-Flash-NVFP4                    191 G   <- the line's model
    nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4    19 G   <- the NoPE-GQA probe
    moonshotai/Kimi-Linear-48B-A3B-Instruct        92 G   <- the Kimi line, idle since 2026-10-02

Total ~302 G; 55 G was free here with all three present. Weights go to the default
HF cache and never to a shared mount. Kimi-Linear is droppable if space is short --
its queue has been drained since 2026-10-02, and `kimi-base-weights-deleted` records
the precedent for freeing it.

## Never install sglang

No pip/editable install. `PYTHONPATH=<engine>/python` only, in a deps-only env.
The queue runner sets this per job from the row's `ENGINE`; a client importing a
different sglang than the server it measures is the silent failure mode that rule
exists for.

## Two environment variables that are not optional

    CUDA_HOME=/usr/local/cuda    torch sets it to /usr, where nvcc then looks for an
                                 absent /usr/nvvm and the JIT build dies with
                                 "cicc: not found". Every queue row carries it.
    PYTHONPATH=<engine>/python   see above.
