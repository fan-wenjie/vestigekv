---
name: negative-control-arms-rope
description: Index to the RoPE / negative-mechanism control arms that anchor the ICBINB taxonomy — each mirrors a NoPE experiment on a RoPE model (or a broken mechanism) to show the effect is NoPE-exclusive. Run any arm to re-check exclusivity.
---

# Negative-control arms (RoPE / mechanism controls) — ICBINB taxonomy anchor

**PREREG**: none (grouped controls) | **Output**: out/dsv2_*.json, out/dissect_dsv2.json, out/ea_dsv2_matched.json, out/cluster_8192.json, out/derot.json, out/learnedsel_8192.json, out/proj_8192.json, out/invbranch_8192.json, out/dilution_*.json | **Phase**: E

## What it measures
This is an index skill, not a single experiment. Each control mirrors a NoPE-MLA
experiment on a RoPE model (DeepSeek-V2-Lite, arch `dsv2`) or replaces the real
mechanism with a broken one (de-rotation, dedup, clustering, learned selector,
projection, inverse branch, dilution nuke), to demonstrate the VestigeKV signal
is NoPE-exclusive — it should vanish or fail on the control. The taxonomy
(classes A-D) lives in `icbinb/NOTES.md` and `icbinb/routes_measured_and_closed.tex`.

## Frozen decision rule (if pre-registered)
No single frozen bar; each control mirrors its NoPE counterpart's rule to show
exclusivity (the RoPE arm should NOT reproduce the NoPE effect, or the broken
mechanism should fail). See the mirrored PREREG for each arm's bar and the
class assignment in `icbinb/NOTES.md`.

## Reproduce
```bash

# Representative arm — the DSV2 (RoPE) eviction-signal exclusivity check via e2e_dsv2.py:
python e2e_dsv2.py --seq-len 8192 --n-docs 0 --needle-trials 12 --gpu-gib 26 \
  --seed 11 --eastats out/eastats_dsv2.pt --layers matched \
  --ops ea_orig,ea_centered,imp_only,recent --out out/ea_dsv2_matched.json

# Other control arms in this family (bare scripts read the cached out/dsv2 + out/main2 dumps):
python derot.py            # de-rotation: RoPE vs NoPE rotation inflation -> out/derot.json
python clustererr.py       # within-cluster logit-spread control (prints RoPE vs NoPE table)
# plus the e2e.py mechanism controls: --ops nuke,recent (dilution), proj_r (projection),
#   inv_drop,inv_score,inv_fill (inverse branch), learned_sel (learned selector),
#   cluster_n* (clustering) — see queue.jsonl for the exact --out paths
#   (out/dilution_kimi.json, out/proj_8192.json, out/invbranch_8192.json,
#    out/learnedsel_8192.json, out/cluster_8192.json).
```
Requires: Kimi Linear checkpoint in the HF cache; the DSV2 arms also need the
DeepSeek-V2-Lite checkpoint (arch `dsv2` in `extract.py`) and its `eastats`
tensor. HF-harness — does NOT need the mini-sglang package; single-GPU.
`e2e_dsv2.py` uses `--gpu-gib` (not `--gpu-expert-layers`) and `--layers matched`.

## Verdict
Index, not a single result: these arms anchor the ICBINB negative-results
taxonomy by showing exclusivity — the RoPE model and the broken-mechanism
controls do not reproduce the NoPE-MLA vestigial eviction signal. For per-arm
verdicts and the class A-D assignment of each exhibit, read `icbinb/NOTES.md`
(classification only, ICBINB write-up deferred per user directive) and
`icbinb/routes_measured_and_closed.tex`.
