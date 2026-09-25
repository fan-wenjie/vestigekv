---
name: prereg34-tier-wilson-boost
description: Boosts needle-trial counts behind tab:tier to tighten Wilson intervals (+recall 8k row 24→64 trials; 32k digk64 pooled 24→32; 65k digk64 single-seed→two-seed pooled 24); pure measurement, no acceptance bar; prefix-containment of the harness trial stream is verified, not assumed.
---

# PREREG34 — tab:tier Wilson n-boost

**PREREG**: this file | **Output**: results/harness_needle64_8192_{base_mirror,instruct}.json, results/harness_tier_{32768,65536}_seed{11,12}.json | **Phase**: A

## What it measures

Same instrument as PREREG19/PREREG26 (harness needle, `--arch kimi*`, HF
forward), larger n:

(A) 8k `+ recall tier` row of tab:tier (and the post-training-survival pair):
seed-0 runs extended from 24 to 64 trials for `kimi_base_mirror` and
`kimi_instruct`, ops `twotier,imp_only,recent`, rhos 32,128.

(B) 32k `digk64` cells: seeds 11 and 12 extended from 12 to 16 trials each,
pooled n=32 (was 24).

(C) 65k `digk64` cells: historically SINGLE seed 11, n=12 (PREREG08
sidecar_65536 at rho 32/128 + PREREG26 floor64_65536 at rho 64). Re-run
seed 11 and add seed 12, n=12 each, rhos 32,64,128 in one command, pooled
n=24. The rho list does not affect the trial stream (see below), so the
seed-11 re-run covers both historical files.

## Prefix containment (why re-runs reproduce old trials)

In `harness/e2e.py` the filler documents are sequential L-token chunks of the
same streaming fineweb-edu iterator (independent of `--seed`), and the only
seeded consumers are the per-trial draws `code` then `p`, taken in trial
order from one generator (`manual_seed(seed)`). Gate-1/2 consume no draws.
Therefore, for fixed (arch, seq-len), trial `ti` is bit-identical for any
`--needle-trials >= ti+1` at the same seed, and the per-trial stream does not
depend on `--rhos`/`--ops`.

This is VERIFIED, not assumed: run (A) at seed 0 must reproduce the first 24
trials of the surviving files `results/harness_needle_8192_{base,instruct}.json`
exactly. Check (after both archs finish):

```bash
python - <<'EOF'
import json
for arm in ["base", "instruct"]:
    old = json.load(open(f"results/harness_needle_8192_{arm}.json"))["rows"]
    newname = {"base": "base_mirror", "instruct": "instruct"}[arm]
    new = json.load(open(f"results/harness_needle64_8192_{newname}.json"))["rows"]
    old = [r for r in old if r["kind"] == "needle"]
    new = [r for r in new if r["kind"] == "needle" and r["trial"] < 24]
    key = lambda r: (r["trial"], r["op"], r["rho"])
    old, new = sorted(old, key=key), sorted(new, key=key)
    assert len(old) == len(new) == 144, (len(old), len(new))
    bad = [(o, n) for o, n in zip(old, new)
           if key(o) != key(n) or abs(o["d"] - n["d"]) > 0 or abs(o["base"] - n["base"]) > 0]
    assert not bad, f"{arm}: {len(bad)} mismatching rows, e.g. {bad[:1]}"
    print(arm, "containment OK: first 24 trials bit-identical")
EOF
```

If the check fails, ABORT: the extended runs do not contain the old trials,
the pooled counts are invalid, and the discrepancy is reported as-is.

## Frozen decision rule

No acceptance bar — pure measurement. Pooling rule (frozen, same as
PREREG26): pooled counts = all trials of every seed/run listed for a cell;
ALL runs enter the pool regardless of outcome (no per-seed or per-run
selection); the table and Wilson-appendix captions switch to the pooled
fractions with n stated; a same-cell difference between seeds beyond the
binomial band is reported as a discrepancy, not averaged away. Runs (B) and
(C) extend the PREREG08/19/26 cells; their historical pooled counts
(numbers.tex comments) are replaced by the new pooled counts wholesale.

## Reproduce

```bash
# (a) 8k, seed 0, 24 -> 64 trials (containment-verified against the n=24 files)
for ARCH in kimi_base_mirror kimi_instruct; do
  python harness/e2e.py --arch $ARCH --seq-len 8192 --n-docs 0 \
    --needle-trials 64 --gpu-expert-layers 18 --seed 0 \
    --ops twotier,imp_only,recent --rhos 32,128 \
    --out results/harness_needle64_8192_${ARCH#kimi_}.json
done

# (b) 32k digk64, seeds 11+12, 12 -> 16 trials each (pooled n=32)
for SEED in 11 12; do
  python harness/e2e.py --arch kimi --seq-len 32768 --n-docs 0 \
    --needle-trials 16 --gpu-expert-layers 18 --seed $SEED \
    --rhos 32,64,128 --ops digk64,recent \
    --out results/harness_tier_32768_seed${SEED}.json
done

# (c) 65k digk64, seeds 11+12, 12 trials each (pooled n=24; seed 11 re-run
#     covers the historical sidecar_65536 + floor64_65536 pair)
for SEED in 11 12; do
  python harness/e2e.py --arch kimi --seq-len 65536 --n-docs 0 \
    --needle-trials 12 --gpu-expert-layers 18 --seed $SEED \
    --rhos 32,64,128 --ops digk64,recent \
    --out results/harness_tier_65536_seed${SEED}.json
done
```

Requires: the pinned harness stack (ENV_PINS.txt: transformers==4.57.1,
torch==2.13.0+cu130, fla-core==0.4.0). The serving conda env runs
transformers 5.x, which BREAKS the 4.x-era modeling code, so the harness
uses an isolated target dir (env torch/fla fall through unchanged):
`pip install --target=$HOME/.cache/vestigekv/tf457 transformers==4.57.1 "huggingface-hub<1.0" "kernels<=0.9,>=0.6.1"`
then run with `PYTHONPATH=$HOME/.cache/vestigekv/tf457`. Checkpoints must be
plain dirs (see harness/extract.py). One free GPU (runs are sequential;
do NOT co-run with the serving line).

## Verdict

(to be filled after execution: containment check result, per-seed and pooled
counts per cell, any cross-seed discrepancy)
