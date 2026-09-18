# engine/ — the serving stack, as a patch

This directory holds no engine source. It holds the two things needed to
reconstruct one exactly:

| file | |
|---|---|
| `setup_engine.sh` | fetches upstream sglang at the pinned commit and applies the patch |
| `vestigekv.patch` | the VestigeKV change: 32 files, ~10.3k lines added, 4 deleted |

```bash
./setup_engine.sh ~/sglang
export PYTHONPATH="$HOME/sglang/python:$PYTHONPATH"
```

## Why a patch and not a fork

A vendored fork of sglang would be large, would be mostly code this work did not
write, and would make it hard to see what the method actually is. A diff against
a named upstream commit answers that in one file: everything VestigeKV adds is
in `vestigekv.patch`, and everything else is upstream code you fetch yourself
from `github.com/sgl-project/sglang`.

The pinned base is `17ba2c2e7c7b81f31a8a9e693e7435ab262c16b4`. `setup_engine.sh`
verifies it fetched that exact commit and runs `git apply --check` before
touching anything, so a patch that no longer applies fails loudly instead of
half-applying.

## What the patch contains

```
python/sglang/srt/layers/attention/vestigekv/        the method
    eviction.py          tier 1: the sigma signal and the close window
    recall_tier.py       tier 2: the sketch, the certificate, the calibration
    scan_kernel.py       the per-row certified scan
    fused_prologue.py    the in-graph decode prologue
    sigma_fused.py       the fused tier-1 path
    batched_step.py  pack_csr.py  operand_fused.py  telemetry.py
    config.py  defaults.py
python/sglang/srt/layers/attention/vestigekv_mla_backend.py   the backend
test/registered/kernels/test_vestigekv_*.py          kernel tests
test/manual/test_vestigekv_equiv.py                  equivalence against dense
benchmark/kernels/vestigekv/                         kernel benchmarks
```

The four deleted lines are the attention-backend registration the new backend
slots into. Everything else is additive: with `--attention-backend` left alone,
the patched tree behaves as upstream, which is what the equivalence test pins.

## Do not install it

The harness imports the engine from source through `PYTHONPATH`. An installed
copy of sglang shadows the patched tree silently — the run then measures
upstream and reports it as VestigeKV. Every number in the paper was taken from
a source tree, never an installed one.
