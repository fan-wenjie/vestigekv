# engine/ — the serving stack, as a patch

This directory holds no engine source. It holds the two things needed to
reconstruct one exactly:

| file | |
|---|---|
| `setup_engine.sh` | downloads the upstream sglang v0.5.20 source tarball and applies the patch |
| `vestigekv.patch` | the VestigeKV change: 40 files, 13342 lines added against 42 removed |

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

The pinned base is the v0.5.20 release tag. `setup_engine.sh` downloads the
official source tarball and runs `patch --dry-run` before touching anything, so
a patch that no longer applies fails loudly instead of half-applying.

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

Outside its own package the change touches five shared files: the
attention-backend registration branch, the server-arguments block, a
`view`→`reshape` fix in the MLA KV writer that long prefill requires, a GSM8K
test's accounting, and a spelling-checker word list. Every hunk there is a
modification; no upstream feature is removed, so with `--attention-backend`
left alone the patched tree behaves as upstream, which is what the equivalence
test pins.

## Do not install it

The harness imports the engine from source through `PYTHONPATH`. An installed
copy of sglang shadows the patched tree silently — the run then measures
upstream and reports it as VestigeKV. Every number in the paper was taken from
a source tree, never an installed one.
