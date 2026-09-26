# VestigeKV — supplementary archive

Code, run records and pre-registrations for *VestigeKV: Query-Independent
Sparse Attention with Certified Recall for NoPE-MLA Caches*.

The claim this archive supports is narrow and checkable: on a NoPE-MLA model the
cache already contains a query-independent eviction signal, so a serving-path KV
cache can be compressed with no training, no extra forward pass, and a certified
recall trigger — and the archive holds the exact record of what that cost and
what it did not.

## Start here

```bash
# 1. Build the engine: upstream sglang at a pinned commit + one patch.
cd engine && ./setup_engine.sh ~/sglang && cd ..
export PYTHONPATH="$HOME/sglang/python:$PYTHONPATH"

# 2. Unpack the run records.
tar xJf results/results.tar.xz -C results/

# 3. Regenerate every numeric macro in the paper from those records.
python mexp/kimi/make_ruler_numbers.py
python mexp/kimi/make_lb2_numbers.py
python mexp/kimi/make_lb1_numbers.py   # needs py-rouge, see the file's header
```

Step 3 is the point of the archive: the paper contains no hand-typed number.
Every figure in the text is a macro generated from a run record that is in here.

## Layout

| path | what it is |
|---|---|
| `engine/setup_engine.sh` | downloads the sglang v0.5.20 source tarball and applies the patch |
| `engine/vestigekv.patch` | the whole method: 40 files, 13342 lines added against 42 removed |
| `EXPERIMENTS.md` | every experiment command, registered before the run that produced it |
| `mexp/` | experiment drivers, the pre-registrations, and the macro generators |
| `harness/` | the measurement harness and the gates it refuses to run without |
| `results/results.tar.xz` | every run record behind every number (see `results/README.md`) |
| `results/fig_*.png` | the serving figures |
| `tools/` | utilities, including the stream reducer and this archive's build script |
| `docs/` | technical notes the paper points to |
| `config/`, `ENV_PINS.txt` | serving configuration and the pinned environment |

## The engine is a patch, not a fork

The archive ships no vendored engine. `engine/vestigekv.patch` is a plain `git
diff` against the sglang v0.5.20 release tag, which `setup_engine.sh` downloads
as a source tarball from the official repository. So what you build is upstream
code plus a diff you can read end to end. Outside its own package the change
touches five shared files — the backend registration branch, the server
arguments block, a `view`→`reshape` fix the MLA KV writer needs at long
prefill, a GSM8K test's accounting, and a spelling-checker word list — and
every one of those hunks is a modification, not a removal.

Do **not** `pip install` the resulting tree. The harness runs it from source via
`PYTHONPATH`; an installed copy shadows the tree you just patched, and every
measurement here was taken from source.

The backend registers as `--attention-backend vestigekv_mla`.

## Hardware and scope

Everything was measured on one machine: 2× RTX PRO 6000 Blackwell (SM120),
tensor parallel 2, on Kimi Linear 48B A3B (Base and Instruct). That is the
study's main limitation and the paper says so rather than implying breadth it
did not measure.

## Run records are reduced, and exactly so

The raw records are 14 GB — one inter-token latency per generated token, 520189
of them per streaming run, plus activation dumps and raw generations. That does
not fit an archive, so `tools/reduce_streams.py` reduces them: it computes every
statistic the paper reports **from the full array before discarding it**, stores
those exactly, and keeps only a head/tail-dense sample of the trace for
plotting. 490 MB of streaming records become 12.8 MB with the reported numbers
unchanged, proven per file by the script's `--verify` mode and end to end by
regenerating the paper's serving curve byte-identically.

Two categories are not here at all and `results/README.md` says why: raw
activation dumps (7.1 GB, regenerable from the public checkpoints by the
included extraction script) and raw RULER generations (4.8 GB, which no reported
number reads — the scored cells are 0.2 MB and are included).

## Pre-registration

`mexp/*/prereg*.md` are decision rules written **before** their data, with
verdicts appended after, including the ones that failed and the predictions that
were retracted. They are in the archive because the negative results are part of
the record: several mechanisms that looked like they should close the remaining
multi-key gap were pre-registered, run, and failed their own gates, and the
archive shows the gate being written first.

## What this does not claim

The residual RULER multi-key gap is open. The paper identifies its mechanism —
a resolution failure of the rank-64 content sketch on queries that must match
several distinct keys, not a threshold that could be inflated past — and does
not close it. The fence and the parametric certificate are documented failures,
not features.
