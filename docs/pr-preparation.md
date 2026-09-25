# What to delete before the upstream PR, and what must survive it

A fork accumulates two kinds of code that look identical to a reviewer and are
not: code that substantiates how we got here, and code that reproduces a
number someone can read in the paper. The first is ours to delete; the second
is not, because deleting it makes a published figure unreproducible while
leaving the figure in print.

**The rule: delete what only substantiates history; keep what reproduces a
published number or guards correctness.** Everything below is that rule
applied, with the evidence each verdict rests on rather than an opinion about
it. The evidence is a count: how many times the switch appears in `README.md`
(which is the registry of experiments this paper ran) and in the `mexp/kimi/make_*.py`
generators (which turn logs into the macros the paper prints). A switch that
appears in neither is not producing anything a reader can see.

Counts taken 2026-09-21; recount before acting on this, because the registry
grows.

## The switches

| switch (`SGLANG_DEBUG_VESTIGEKV_*` unless noted) | README | generators | verdict |
|---|---|---|---|
| `NO_OVERFLOW_FALLBACK` | 3 | **1** | **keep** — the only one a generator reads directly; it produces the fallback numbers in Appendix F |
| `DUMP_DIR` | 11 | 0 | **keep** — the calibration snapshots and the sidecar spectrum study are taken through it |
| `STATS` | 9 | 0 | **keep** — every VKSTATS job |
| `OMITTED_BLEND` | 1 | 0 | **keep** — one registered RULER arm |
| `FENCE_STUB` | 1 | 0 | **keep** — one registered arm |
| `STEPDUMP` | 1 | 0 | **keep** — one registered arm |
| `MEM_DIR` | 1 | 0 | **keep** — the OOM memory trace |
| `ROWS`, `TAIL` | 0 | 0 | **keep, weakly** — assertions, not an alternative scheme, and they are what caught the int32 CSR overflow that corrupted rows silently. Cheap and off by default |
| `SPREAD_TRUNCATE` | 0 | 0 | **delete** |
| `RANDOM_FENCE` | 0 | 0 | **delete** |
| `SGLANG_VESTIGEKV_POOL_READ` | 0 | 0 | **delete** |
| `SGLANG_ENABLE_VESTIGEKV_INGRAPH_SCAN` | 0 | 0 | **delete** — see below |

`INGRAPH_SCAN` is the one that matters. It is `EnvBool(True)` and its off-path
is not a flag but a **second complete implementation**: `_ingraph_pack`,
`_ingraph_host_step` and `init_forward_metadata_in_graph` against the legacy
scan graph. Two captured paths, and no registered experiment has ever selected
the second one — so nothing in this paper's evidence exercises it. Upstream
will ask why there are two, and the honest answer is that there is no longer a
reason. Deleting it removes the larger half of the backend's branching, not
one environment variable.

## `benchmark/kernels/vestigekv/bench_vestigekv_kernels.py`

Keep it here, leave it out of the PR.

It costs the shipped code nothing: the implementations it compares against
(`eager()`, `torch_chain()`) are defined **inside the benchmark file**, not
left alive as dead branches in the modules. So this is not a question of
dragging weight into the PR.

It is a question of what the benchmark asserts. Its claim is "the shipped
kernel is faster than the one it replaced", and upstream never had the one it
replaced — there, it benchmarks against a strawman that cannot regress. For us
it is a real regression guard, and the GLM port and any context-parallel work
are exactly when a kernel quietly gets slower, so it stays in this tree.

If it is ever wanted upstream, the framing has to change from *shipped vs
replaced* to *absolute cost at serving shapes*, which survives the transplant.

## The naive reference stays

`test/manual/test_vestigekv_equiv.py` asserts `recall_tier.py` equal to an
in-test naive reference, and `recall_tier.py`'s own docstring carries the
rule: change the math only in lockstep with that reference, never one-sided.
This is the correctness net, it is the kind of thing upstream wants, and it is
not what "old scheme" means here.

## Sequencing: not yet

None of these deletions happens while the experiment queue is producing the
paper's numbers. `INGRAPH_SCAN` in particular touches a wide area of the
backend, and the tree it touches is the tree the results come from. Breaking
it mid-queue does not cost a deletion, it costs the runs.

The order:

1. the queue drains and `results.tar.xz` is built;
2. a PR-prep branch, off the tree the results were produced on;
3. deletions one commit each, so a bisect can name which one broke something;
4. the gate is the GPU kernel suite plus `test/manual/test_vestigekv_equiv.py`,
   both of which must pass at every commit, not only at the end;
5. the paper's numbers are regenerated once on the stripped tree and compared
   byte-for-byte against the ones it was submitted with. A deletion that
   changes a published number is not a deletion, it is a silent edit to the
   paper.

Step 5 is the one that is easy to skip and expensive to skip.
