# Why the VestigeKV arm is not bit-reproducible, and the dense arm is

Known issue. Not fixed on this tree; the fix and why it is deferred are at the
end.

## The observation

Two runs of GSM8K-Platinum 64-shot, $n{=}1209$, same seed, same tree, same
serial protocol, radix cache off, one request at a time:

| arm | run `gsm8k-*` | run `gsm8kcount-*` |
|---|---|---|
| dense | 0.916 | 0.916 (1107/1209) |
| VestigeKV | 0.910 | 0.907 (1097/1209) |

The dense arm repeats exactly. The VestigeKV arm moves by about four
questions. A third pair (`gsm8krep3-*`) is registered to turn this from two
points into a spread.

The paper's older printed value for the dense Instruct arm was 0.911, which is
neither of these. That is a different matter and must not be folded in here:
dense is deterministic, so 0.911 cannot be another roll of 0.916 — it came
from a different setup. Nondeterminism explains the VestigeKV column only.

## The principle

A run is bit-reproducible when the token emitted at step $t$ is a function of
inputs the experiment controls: the prompt, the weights, the sampling seed,
and the tokens already emitted. Greedy decoding removes the seed, so for the
dense arm the emitted token is a function of the token stream alone. Every
kernel is invoked on the same shapes with the same rows in the same order, and
the run repeats exactly. That is what the table shows.

VestigeKV introduces one further input: **which tier-2 index is installed at
step $t$.** Tier 2 is built twice for every (request, layer). A *provisional*
index is built on the first decode step from cache-row proxies; the
*calibrated* one is built once `n_cal` real decode queries exist, because
calibration needs queries whose best-scoring row was evicted and those only
exist once the request is decoding. Until the calibrated index is installed,
the request is served by the provisional one, which takes the most
conservative rung with the gate open — sound, but it over-fetches.

The step at which a request stops using the provisional index and starts using
the calibrated one is therefore part of the computation. And that step is not
a function of any input the experiment controls.

## Where the clock enters

`RecallTier.build` runs on a **background thread and a side CUDA stream**
(`vestigekv_mla_backend.py`, `_build_worker_loop`, `_build_stream`), signalling
completion through a `threading.Event`. The main thread calls
`_install_finished_builds()` once per decode step and adopts whichever jobs
have their event set *at that moment*.

So the transition step is decided by how quickly the worker thread was
scheduled by the OS, how the side stream interleaved with the decode stream on
the device, and what else the GPU was doing. None of those is a function of
the prompt, the weights, the seed or the token stream. The transition step is
an **exogenous input**, and no seed controls it. This is the whole cause.

The asynchrony is deliberate and not an oversight: the build costs a few
milliseconds per layer, and blocking decode on it would stall every request at
exactly the point the method is trying to be cheap.

## Why an exogenous transition step changes output bits

The two indexes do not select the same rows. Tier 2 fires a row when its
certified score beats the kept maximum for its head, and the certificate's
inflation term `z` is what the two indexes disagree about: the provisional one
takes the ceiling rung, the calibrated one fits `z` on the request's own
queries. Different `z` gives a different fired set, hence a different set of
archived rows fetched and attended, hence a different attention output, hence
different logits.

Under greedy decoding a perturbed logit vector usually yields the same token:
the perturbation has to cross an argmax boundary to matter. Most steps
therefore agree, and the two runs stay identical. Occasionally a step does
cross, the two runs emit different tokens, and from there their prefixes
differ and they diverge outright.

That is exactly the observable shape: almost every question identical, a
handful flipped. Four of 1209.

## Why GSM8K shows it most

64-shot GSM8K-Platinum has a long prompt and a short answer. The
provisional-to-calibrated transition lands *inside* the answer, so the steps
it affects are a large fraction of the tokens that decide the question. A task
with long generations dilutes the same effect across many more steps, and a
task whose answer is shorter than the calibration window never transitions at
all. The same mechanism was recorded on the GLM line, where RULER cells moved
between runs at identical seeds.

## What this is not

It is not an error, and it is not the certificate failing. Both indexes are
sound: the recall guarantee holds under either, because both inflate the
certificate conservatively. What is not unique is the *output*. The method
guarantees a property of the rows it keeps, not a particular token string, and
two sound configurations of the same method can answer one question
differently.

Stating it the other way round is the honest version: **the guarantee is
deterministic, the output is not.**

## The fix, and why not now

Make the transition a function of the token stream instead of the clock:
install the calibrated index at a fixed decode step, blocking there until the
build is ready. Reproducibility returns, at the cost of a synchronisation the
design deliberately avoids, and a decode stall whose size is the build time.

It is not applied on this tree because it would change every measured
VestigeKV quality number, and those numbers are the paper's. Changing them
days before a deadline to gain reproducibility we can instead *disclose* is
the wrong trade. The disclosure is in the measurement-integrity appendix; the
frozen quality bars are stated in questions rather than in the direction of a
difference, which is what lets the conclusion survive a four-question wobble.

For the GLM line, where nothing is frozen yet, prefer the deterministic
install from the start.
