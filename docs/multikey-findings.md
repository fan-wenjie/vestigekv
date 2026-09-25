# The multi-key gap: what it is, and what catches it

Working notes, 2026-09-19. **Not yet integrated into the paper beyond one body
clause and one appendix paragraph** (see *Status* at the end). Kept here so the
next pass can decide what earns space rather than re-deriving it.

## The finding

RULER's multi-key needles are where VestigeKV loses ground, and the paper
attributes that to selection: a rank-`r` sketch fitted to the archive's own
content cannot separate rows drawn from the bulk it was fitted to, so the
certificate fires on the wrong row. Reading the generations says that account
is right about **half** the failures.

`mk3-copyfidelity-n50`, 100 items per task and arm, mainline configuration.
`niah_multikey_3` and `niah_single_3` are the same 36-character UUID copy at
the same lengths; only the haystack differs, prose against key-value pairs:

| | `multikey_3` (key-value haystack) | `single_3` (prose haystack) |
|---|---|---|
| dense | 100/100 | 100/100 |
| VestigeKV | **88/100** | 100/100 |

The copy never fails without compression, and under compression it never fails
in prose either. The 12 failures split evenly:

- **6 mis-selected** — a different needle's value. The selection failure the
  paper describes.
- **6 mis-copied** — the *right* needle with one or two characters wrong.

A 36-character UUID is about 20 decode steps, which is what makes the two
separable: a wrong row gives a wholly different string, a mis-copy gives 34 or
35 correct characters out of 36. Random UUIDs do not share 35 of 36 characters
by chance. `niah_multikey_2` shows only the first kind, and that fits: its
7-digit values are a three-step copy, too short to expose the second.

## What was ruled out, and how

The obvious suspect for a mis-copy in a haystack of hundreds of UUIDs is
blending with a competitor. It is not that: each mis-copy sits at **edit
distance 1-2 from its target and 21-24 from the nearest other UUID in its own
haystack**.

Two earlier arguments for the same conclusion do **not** hold and should not be
reused:

- *"The substituted character appears in ~6% of other haystack UUIDs, against
  1/16 = 6.25% at chance."* True but **powerless**: if the model blended with
  one specific competitor, that competitor's character is as rare among the
  rest as any other. It rules out a broad pull toward the haystack, nothing
  more. The distance test is what carries the conclusion.
- *"`niah_single_3` shows the same failure shape with one UUID in its
  haystack."* That came from the **randfence ablation arm**, not the shipped
  one. At n=50 on mainline, `single_3` does not fail at all.

## What is still open

Which per-step mechanism produces a mis-copy:

1. a step that failed to fetch its row, or
2. a step that fetched it and decoded wrong under the omitted mass.

These have different fixes and the data cannot separate them. Both signals
exist — `stepattr` records `top1_attended` with a minimum of 0.5625 and
`coverage` with a minimum of 0.11 — but that path computes true scores over the
whole closed prefix and says so in its own docstring: *"a debug-only path: it is
the dense attention the method exists to avoid"*. It cannot be read in
production and its records carry no per-item outcome, so they cannot be joined
to the failures above.

**The cheap instrument that would settle it** is already half-built. The scan
computes, per archived row,

```
acc   = branch·q  +  sketch·q          # the point estimate
score = acc*sc    +  cc*rho*qres       # + the certificate term
fired = score > max1
```

so "this row fired only because of its error bound" is `acc*sc <= max1 <
score` — one more comparison on registers already loaded, and the scan's cost
is dominated by the two per-row loads, not by arithmetic. Counting those per
step gives a free measure of how much a step's decision rests on the bound
rather than the estimate. Queued behind `ruler-base-n50`.

## Detection, measured

Three checks, on the same 100 items:

| check | cost | caught | false alarms |
|---|---|---|---|
| **provenance**: is the answer the value next to the asked key? | string work, **no attention** | **12/12** | **0/88** |
| **verbatim**: is the answer anywhere in the context? | string work, no attention | 5/12 | 0/388 |
| certificate slack | free, already computed | untested | — |
| dense recheck | the whole saving for that step | all | none |

The provenance check dominates the verbatim one for a structural reason: a
mis-copied answer is not next to the key because it is not in the text at all,
while a mis-selected answer is in the text but next to a *different* key. One
question sees both mechanisms. The verbatim check's one miss is a truncation,
and a truncated copy is still a substring of its source.

**Limits, which matter more than the 12/12.** The provenance check parses a
known answer format: it is a check for structured retrieval — key-value stores,
logs, JSON, tables — and not a hallucination detector for open generation.
Twelve errors is a thin base however clean the split looks. And it detects
without correcting, though this architecture has somewhere to escalate to: the
overflow path already attends the request's full row set exactly.

Worth noting: the workload where compression hurts most — a haystack that is
itself key-value pairs — is exactly the workload whose structure makes the
check possible.

## On the spectral route

Asked whether NoPE admits a spectral verification. I think not. The spectrum in
this method is a low-pass over the *sequence axis of branch values*, a content
statistic; RoPE's frequency structure comes from *position*, and NoPE is the
setting that removes it. There is no position-side handle to check against.

What NoPE does confer is the certificate: because the score is one fixed
bilinear form, a single per-row scalar bounds the error for **all** future
queries. Under RoPE the same bound must hold over the whole rotation orbit and
inflates out of usefulness. That is the paper's own Section 3(c), and it is the
NoPE-specific verification quantity — free, and untested against these
failures.

## Status

| where | what |
|---|---|
| paper body, Section 6.2 | one clause: 6 of 100 return a different needle, as many return the right one mis-copied |
| paper appendix H | the table above, the distance test, and a plain statement that we have no mechanism for the second half |
| **not in the paper** | everything in *Detection, measured* and *On the spectral route* |

Reproduce with `mexp/kimi/analyze_multikey.py` (`classify`, `interference`,
`verbatim_in_context`, `key_adjacent`); the run is registered in README as
`mk3-copyfidelity-n50`.
