# Material cut from the paper, and why it is here

The paper grades its own evidence into three tiers, and this directory is the
third. The rule:

| tier | what goes there | test |
|---|---|---|
| **body** (9 pages) | core performance data, and whatever a reader needs to understand the algorithm and its engineering | would the narrative be incomplete without it? |
| **appendix** (6 pages) | the defence of claims a reviewer is likely to contest | could an objection here decide acceptance? |
| **this directory** | the remaining evidence | true, checkable, and not load-bearing for either of the above |

The point of the split is focus. A paper that carries every measurement it took
is a document collection, and a reader cannot tell which numbers the argument
rests on. Everything below is real and was in the paper at some point; none of
it is needed to follow the argument or to judge it.

Each file is the LaTeX exactly as it stood in the paper, so nothing is
paraphrased in the move. Numbers appear as macros (`\rulerKDeltaNiahMKc`) whose
values are generated from `results/results.zip` by
`mexp/kimi/make_ruler_numbers.py` and `mexp/kimi/make_lb2_numbers.py` — the same
generators the paper uses, so these files and the paper cannot disagree.

## Contents

### `deployment-spec.tex` — the per-stage operational specification
What happens at a block close, at an index build, at a decode step, and under
the optional per-layer cascade, each with the measured status of its invariant.
**Why here:** a reimplementer needs this; a reviewer does not. The paper keeps
the two cost derivations (read budget, memory–time product) because a reviewer
may well check the accounting, and drops the checklist.

### `related-work-taxonomy.tex` — the extended related work and its table
Every family sorted by what its warm-up costs, what that warm-up settles, and
whether the decision is reversible — the axis on which this method differs
(its warm-up is paid in throughput and decides how much to read back, not what
to discard). **Why here:** positioning, not evidence. The body's related work
makes the argument in prose; this is the systematic version.

### `scope-boundaries.tex` — the long form of the limitation
The full three-factor argument (planted content, a haystack of competing
records, a short decode), the direction each factor pushes, the
remove-one-factor evidence, and the two-protocol distinction between LongBench
v2 as a benchmark and as a corpus. **Why here:** the *finding* — that the gap
sits only where the haystack is itself key–value pairs, and that multiplicity
alone costs nothing — is in the body, and the classification table is in the
appendix. This is the reasoning around them.

### `supporting-tables.tex` — four tables the paper cites but does not print
`tab:anatomy` (the branch as a trained salience channel, static analysis),
`tab:bpb` (general language-modelling cost, ΔCE in nats/token), `tab:tier`
(per-tier recovery detail) and `tab:config` (the empirical defaults, each
traced to a pre-registration). **Why here:** each supports a body sentence that
already states its result. `tab:config`'s values are also machine-checked
against the source by `tools/check_recommended_config.py`.

### `tier1-ablation.tex` — the tier-1-only ablation table
Needle intact rate with the recall tier **off** — not the deployed form.
**Why here:** it is an ablation of a configuration the paper does not ship. The
baseline table it used to sit beside (what H2O and friends do on this model)
stays in the appendix, because that one is the paper's premise and a prime
target for "you misconfigured the baselines".

### `interval-estimates.tex` — Wilson intervals on the needle rates
95% intervals on the exact fractions behind the small-n needle counts, plus the
gates table. **Why here:** the separations the paper leans on are stated in the
body with their trial counts; this makes the resolution explicit for a reader
who wants it.

### `../nope-dividends.md` — what NoPE-MLA confers on prior algorithms
Seven derivations, no new measurements, three of them empty on this checkpoint.
Written as Markdown rather than LaTeX because it is the one piece meant to be
read outside the PDF. **Why here:** it is a consequence of the paper's claim,
not evidence for it.
