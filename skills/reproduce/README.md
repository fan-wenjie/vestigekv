# VestigeKV reproduction & handoff package

Self-contained package to bring up VestigeKV on a fresh dual RTX 6000 Pro
(Blackwell/SM120) setup, reproduce every paper number, and continue the
project (paper + experiments) from a clean session. Read in this order:

1. **BOOTSTRAP.md** — from zero: drivers, env, code checkout, checkpoint.
2. **REPRODUCE.md** — launch + the three headline metrics, each with its bar.
3. **RULES.md** — standing project constraints (data provenance, parity,
   PR/paper hygiene). Inherit these when running experiments or editing.
4. **CONTEXT.md** — project state restore: what is done, what is open, the
   measured attributions — so a fresh session can continue paper/code work.
The paper source is deliberately NOT in this repository (it carries the
named-build author block, and this repo is the anonymous artifact); it
travels in the separate migration archive alongside this package.

## Why this package exists in this form

The two-node testbed this project ran on lost node1 (its mapped SSH port
stopped answering, platform-side). Everything needed to resume on different
hardware is here: bring up the environment (BOOTSTRAP), reproduce the
numbers (REPRODUCE), keep the working rules (RULES), restore what was
decided and what is still open (CONTEXT), and continue the write-up
(paper source, shipped in the migration archive). The one experiment left unrun is the Instruct checkpoint's
serving throughput/latency — CONTEXT names it under "Known open items".

Everything traces to the experiment repo (this repo): `results/` for the
serving metrics, `harness/` for the algorithm claims, `mexp/` and
`engine/` for the run machinery. No identity or paths outside the repo.
