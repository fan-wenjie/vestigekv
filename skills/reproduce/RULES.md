# VestigeKV — standing rules (inherit these when continuing the project)

These are project constraints learned the hard way. Follow them when
running experiments or editing the paper.

## Experiment / data provenance

- **README-prereg**: any number the paper uses must have its launch+bench
  commands written into README FIRST, audited, then run VERBATIM. The only
  allowed deviation is the deployment-specific PP partition
  (`SGLANG_PP_LAYER_PARTITION`). Changing a protocol means editing README
  first, not improvising at the shell.
- **PR-equivalent tree**: experiment data must come from the vestigekv PR
  branch tree. The launch guard checks `HEAD^{tree} == vestigekv^{tree}`
  and aborts otherwise. Never quote a number from a dirty or non-PR tree.
- **Same-launch vs baseline identity**: the dense baseline is stock
  `--attention-backend triton` on the SAME tree. Do NOT substitute the
  VestigeKV "full arm" (all rows kept) for the baseline — it uses the
  faster in-graph machinery and silently inflates the ratio. (This bug
  once flipped bs=1 from 0.976x to a false 1.03x.)
- **Three-way comparison**: baseline + history + theory; stop and attribute
  the moment a delta exceeds noise; keep protocols in separate columns.
- **Profile before guessing**: one guess allowed; a wrong guess means
  nsys/ncu first, no fix before instrument data.

## Correctness

- **No quantization** of cache rows (bf16, bit-exact). Only the 16-bit scan
  index metadata is reduced-precision, calibrated after rounding.
- **Recall is mandatory**: both tiers always on; tier-1-only is an ablation
  only. The certificate must hold (conservative over-fetch, never
  under-recall).
- **Bit-parity gates**: every kernel/optimization keeps a pre-fusion oracle
  and must pass a greedy bit-identity gate. The gate MUST cover a block
  close — i.e. run past 4096 decode tokens. (A 2048-token gate once missed
  a host-mirror drift that only manifests at the block-close rebalance.)
- **Block-close safety**: kept_len changes NON-incrementally at a block
  close (tier-1 global rebalance). Any per-step host mirror of kept_len is
  unsafe unless invalidated at the close. (This is why the kept_len mirror
  optimization was reverted.)
- **Audit that can fail**: a guard must be driven with a known-bad input
  and shown to refuse it; a STATS assertion that never fires is untested.

## PR / submission hygiene

- PR branch is a SINGLE commit; re-squash from `vestigekv-dev` and
  force-push. The public/upstream PR uses the project author's real
  name/email and `Co-authored-by: Claude ...` (lowercase, upstream style); NO
  `Claude-Session:` trailer on the public PR (privacy).
- The ANONYMIZED experiment repo uses `Anonymous <anonymous@example.com>`
  and must be identity-clean (no fanwj/wenjie/ustc/yotta / home paths).
- Stock-file changes minimal; the vestigekv backend lives in its own
  namespace; tests registered under test/registered with correct suites.

## Paper

- Every data macro traces to a results/ file or a harness run; nothing the
  final code cannot reproduce appears in the paper.
- bs=1@64k near-dense is honest physics (attention ~7% of the step), not a
  defect; the win is long-context (latency) and batch (throughput), and is
  deployment-dependent in the honest direction.
- Two build targets: anonymous (main.tex, double-blind) and named-arXiv
  (preprint_named.tex with \NAMEDBUILD). Both must compile clean and be
  self-contained (figures inlined, .bbl included).
