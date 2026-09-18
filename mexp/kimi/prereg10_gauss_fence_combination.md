# Pre-registration 10: the Gaussian fit and the fence, composed

Status: **frozen 2026-09-18 02:5x**, before `a0-stats-ruler-5len` reports and
before the combination arm exists. The owner's decision rule is recorded below
verbatim in effect and is not mine to soften after the data.

## Why compose them

The two mechanisms move opposite ways on the same axis.

| | fallback | multi-needle accuracy |
|---|---|---|
| fence 256 | **up** | **up** (+0.100 on multikey_3 at n=50) |
| Gaussian 0.99 | **down** (0.195, against an A0 not yet measured) | **unchanged** (+-0.000, n=10) |

So the Gaussian frees budget and the fence spends it. If A0 at this shape is
near 0.38 -- `fence-mk1024`, which barely fires, reports 0.379 -- the Gaussian
frees about 19 points and the fence costs about 6.5, which would buy the
fence's multi-needle gain BELOW the delivered fallback rate. That arithmetic is
the point of the arm and is suspended until `a0-stats-ruler-5len` lands.

## The arm

`--vestigekv-cert-gaussian-target 0.99 --vestigekv-multikey-fence-rows 256`,
13 tasks, 4k-64k, n=10 to place it, then n=50 on the target family.

## The owner's decision rule

**If multi-needle accuracy does not rise, the Gaussian is dropped.**

Stated plainly because it is the rule that matters: the Gaussian's own value is
on the cost side and it buys no accuracy alone, so the only thing that makes it
worth carrying is that it does not cost the accuracy the fence buys. An arm
where the fence's gain fails to appear is an arm where the Gaussian cancelled
it, and the Gaussian goes.

## Gates

Against A0 at the same shape and n.

- **C1** (the owner's rule) `niah_multikey_3` rises. If it does not,
  `cert_gaussian_target` is abandoned and not revisited.
- **C2** the rise is within 0.02 of what fence 256 achieves alone, i.e. the
  Gaussian preserves the fence's gain rather than eroding it.
- **C3** the combined fallback rate is BELOW A0's. If composing costs more
  than the fence alone, the composition has no reason to exist.
- **C4** no task among the thirteen falls by more than 0.02.

## What the paper may not claim either way

The conformal guarantee. zp exists to deliver coverage at RECALL_TARGET = 0.90;
lowering it lowers nominal coverage, and the measured coverage at answer steps
is already 77% against that nominal 90%. **A paper cannot both claim the
conformal guarantee and take the Gaussian saving.** If this arm is adopted, the
guarantee sentence is removed and replaced with the measured coverage.

## Not revised after the data

C1 in particular. Deadline 2026-09-26.
