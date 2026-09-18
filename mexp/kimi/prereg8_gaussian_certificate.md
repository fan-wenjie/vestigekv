# Pre-registration 8: a parametric certificate

Status: **frozen 2026-09-18 01:4x**, with the owner's authorisation to change
the default if the data supports it, and BEFORE `gauss-099-ruler` and
`gauss-0999-ruler` report. Both were queued before this file existed.

## What is already established, offline, and is not under test here

`z_req = (true - idxs) / cert` is an inner product of two residuals in the
(kv - r) = 448-dimensional orthogonal complement, divided by sqrt of that
dimension. Under isotropy it is standard normal **by construction**, and over
24 Kimi calibration dumps it measures that way: skewness -0.18, excess
kurtosis -0.00. The Gaussian model is not an added assumption; it is the one
the normalisation was built for.

What the conformal order statistic cannot do, measured on the same dumps:

| | |
|---|---|
| conformal at n=128 (delivered) | 1.999 |
| conformal at n=18, the `min_hard` bar | 1.877, **sd 1.348 across layers** |
| sample maximum, conformal's ceiling | 2.749 |
| Gaussian at 0.99 / 0.999 / 0.9999 | 2.759 / 3.688 / 4.392 |
| Z_MAX, the safety clamp | 8.000 |

At the bar where a tier leaves the clamp, the order statistic IS the maximum of
eighteen samples and varies by 72% of its median across layers. It can never
exceed the sample maximum, which is about the 0.99 Gaussian point, so higher
targets are inexpressible. And conformal's one advantage -- a distribution-free
finite-sample guarantee under exchangeability -- is bought at that price while
exchangeability demonstrably fails: calibration queries lose the top archived
row on 0.1% of cases, answer steps on 22.85%.

## What is under test

`--vestigekv-cert-gaussian-target t`: fit `zp = mu + Phi^-1(t) * sigma` over
the calibration requirements instead of the order statistic. 0 keeps conformal.
Arms at t = 0.99 and 0.999.

**The expected mechanism, stated so it can be wrong.** A larger zp raises every
archived row's certified score by `zp * cert`, most for rows with large
`|q_perp| * rho`. Today's failure is a row that truly beats max1 being scored
below threshold, so if those rows carry large cert, a larger zp fires them.
It also fires irrelevant rows, so the cost is fallback. **This may therefore be
the same "be dense more often" trade as the fence, with a principled knob
instead of a fitted constant.** `randfence-20-ruler` bounds how much of any
gain is attributable to density alone, and applies here too.

## Gates

At n=50, 4k-64k, against A0 at the same n. Target family is multi-needle
(`niah_multikey_2`, `niah_multikey_3`); `ruler_qa_hotpot` is the multi-hop
control and is NOT in the target -- see pre-registration 7 for why.

- **G1** `niah_multikey_3` improves by >= 0.05 and `niah_multikey_2` does not fall.
- **G2** no task among all thirteen falls by more than 0.02.
- **G3** the 256k production stream within 2% of A0's 4.364 ms/token.
- **G4** the RULER fallback rate at 64k does not exceed the fence's 0.445. A
  parametric certificate that buys the same accuracy at a higher cost than an
  empirical constant is not an improvement on it.
- **G5** the gain exceeds `randfence-20-ruler`'s on the target by >= 0.03. If
  it does not, the certificate is not doing the work and the honest
  description is a density trade.

## Adoption

If G1-G4 pass, the default changes: `cert_gaussian_target` ships at the lowest
target that passes, and `RECALL_TARGET`'s role in fitting zp is retired in
favour of it. `min_hard` reverts to the existence bound and
`--vestigekv-min-hard-factor` is removed, because the variance it was added to
suppress is a property of the order statistic and does not exist in the fit.

If G5 fails while G1-G4 pass, the change still ships -- it is stable where the
order statistic is not -- but the paper describes it as a density trade with a
principled parameterisation, not as a better certificate.

If G1 or G2 fails, conformal stays and this is reported as the arm that a sound
model did not rescue, which is worth stating: the normality is real and the
gain is not, and that would locate the failure somewhere other than the
quantile.

## Not revised after the data

The gate thresholds, the target family, and G5's role in the description rather
than the adoption. Deadline 2026-09-26.
