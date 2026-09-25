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

## Amendment, before the arms report (2026-09-18)

`stepattr2-mk3` landed after this file was frozen and predicts G1 will fail.
The rows that go missing fall **7.06 scaled logits** short of max1 and rank
24th at the median under the certified score; the 0.999 Gaussian point adds
1.7 * cert where cert is of order 1, which does not close a deficit of 7. The
defect is in the ORDERING that `idxs` produces, and a parametric fit changes
only the inflation.

The arms still run, and the thresholds are NOT revised. What they now test is
narrower and still worth having: the fit is stable where the order statistic is
the maximum of eighteen samples, and a G1 failure alongside confirmed normality
locates the defect in the sketch rather than the quantile -- which is the
conclusion pre-registration 9 acts on.

## Result of the 0.99 arm, n=10 (2026-09-18)

**G1 fails exactly as the amendment predicted.** niah_multikey_2 and
niah_multikey_3 both move by +-0.000; twelve of thirteen tasks are unchanged
and only ruler_vt moves, by -0.012. The 13-task mean is 0.9179 against A0's
0.9179, identical to four decimals. An independent arm therefore confirms the
stepattr2 diagnosis: the deficit is in the ORDER idxs induces, and changing the
inflation does not touch it.

**An unexpected cost result, and a comparison I had to withdraw.** The arm
reports fallback 0.195. I first read that against 0.305 and called it a 36%
reduction; 0.305 is from a 65536-only run and this shape is five lengths, so
the comparison was invalid. There is no matched A0 with stats at this shape --
`a0-stats-ruler-5len` is queued to supply one. The same-shape fence arms
bracket it: fence 1024 barely fires and reports 0.379, so A0 here is likely
near 0.38, which would make the Gaussian fit a ~49% cut AND would mean the
fence-256 cost I have been quoting as +14 points is nearer +6.5. Both numbers
are suspended until the baseline lands.

**Why the direction reversed.** Offline, at n=128, the Gaussian point at 0.99
(2.759) sits ABOVE the conformal quantile (1.999), so I expected more rows
fired. In serving a tier leaves the clamp at min_hard = 18 samples, where the
conformal quantile IS the maximum of eighteen draws and systematically
overshoots mu + 2.33 sigma. So conformal is the larger of the two in the
regime that matters, and the parametric fit fires FEWER rows. That is a cost
finding, not an accuracy one, and it is outside the gates below, which were
written for the multi-key question.

## The matched baseline lands, and reverses the cost reading (2026-09-18)

`a0-stats-ruler-5len`, same thirteen tasks, same five lengths, same stats
setting:

| arm | fallback | vs A0 | multi-needle | 13-task |
|---|---|---|---|---|
| **A0 (matched)** | **0.175** | -- | 0.880 | 0.9141 |
| gauss 0.99 | 0.195 | **+0.020** | 0.890 | 0.9179 |
| fence 1024 | 0.379 | +0.204 | 0.890 | 0.9154 |
| fence 256 | 0.445 | **+0.270** | 0.980 | 0.9310 |
| fence 64 / 4 / 1 | 0.638 / 0.718 / 0.696 | +0.46 to +0.54 | | |

**Both of my earlier readings were wrong, and in the wrong direction.** I read
the Gaussian's 0.195 first as a 36% cut against 0.305, then revised to "maybe
49% against ~0.38". It is an INCREASE of 0.020. The offline prediction had it
right all along -- a higher target gives a larger zp, which fires more rows --
and the story I invented to explain the reversal, that serving fits on few
enough samples for the conformal maximum to overshoot, was wrong too.

The cause each time was an unmatched baseline: 0.305 is a 65536-only run and
these are five-length runs. That is the third time today the same mistake has
produced a confident number.

**The fence's cost is also far larger than I have been quoting**: +0.270 at
fence 256, not the +14 points and then +6.5 points I said. Even fence 1024,
which barely fires, costs +0.204.

**A useful by-product**: two A0 runs at n=10 differ by 0.010 on the
multi-needle pair and 0.0038 on the 13-task mean. That is the run-to-run floor,
and it is the reason the Gaussian's +0.010 is not a gain.

**Verdict under the stated objective** (accuracy up at least cost): the
Gaussian buys nothing and costs a little, so it is not a candidate. It remains
correct, stable and better-founded than the order statistic, which is worth a
sentence in the paper about method rather than results.

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

## The min_hard direction is retired (2026-09-18)

`--vestigekv-min-hard-factor` was added to keep tiers on the Z_MAX clamp
longer, because the clamp loses the top archived row on 16.0% of answer-step
records against the fitted certificate's 22.85%. Its arms at x4 and x16 are
**withdrawn before running**, by the owner, under the objective of raising
accuracy at minimal cost to performance.

The reason is that the clamp buys its lower miss rate by over-fetching -- a
provisional tier fires 1663 to 5629 rows where a calibrated one fires 0 -- so
it is the worst exchange rate available: a great deal of throughput for a
bounded accuracy gain. And the small-sample variance it was meant to suppress
is a property of the order statistic, which the parametric fit removes from the
other end without spending anything.

The flag stays in the tree at its historical default of 1.0. It is not to be
proposed again without a reason that answers the exchange rate.

## Not revised after the data

The gate thresholds, the target family, and G5's role in the description rather
than the adoption. Deadline 2026-09-26.
