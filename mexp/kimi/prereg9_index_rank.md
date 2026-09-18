# Pre-registration 9: the sketch rank

Status: **frozen 2026-09-18 02:0x**, before `rank128-ruler` and `rank192-ruler`
run. Nothing about the rank has been measured on this model beyond the
delivered 64.

## Why this arm, and why it is different from the eight before it

Eight mechanisms were ruled out on the recall side and every one of them was
aimed by inference. This one is aimed by a measurement of the failure itself.

`stepattr2-mk3`, at answer steps on calibrated tiers, over the 420 records that
lost the top-scoring row of the whole context:

| | p50 | p90 | max |
|---|---|---|---|
| rank of that row under the certified score | **24** | **1177** | **8747** |
| its certified score minus max1 | **-7.06** | | p10 -11.98 |

The archive is 59521 rows. A deficit of 7.06 scaled logits is about 1100x in
softmax weight, so it is not a threshold that a margin or a larger z can reach:
the 0.999 Gaussian point adds 1.7 * cert with cert of order 1. The clamp at
z = 8 adds roughly 6 * cert, which nearly covers it, and that is precisely why
the clamp loses the row on 16.0% of records against the fitted certificate's
22.85%.

So the deficit is in `idxs = q_side . side + q_sk . csk` -- the ORDER the
certificate induces -- and the one term of it that is an approximation is
`q_sk . csk`, the rank-r projection of the content. **The hypothesis is that
r = 64 does not resolve a query that must match several distinct keys.**

## The arm

`--vestigekv-index-rank`, already a flag, at **128** and **192** against the
delivered 64. No code change.

**The cost is real and is the point of G3.** The sketch is 2 bytes per rank per
archived row per layer, in both memory and scan traffic, so 128 doubles it and
192 triples it. At 64k with a 59521-row archive and 7 MLA layers that is about
53 MB per request at 64, 107 at 128, 160 at 192.

## Gates

At n=10 first (13 tasks, 4k-64k) to place the arms, then n=50 on the target
family for whichever passes. Against A0 at the same n.

- **G1** `niah_multikey_3` improves by >= 0.05 at n=10 and holds >= 0.05 at
  n=50; `niah_multikey_2` does not fall.
- **G2** no task among the thirteen falls by more than 0.02.
- **G3** the 256k production stream within 5% of A0's 4.364 ms/token. Looser
  than the usual 2% because the extra traffic is inherent, not incidental, and
  a real accuracy gain may be worth 5%; more than that and the arm is reported
  as an accuracy/throughput trade rather than an improvement.
- **G4** the fallback rate at 64k does not rise above A0's 0.305 by more than
  0.05. A higher rank should fire FEWER rows, not more -- the flag's own help
  says so -- so a rising fallback rate would mean the rank is not doing what it
  is supposed to and the result needs explaining before it is believed.

## What each outcome means

- **G1 passes**: the multi-key gap is a resolution problem in the sketch, and
  the fix is a parameter the design already exposes. That is the strongest
  outcome available to this line, and it retires the fence and the parametric
  certificate as workarounds for something that had a cause.
- **G1 fails while the rank rises**: the ordering deficit is not resolvable by
  rank, which would point at the BASIS rather than its size -- V is fitted to
  the archive's content by a rank-r decomposition, and a query matching several
  keys may need a basis fitted to something else. That is a harder problem and
  worth stating as one.
- **G4 fails**: investigate before believing anything else in the run.

## Not revised after the data

The gates, and the n=10-then-n=50 order. Deadline 2026-09-26.
