# What the sidecar's spectrum shows, and what it does not

Measured on `sidecardump-vestigekv`: one RULER `niah_multikey_3` request at
64k, 14 snapshots (7 MLA layers x 2 TP ranks), whole 576-dim pool rows.

## Measured

A standing periodic component at bin 71-72 of a 4096-row block -- a period of
56.9 to 57.7 tokens -- above the kappa=16 cutoff, so it lands in sigma's
residual by construction.

    peakiness (max bin / median bin, above the cutoff)
      sidecar        748 - 1699     across blocks 0-5, layer 19
      content half    34 -   73     same blocks, 20x weaker
      broadband reference  1.4      planted white noise
      planted peak      9224        a sine at bin 200

It is in every block, not one, and at the same bin in every layer.

Notching bins 17, 18, 71, 72, 141, 142, 143 out of the residual changes which
rows tier-1 keeps:

    layer   rho(sigma, notched)   top-3% kept   top-10% kept
    lid11          0.861             60.7%         67.5%
    lid15          0.742             42.6%         58.7%
    lid19          0.745             33.6%         53.3%
    lid23          0.730             30.3%         50.1%
    lid26          0.928             81.1%         63.8%

So on this workload a large share of the kept set is decided by a component
at one frequency rather than by per-row anomaly.

## Not measured, and not to be asserted

The cause. Two guesses were made and both were wrong: that the branch is
RoPE'd (it is the un-roped half, by its own definition) and that the period
is a repeated filler sentence (the prompt's 1130 sentences are distinct). The
component is present in both halves of the latent and 20x stronger in the
sidecar; that is the whole of what is established.

Whether the notched ranking is BETTER. Changing 30-70% of the kept set says
the statistic is sensitive to this component, not that removing it helps. The
accuracy question needs a run with the notch applied, against the same cells.

Whether real prose shows it at all. This is a synthetic haystack.
`sidecardumpprose-vestigekv` dumps the same snapshots during a novel
continuation; if the peak is absent there, the finding is about the benchmark
and belongs beside the other RULER-is-adverse arguments. If it is present,
it is about the method.

Nothing from this file goes in the paper until those two are answered.
