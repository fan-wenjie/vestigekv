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

## Answered: it is mostly the benchmark

`sidecardumpprose-vestigekv`, the same snapshots during a novel continuation,
layer 19, one request each:

                  peak bin   period   peakiness   top-3% kept   top-10% kept
    RULER               71   57.7 t       889.4        30.3%         50.4%
    real prose          16    256 t        90.2        75.4%         83.6%

The location is the finding, not the height. Bin 16 is the first bin ABOVE
the kappa=16 cutoff, so prose's largest above-cutoff component is the shoulder
of the content the low-pass keeps -- what any smooth signal does at a
boundary. RULER's sits at bin 71, an isolated spike far from the cutoff, and
notching it moves seventy percent of what tier 1 keeps where the same
operation on prose moves a quarter.

So this is a third way RULER is adverse, after planted content and short
answers, rather than a property of the method.

## All seven layers, at no cost

The dumps already held them -- seven MLA layers times two TP ranks -- and only
layer 19 had been read. Every layer tells the same story:

    layer   RULER: bin  peakiness  top3%      prose: bin  peakiness  top3%
        3          71      175.7   40.2%              17       28.2  79.5%
        7          71      444.3   37.7%              16       44.0  77.9%
       11          71      467.9   59.8%              16       57.3  88.5%
       15          71     1048.0   51.6%              16       88.1  83.6%
       19          71      889.4   30.3%              16       90.2  75.4%
       23          71      410.7   35.2%              17       55.5  79.5%
       26          71       73.9   79.5%              17        8.1  85.2%

RULER's spike is at bin 71 in seven layers of seven; prose's largest
above-cutoff component sits on the cutoff, 16 or 17, in seven of seven. That
consistency is what one layer could not establish.

## Still not measured

Whether the notched ranking is better on either workload -- changing the kept
set is not improving it, and neither dump can say which.

## Four requests on RULER: it is a property, not a sample

`sidecardump5`, layer 19, one row per request:

    seq        peak bin   peakiness   top-3% kept
    64623            72       623.4        35.2%
    64729            71       635.1        39.3%
    64735            72       581.2        30.3%
    64809            71       890.9        30.3%

Same bin, same order of peakiness, same effect on the kept set. With seven
layers already agreeing, the RULER half is settled.

## The prose half is still n=1, and the reason is the dump's filename

`sidecardumpprose5` asked for five requests and produced one. The snapshot is
written to `cal_tp{rank}_slot{s}_lid{l}_seq{n}.pt`, and the `continue` client
sends five prompts of identical length through one slot -- same rank, same
slot, same layer, same seq, so each request overwrote the last. RULER escaped
it only because its prompts differ in length by a few dozen tokens.

Rather than change instrumentation that ships, three more prose dumps run at
62k, 60k and 58k: different lengths give different names, and length
independence is worth knowing anyway.
