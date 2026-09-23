# Runtime haystack detection: the data, and why the notch variant is not worth building

Offline study over calibration dumps already on disk (`results/kimi/sidecardump*`,
`sidecardump_prose*`): 35 (request, layer) snapshots per corpus, 7 MLA layers,
~64k of context each. No new measurement, no behaviour change.
Reproduce with `python mexp/kimi/spectrum_notch_study.py`.

The question was whether a server could notice at block close that it is being
fed a synthetic haystack -- RULER's filler carries a standing component above
tier 1's low-pass cutoff -- and act on it, either by notching that component
out of sigma or by falling back to the exact path for the request.

## What separates the two corpora is the peak's LOCATION, not its size

| corpus | peak bin (4096-row block) | peakiness p50 | range over layers |
|---|---|---|---|
| RULER haystack | 71/72 in all 35 snapshots | 126-750 by layer | 50 (layer 26) - 1048 (layer 15) |
| real prose | 16/17 in all 35 snapshots | 8-91 by layer | 8 (layer 26) - 93 (layer 19) |

Bin 16 is the first bin past the cutoff, the shoulder any smooth signal leaves;
bin 71 is a period near 57 tokens, the filler's own sentence cadence. The bin
separates the two corpora 70/70. **Peakiness does not**: RULER's minimum (50,
layer 26) sits below prose's maximum (93, layers 15/19), so a single magnitude
threshold has no margin. Any detector should key on where the peak is relative
to the cutoff, not how tall it is. The paper's "724 times the median bin above
the cutoff -- in every layer" is the layer-19 figure; the standing component is
present in every layer, its prominence is not 724x in every layer.

## Notching sigma does not move it toward the queries

The dumps carry the request's own absorbed decode queries, so the rows those
queries really score highest are computable: an oracle top-m at the same 3%
budget. Overlap of tier 1's selection with that oracle, plain sigma against
notched sigma, averaged over the 35 snapshots:

| corpus | sigma vs oracle | notched sigma vs oracle | random baseline |
|---|---|---|---|
| RULER haystack | 4.7% | 4.3% | 3% |
| real prose | 9.9% | 10.2% | 3% |

Notching changes the kept set a great deal -- only 24-67% of RULER's kept rows
survive it, 69-88% of prose's -- and buys nothing: it moves the selection
*away* from the queries on the corpus it was meant to fix. The notch variant is
therefore not worth building.

## What the same table says about tier 1, and why the fallback variant is worse

Sigma's selection overlaps the oracle by 1.6x random on RULER and 3.3x on prose
(11x on prose at layer 26). Tier 1 is query-independent by construction, so a
weak overlap is expected and is exactly why tier 2 exists -- but it means a
haystack detector would be detecting "the tier-1 signal is weakly informative
here", which the recall tier already reports per query, per step, with a
certificate. Falling back to the exact path on detection would additionally
make every benchmark that trips the detector stop being evidence about the
sparse path: RULER is where the retrieval evidence lives, and the detector
fires there.

## What is worth building

Telemetry only: record the peak bin at each block close (the block close
already projects onto the low-frequency basis; one extra rFFT per block per
layer is ~1-2 us, amortized to well under 0.01 us per decode step) and expose
it. It tells an operator when tier 1's signal is degenerate without changing
what the server attends, and it is the distribution any future threshold would
have to be set from. Acting on it stays out of the deployment.
