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

## Built: telemetry only (engine c9a9a7a480)

`SGLANG_DEBUG_VESTIGEKV_SPECTRUM=<blocks between reports>` (0 = off) adds one
rFFT per closing block per layer, off the per-step path, and logs per layer:

    VKSPECTRUM layer=19 blocks=14 peak_bins=128:8,17:2,16:2 peakiness_mean=59012
      above_cutoff_share_mean=0.9xx (telemetry only; a peak far above the
      cutoff means tier 1 is ranking a standing component)

Nothing reads it back. Live check on Kimi, two 24k requests through one
server: the repeated filler reports bin 128 -- its own 32-token cadence, not
RULER's 57-token one, which is the point: the detector reports whatever
repeats -- on 8 of 14 blocks at peakiness 1e4 to 6e4, and real prose (gov_report
text) reports bins 16/17, the shoulder, on the rest. The needle probe passes
with the telemetry on. `mexp/kimi/spectrum_notch_study.py` is the offline
counterpart over dumps.

## What it costs, measured

137 us per closing block per layer (microbenchmark on [4096, 64] bf16), which
is about fifteen kernel launches and not arithmetic: median 58, the energy
reduction 35, the transform 21, argmax and max 17. The first version cost 195
because it subtracted a mean that bin 0 carries and the cutoff excludes, and
read the device four times per block instead of once per report.

End to end on Kimi, 4k prompt with 2048 decoded tokens, two repeats per arm,
one launch each:

| | mean ITL | median ITL |
|---|---|---|
| telemetry off | 3.95, 3.92 ms | 3.90, 3.90 ms |
| telemetry on | 3.92, 3.91 ms | 3.89, 3.89 ms |

The arms sit closer than the repeats within an arm. A 4k prompt closes one
block per layer and all of it in prefill: 2.2 ms there, nothing per decode
step. At 128k it is 32 closes per layer, 80 ms over the request, 0.6 us per
decode step amortized. Batching the sixteen layers into one call would cut it
sixteenfold if it ever needs to be cheaper.

The accumulator pools a layer's blocks across requests on purpose: an operator
wants the traffic's distribution, not one request's. A per-request view would
be the first change if this ever fed a decision, which on this evidence it
should not.
