# results/ — frozen final measurements (the numbers in the paper)

Raw data first, figures derived; nothing here regenerates silently.

| file | what it is |
|---|---|
| `latency_stream_4k-512k_{vestigekv,dense}.jsonl` | bs=1 streaming decode, one request 4k prefill -> 512k; official `sglang.benchmark.serving --output-details` records (per-token `itls`) |
| `latency_stream_serverlog_{vestigekv,dense}.log` | node-0 server logs of the same runs; the per-token authority above ~240k (client inter-token stamps degrade under stream batching) |
| `throughput_64k+4k_bs{1,2,4,8,12,16}_{vestigekv,dense}.jsonl` | throughput sweep, 64k prefill + 4k decode, `max_concurrency`=bs (bs>16 exceeds this card's memory); both arms switched on ONE live server (FULL-arm flag) so the fixed launch-state terms cancel |
| `quality_gsm8k_64shot_n800_{vestigekv,dense}.txt` | full few_shot_gsm8k output, 64-shot (~9k-token prompts), n=800, same question set both arms |
| `quality_mauve_scores.json` | MAUVE (gpt2-large featurizer) over the three text sets below |
| `quality_mauve_texts_{vestigekv,dense}.json` | 16 generations per arm, 256 tokens, temp 1.0 top-p 0.95, per-context seeds shared across arms |
| `quality_mauve_contexts_tokens.json` | the 16 shared fineweb-edu contexts (token ids) |
| `fig_latency_curve.png`, `fig_throughput.png` | the two README/paper figures, rendered from the files above by `mexp/bench/plot_*.py` |

Both arms of every run load byte-identical weights: each server launch log
prints the checkpoint's content-level `WEIGHT-FP`, and the two serverlog
files carry the same value.
