# results/ — frozen final measurements (the numbers in the paper)

Raw data first, figures derived; nothing here regenerates silently.
Superseded protocol versions (`_v2`/`_v3`/`_v4b`, the retired `mauve64k`
n=16 run, and the older `_base` latency/throughput sweeps) were pruned
2026-09-13; the final frozen set below is what the paper's numbers come
from. k = 1024 tokens throughout.

| file pattern | what it is |
|---|---|
| `latency_stream_4k-512k_{vestigekv,dense}[_instruct].jsonl` | bs=1 streaming decode, one request 4k prefill -> 512k; official `sglang.benchmark.serving --output-details` records (per-token `itls`). Non-suffixed = Base (the paper figure), `_instruct` = the second-checkpoint replication |
| `latency_stream_serverlog_{vestigekv,dense}[_instruct].log` | node-0 server logs of the same runs; the per-token authority above ~240k (client inter-token stamps degrade under stream batching) |
| `throughput_64k+4k_bs{1,2,4,8,12,16,24,32}_{vestigekv,dense}[_instruct].jsonl` | throughput sweep, 64k prefill + 4k decode, `max_concurrency`=bs; bs 24/32 are the post-freeze sweep extension (the bs=32 dip on both arms is chunked prefill occupying the window) |
| `quality_gsm8kplatinum_64shot_n1209_{dense,vestigekv}_{base,instruct}.txt` | GSM8K-Platinum 64-shot, n=1209, serial + radix-off |
| `quality_mauve_scores_{base,instruct}.json` | MAUVE (gpt2-large featurizer) verdicts, 4k context |
| `quality_mauve_texts_{dense,vestigekv}_{base,instruct}.json` | the generations the scores are computed over (16 ctx x 256 tokens, temp 1.0 top-p 0.95, seeds shared across arms) |
| `quality_mauve_contexts_tokens.json` | the 16 shared fineweb-edu contexts (token ids) |
| `quality_mauve64k64_{dense,vestigekv}[_instruct].json` + `_verdict[_instruct].json` | MAUVE at 64k prefill, n=64 (the n=16 first run saturated and was retired) |
| `quality_needle{,32,_zh}_{dense,vestigekv}[_instruct].json` + `_verdict[_instruct].json` | serving needles: 128k x8, 128k x32 fixed-seed, and the Chinese (sanguoyanyi filler) variant |
| `quality_needle_gptoss120b{,_verdict}.json` | gpt-oss-120b banded-sparse control (strict 5/8, digit-aware probe 8/8) |
| `quality_continuation_*` | archived continuation-agreement runs (gate retired, ERRATA #20; kept for audit) |
| `harness_anatomy_{base,instruct}.json`, `harness_needle_8192_{base,instruct}.json` | HF-forward harness outputs (model anatomy; prereg34 needle trials) |
| `fig_latency_curve.png`, `fig_throughput.png` | the two README/paper figures, rendered from the files above by `mexp/bench/plot_*.py` |

Both arms of every run load byte-identical weights: each server launch log
prints the checkpoint's content-level `WEIGHT-FP`, and the two serverlog
files carry the same value.
