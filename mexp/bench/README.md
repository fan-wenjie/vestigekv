# Experiment-side benchmark tooling (minimal derivation; official code is authoritative)

Data chain: official sglang collection -> JSONL -> plot directly. No
intermediate CSVs (single source of truth).

- Collection (zero derivation, official module verbatim):
    python -m sglang.benchmark.serving --backend sglang --num-prompts 1 \
      --dataset-name random --random-input-len 4096 --random-output-len <N> \
      --max-concurrency 1 --output-details --output-file <jsonl>
  Entry i of the JSONL "itls" array = the single-step decode time at
  sequence length prefill+i.

- plot_curve_jsonl.py : the only latency plot script. Multi-arm overlay:
    python plot_curve_jsonl.py --arm dense:<jsonl> --arm vestigekv:<jsonl> \
      [--srv NAME:node0_log] [--clip S] [--ylim LO:HI] --out fig.png
  Rolling median + IQR from client itls; --srv overlays server-side
  Decode-batch bucket rates (the per-token authority above ~240k, where
  client inter-token stamps degrade under stream batching); table numbers
  are computed straight from the JSONLs, no intermediate files.

- plot_batch_throughput.py : the only throughput plot script. One bench run
  per (arm, bs); reads max_concurrency + output_throughput from the
  official record. Linear batch axis.

- bench_long.py : the single retained thin wrapper -- raises
  bench_one_batch_server's 600s client timeout to 3600s at runtime for
  512k-scale prefill-style discrete points. Does not modify the sglang tree.

## Figure conventions
- **bs=1 streaming decode** -> `plot_curve_jsonl.py`: y = per-token decode
  latency (ms/token), x = sequence length S; both arms on one figure.
  Latency is the natural bs=1 metric.
- **multi-bs comparison** -> `plot_batch_throughput.py`: y = decode
  throughput (tok/s), x = batch size (linear). Throughput is the natural
  batch metric.
