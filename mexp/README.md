# mexp/ — active experiment scripts (sglang-only)

The only dependency is `sglang` (put on sys.path by `_bootstrap.py`; a
sibling checkout takes precedence over the submodule). Principle: never
write what sglang already provides; this directory holds only (1) data
collection and (2) functionality sglang lacks.

- `bench/` — collection and plotting (official `sglang.benchmark.serving`
  is authoritative; see its README)
- `launch_vestige_server.sh` — the canonical algorithm-arm launch (no magic
  numbers; enforced by `tools/check_recommended_config.py`)
- `quant_ab_probe.py` — scan-quantization gate 2: greedy signature + 256k
  needle, two-arm A/B (bars frozen in the docstring)
- `fire_set_stability.py` — quantization gate 1: fire-set stability
  (quantized vs fp32 arm, independently calibrated)
- `m7_mauve_serving.py` — MAUVE generation-quality client (HTTP-only; text
  out, offline scoring)
- `attn_only_bench.py` — attention-only pseudo-decode microbench (serving
  semantics: below D.ACTIVATION_MIN_TOKENS the vestigekv arm IS the dense
  step, mirroring the backend's activation threshold)
- `glm53/` — GLM-5.3-Flash-NVFP4 on 2x RTX PRO 6000 (branch `vestigekv-pro6000x2`):
  `common.sh` (quality/RULER line: CUDA graph on, radix off, `--random-seed`),
  `baseline.sh` (the model as shipped: DSA), `vestigekv.sh` (DSA off +
  vestigekv_mla, fp8 side pool), `dense_mla.sh` (the substrate vestigekv wraps;
  an ablation, not the baseline), `run_ruler.py` (lm-eval RULER, N per cell,
  seeded, /v1/completions), `needle.py` (10k-token head-needle probe),
  `queue_runner.py` + `queue.jsonl` (the JSONL job queue that drives the line)
- `_bootstrap.py` — path bootstrap
