# harness/ — HF-forward research harness (algorithm validation)

This is the **algorithm-validation** harness, separate from the serving
port (`engine/` + `mexp/`). It patches attention inside a plain
HuggingFace forward pass to measure *which rows the selection/recall logic
keeps and fetches* — needle retrieval, the observed-attention baselines
(H2O / SnapKV / StreamingLLM), the RoPE-collapse control, the sketch-rank
and branch-width ablations, and teacher-forced bits-per-byte. These are
claims about the **algorithm** (a function of the model and the selection
rule), not about the fused kernels; the serving port reproduces the same
selection bit-for-bit (the registered parity tests), and the serving-path
metrics (latency, throughput, gsm8k, MAUVE) live in `results/`.

Two self-contained files, HuggingFace only (no engine/ dependency):
- `e2e.py` — the experiment driver; `--ops` selects which measurement runs.
- `extract.py` — model loading + device map + the attention patch.

```bash
# example: baselines + needle at 8k (PREREG17 / tab:baselines)
python harness/e2e.py --arch kimi --seq-len 8192 --n-docs 0 \
  --needle-trials 24 --gpu-expert-layers 18 --seed 0 \
  --ops baselines --out results/harness_baselines_8192.json
```

Each paper table sourced here names its `--ops` and PREREG in the matching
skill file (see `EXPERIMENTS.md`).
