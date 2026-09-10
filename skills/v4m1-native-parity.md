---
name: v4m1-native-parity
description: Native pip-free KimiLinear stack parity — chunked NLL vs a saved HF reference, plus an fp32 single-layer referee; blocks V4-M2+.
---

# V4-M1 native parity (native stack == HF reference)

**PREREG**: none | **Output**: out/v4m1_parity.json | **Phase**: B

## What it measures
Two-process gate. Process A (v4m1_ref.py) saves the HF one-shot reference NLL vector (1024 wikitext tokens) to out/v4m1_ref.pt. Process B (v4m1_native.py) loads the native pip-free KimiLinearForCausalLM directly from the checkpoint (paged latent pool + KDA state pool) and computes chunked (chunk=128) NLL against the saved reference. Debug companions isolate divergence per layer (v4m1_dbgcmp/dbgref) and run an fp32 single-layer referee, KimiKDA vs HF KimiDeltaAttention at layer 0 (v4m1_layer0).

## Frozen decision rule (if pre-registered)
Frozen in v4m1_native.py (same family as M3): median|dNLL| <= 1.02e-2 (1.5x the bf16 floor median 6.83e-3) AND |mean NLL diff| <= 1.82e-3 nats. FAIL blocks V4-M2+. The fp32 referee expects ~1e-6 for math-identical layers (a real bug shows large).

## Reproduce
```bash

python mexp/v4m1_ref.py        # -> out/v4m1_ref.pt (HF reference, proven path)
python mexp/v4m1_native.py     # -> out/v4m1_parity.json (native chunked vs ref)
# optional referees:
python mexp/v4m1_layer0.py     # fp32 single-layer KimiKDA vs HF KimiDeltaAttention
python mexp/v4m1_dbgcmp.py ; python mexp/v4m1_dbgref.py
```
Requires: the mini-sglang package importable (see project README "Setup"), Kimi Linear checkpoint in the HF cache. Phase C (dual-machine) also needs the second box reachable with the gloo transport — see mexp/README.md.

## Verdict
PASS — the native pip-free stack reproduces HF-reference NLL within the floor-relative bar; the fp32 single-layer referee confirms KimiKDA is math-identical to HF KimiDeltaAttention. Gate in out/v4m1_parity.json.
