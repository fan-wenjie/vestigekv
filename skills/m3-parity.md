---
name: m3-parity
description: HF-harness parity gate — KimiEngine chunked forward must match HF one-shot NLL to floor resolution; run first, it blocks M4–M7.
---

# M3 parity gate (native engine == HF one-shot)

**PREREG**: none (parity gate) | **Output**: out/m3_parity.json | **Phase**: B

## What it measures
KimiEngine chunked prefill (chunk=128, 2048 wikitext tokens) vs HF one-shot forward, per-position |dNLL|. Small chunk on purpose: maximizes chunk boundaries so KDA short-conv continuation bugs cannot hide. Companions localize any divergence (m3_diag: path-vs-chunking split) and set the ruler floor (m3_floor: bf16-vs-fp32-upcast |dNLL|).

## Frozen decision rule (if pre-registered)
Not PREREGged; the script freezes the bar (PORT.md): max|dNLL| <= 2e-3 AND median <= 5e-4 over all scored positions (floor-relative). Audit rule 1: aborts if it scored < 90% of the 2048 positions. FAIL blocks M4–M6.

## Reproduce
```bash

python mexp/m3_parity.py
# floor + divergence localization (companions):
python mexp/m3_floor.py     # -> out/m3_floor.json (ruler resolution)
python mexp/m3_diag.py      # -> out/m3_diag.json  (path vs chunking split)
```
Requires: the mini-sglang package importable (see project README "Setup"), Kimi Linear checkpoint in the HF cache. Phase C (dual-machine) also needs the second box reachable with the gloo transport — see mexp/README.md.

## Verdict
PASS — the chunked native engine reproduces HF one-shot NLL within the bf16 floor; the eager backend is the validated reference (sdpa disagrees by 1.4e-1 because v_head_dim != qk_head_dim). Gate results in out/m3_parity.json / out/m3_verdict.json; M4–M7 unblocked.
