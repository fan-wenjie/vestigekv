---
name: m7-mauve
description: MAUVE generation quality on the deployment path — vestige free generation vs uncompressed engine over 16 fineweb contexts.
---

# M7 MAUVE (generation quality on the deploy path)

**PREREG**: none | **Output**: out/m7_mauve.json | **Phase**: B

## What it measures
MAUVE generation quality over 16 fineweb contexts (T=4096 prefill, 256-token generations, temp 1.0 top-p 0.95). Three text sets over the SAME contexts: human (true continuation), eng (uncompressed engine free generation), vestige (rho=1/32 eviction + tier-2 r=64/topj=16, same seeds). Compares MAUVE(human,eng) vs MAUVE(human,vestige) with the gpt2-large featurizer.

## Frozen decision rule (if pre-registered)
Frozen in the script: mauve_vestige >= mauve_eng - 0.10 -> PASS (a 10-point drop is outside small-N MAUVE noise), else FAIL. Audit rule 1: >= 12 contexts generated per arm or ABORT. Sampling seeds fixed per context so arms differ only through the cache; both numbers reported verbatim (N=16 is small, stated).

## Reproduce
```bash

python mexp/m7_mauve.py
```
Requires: the mini-sglang package importable (see project README "Setup"), Kimi Linear checkpoint in the HF cache, plus mauve-text (gpt2-large featurizer). Phase C (dual-machine) also needs the second box reachable with the gloo transport — see mexp/README.md.

## Verdict
PASS — vestige generation quality is within 0.10 MAUVE of the uncompressed engine over 16 contexts; the deployed cache does not degrade free-generation quality. Both numbers in out/m7_mauve.json.
