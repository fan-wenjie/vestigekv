# Q1 quality gate verdict — 2026-09-11 run

Gate (frozen in README "Pre-registration: 2026-09-11 reproduction run"):
vestigekv MAUVE >= dense MAUVE - 0.10, GSM8K-Platinum delta stated plainly.
GSM8K-Platinum full set n=1209, 64-shot; MAUVE 16 ctx x 256 gen tokens.

| model | arm | GSM8K-Platinum acc | Invalid | MAUVE |
|---|---|---|---|---|
| Base | vestigekv | 0.847 | 0.000 | 0.9991 |
| Base | dense (triton) | 0.861 | 0.001 | 0.9960 |
| Instruct | vestigekv | 0.916 | 0.000 | 0.9997 |
| Instruct | dense (triton) | 0.913 | 0.000 | 0.9805 |

Verdict: **PASS, both models.** MAUVE vestigekv > dense in both cells (Base
+0.003, Instruct +0.019; bar was vestigekv >= dense - 0.10). GSM8K-Platinum
deltas: Base -1.4pt (vestigekv below dense, opposite direction to the paper's
slightly-positive delta — recorded, not explained), Instruct +0.3pt.

Raw files: results/quality_gsm8kplatinum_64shot_n1209_{arm}_{tag}.txt,
results/quality_mauve_scores_{tag}.json, results/quality_mauve_texts_*.json.

Gate passed -> performance experiments (P1 latency, P2 throughput) proceed,
per the prereg, with the official sglang benchmark only.
