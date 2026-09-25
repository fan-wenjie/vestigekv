"""Greedy continuation agreement: the decode-path fidelity gate.

Replacement for the retired serving bpb (ERRATA #18): teacher-forced
input logprobs are pure prefill, a path vestigekv delegates unchanged, so
that gate could never see compression error. This gate measures the DECODE
path directly: 16 fineweb-edu contexts of 65536 tokens each (compression
genuinely engaged: threshold 0, blocks close, the archive is read), then
512 greedy tokens (temperature 0; an arm that emits EOS early is kept
as-is, and a length mismatch counts as divergence at the shorter
length). Server must be the quality
configuration: --disable-radix-cache and serial requests (this script is
serial), which pins the batch to 1 and makes each arm deterministic.

Arms differ only through the cache policy, so every divergence is an
honest compression effect on the generation trajectory. Reported per doc:
first-divergence position and agreement fraction over the 512 tokens.

Frozen bar (fixed before any data): PASS iff median first-divergence
>= 128 tokens AND the worst doc diverges no earlier than token 8.
Rationale: dense-vs-dense on this protocol is bit-identical by
construction (deterministic, batch 1), so the noise floor is 512; the bar
asks the compressed path to track dense for a substantial horizon, and
flags any doc where generation derails immediately (compression breaking
the path, not perturbing it).

  python mexp/quality/continuation_agreement.py gen \
      results/quality_cont64k_contexts_tokens.json results/quality_continuation_<arm>.json
  python mexp/quality/continuation_agreement.py compare \
      results/quality_continuation_vestigekv.json results/quality_continuation_dense.json \
      results/quality_continuation_verdict.json
"""
import json
import sys

import requests

MODEL = "moonshotai/Kimi-Linear-48B-A3B-Base"  # tokenizer only; shared by both checkpoints
T, GEN, N_CTX = 65536, 512, 16
URL = "http://localhost:30000/generate"
BAR_MEDIAN_FIRST_DIV = 128
BAR_MIN_FIRST_DIV = 8


def contexts(path):
    try:
        return json.load(open(path))
    except FileNotFoundError:
        from datasets import load_dataset
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu", name="sample-10BT",
            split="train", streaming=True,
        )
        buf, ctxs = [], []
        for rec in ds:
            buf.extend(tok(rec["text"])["input_ids"])
            while len(buf) >= T and len(ctxs) < N_CTX:
                ctxs.append(buf[:T])
                buf = buf[T:]
            if len(ctxs) >= N_CTX:
                break
        json.dump(ctxs, open(path, "w"))
        return ctxs


def gen(ctx_path, out_path):
    ctxs = contexts(ctx_path)
    outs = []
    for ci, ids in enumerate(ctxs):
        r = requests.post(URL, json={
            "input_ids": ids,
            "sampling_params": {"max_new_tokens": GEN, "temperature": 0.0},
        }, timeout=3600)
        r.raise_for_status()
        j = r.json()
        got = j.get("output_ids")
        assert got is not None and len(got) > 0, f"ctx {ci}: no output_ids"
        # early EOS is legal: the arm stopped; compare() treats a length
        # mismatch as divergence at the shorter length
        outs.append(got)
        print(f"ctx {ci}: {len(got)} tokens", flush=True)
    json.dump(outs, open(out_path, "w"))


def compare(vk_path, dense_path, out_path):
    vk = json.load(open(vk_path))
    dense = json.load(open(dense_path))
    assert len(vk) >= 12 and len(dense) >= 12, "ABORT: <12 contexts per arm"
    rows = []
    for ci, (a, b) in enumerate(zip(vk, dense)):
        n = min(len(a), len(b))
        div = next((i for i in range(n) if a[i] != b[i]), n)
        rows.append({
            "ctx": ci,
            "first_divergence": div,
            "agreement": div / n,
        })
        print(f"ctx {ci}: first_div={div}/{n} agreement={div/n:.3f}", flush=True)
    divs = sorted(r["first_divergence"] for r in rows)
    med = divs[len(divs) // 2] if len(divs) % 2 else (divs[len(divs)//2 - 1] + divs[len(divs)//2]) / 2
    worst = divs[0]
    verdict = (
        "PASS" if med >= BAR_MEDIAN_FIRST_DIV and worst >= BAR_MIN_FIRST_DIV
        else "FAIL"
    )
    res = {
        "n_ctx": len(rows), "gen_tokens": GEN,
        "median_first_divergence": med, "min_first_divergence": worst,
        "mean_agreement": sum(r["agreement"] for r in rows) / len(rows),
        "bar": {"median_first_divergence >= ": BAR_MEDIAN_FIRST_DIV,
                "min_first_divergence >= ": BAR_MIN_FIRST_DIV},
        "verdict": verdict,
        "protocol": "64k ctx, 512 greedy tokens, temp 0, serial, radix off",
        "per_ctx": rows,
    }
    json.dump(res, open(out_path, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_ctx"}, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "gen":
        gen(sys.argv[2], sys.argv[3])
    else:
        compare(sys.argv[2], sys.argv[3], sys.argv[4])
