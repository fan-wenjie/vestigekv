"""Gate-2 A/B probe for scan-operand quantization (sglang serving path only).

Run once per arm against a live vestigekv server (CUDA graph on), writing one
JSON per arm; compare_quant_ab() diffs the two. Probes:
  greedy  -- fixed 64k-token prompt (seeded), 2048 greedy continuations; the
             token stream is the signature of the whole decode stack.
  needle  -- 8 seeded needles at 10..90% depth in a ~250k-token context,
             8-token greedy retrievals (recall evidence, mandatory).
PASS bar (frozen before data): greedy streams identical for >= 512 tokens and
needle retrievals differ by <= 1 of 8.

Usage:  python quant_ab_probe.py <arm-name> <out.json> [--base-url URL]
        python quant_ab_probe.py --compare a.json b.json
"""
import argparse
import json
import random
import sys

import requests

MODEL = "/home/user/.cache/huggingface/hub/models--moonshotai--Kimi-Linear-48B-A3B-Base/snapshots/3b171c17bfc4ee348599b6781a2ca8715c21c8dc"
GREEDY_PREFILL = 65536
GREEDY_OUT = 2048
NEEDLE_CTX = 249856  # tokens of filler; + needles stays under 262144 ctx
DEPTHS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)[:8]
SEED = 20260907


def _gen(url, input_ids, max_new):
    r = requests.post(
        f"{url}/generate",
        json={
            "input_ids": input_ids,
            "sampling_params": {
                "max_new_tokens": max_new,
                "temperature": 0.0,
            },
            "return_logprob": False,
        },
        timeout=3600,
    )
    r.raise_for_status()
    return r.json()["output_ids" if "output_ids" in r.json() else "text"]


def _tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)


def run_arm(arm, out_path, base_url):
    tok = _tok()
    rng = random.Random(SEED)
    vocab = tok.vocab_size
    res = {"arm": arm}

    # greedy signature: seeded pseudo-text prompt (token ids directly)
    ids = [rng.randrange(1000, vocab - 1000) for _ in range(GREEDY_PREFILL)]
    res["greedy"] = _gen(base_url, ids, GREEDY_OUT)

    # needle: filler sentences + planted facts, greedy retrieval per needle
    filler_ids = tok(
        "The sky was clear and the market stayed calm through the afternoon. ",
        add_special_tokens=False,
    )["input_ids"]
    n_rep = NEEDLE_CTX // len(filler_ids)
    ctx = (filler_ids * n_rep)[:NEEDLE_CTX]
    needles = []
    for i, d in enumerate(DEPTHS):
        key, val = f"K{i}X{rng.randrange(100,999)}", rng.randrange(10000, 99999)
        nid = tok(
            f" The secret number for {key} is {val}. ", add_special_tokens=False
        )["input_ids"]
        pos = int(len(ctx) * d)
        ctx = ctx[:pos] + nid + ctx[pos:]
        needles.append((key, val))
    res["needle"] = []
    for key, val in needles:
        q = tok(f" The secret number for {key} is", add_special_tokens=False)[
            "input_ids"
        ]
        out = _gen(base_url, ctx + q, 8)
        got = out if isinstance(out, str) else tok.decode(out)
        res["needle"].append(
            {"key": key, "want": val, "got": got, "hit": str(val) in got}
        )
    json.dump(res, open(out_path, "w"), indent=1)
    hits = sum(n["hit"] for n in res["needle"])
    print(f"[{arm}] greedy={len(res['greedy'])} needle_hits={hits}/{len(needles)}")


def compare(pa, pb):
    a, b = json.load(open(pa)), json.load(open(pb))
    ga, gb = a["greedy"], b["greedy"]
    if isinstance(ga, str):
        div = next((i for i, (x, y) in enumerate(zip(ga, gb)) if x != y), None)
    else:
        div = next((i for i, (x, y) in enumerate(zip(ga, gb)) if x != y), None)
    same = div is None and len(ga) == len(gb)
    ha = sum(n["hit"] for n in a["needle"])
    hb = sum(n["hit"] for n in b["needle"])
    g_ok = same or (div is not None and div >= 512)
    n_ok = abs(ha - hb) <= 1
    print(f"greedy: identical={same} first_divergence={div} (bar: >=512) {'PASS' if g_ok else 'FAIL'}")
    print(f"needle: {a['arm']}={ha}/8 {b['arm']}={hb}/8 (bar: |diff|<=1) {'PASS' if n_ok else 'FAIL'}")
    print("GATE2", "PASS" if (g_ok and n_ok) else "FAIL")
    return 0 if (g_ok and n_ok) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("args", nargs="+")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--base-url", default="http://localhost:30000")
    ns = ap.parse_args()
    if ns.compare:
        sys.exit(compare(*ns.args[:2]))
    run_arm(ns.args[0], ns.args[1], ns.base_url)
