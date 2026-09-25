#!/usr/bin/env python3
"""Radix-cache correctness probe for vestigekv backend.

Sends the same long prefix twice (2nd request hits radix cache), greedy decoding,
and checks token-level identity of the two outputs. Also sends a divergent-suffix
request sharing the prefix to check prefix reuse under compressed path.
"""
import json
import sys
import urllib.request

import requests

BASE = "http://127.0.0.1:30000"

def gen(prompt_ids, max_new=128, rid=""):
    r = requests.post(
        BASE + "/generate",
        json={
            "input_ids": prompt_ids,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new, "ignore_eos": True},
        },
        timeout=600,
    )
    r.raise_for_status()
    j = r.json()
    return j["output_ids"] if "output_ids" in j else None, j["text"], j["meta_info"]

def main():
    # Build a deterministic ~40k-token prefix: repeat a varied pattern, no trivial periodicity.
    rng_words = []
    seed = 12345
    for i in range(14000):
        seed = (seed * 1103515245 + 12345) % (2**31)
        rng_words.append(str(seed % 100000))
    prefix_text = "The following is a record of measurements: " + " ".join(rng_words) + ". Question: how many measurements are listed above? Answer with a number: "

    print("tokenizing prefix locally...")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("moonshotai/Kimi-Linear-48B-A3B-Base", trust_remote_code=True)
    ids = tok(prefix_text)["input_ids"]
    print(f"prefix tokens: {len(ids)}")
    if len(ids) < 20000:
        print("WARN: prefix shorter than intended; radix probe still valid but chunk-crossing less exercised")

    outs = {}
    for tag in ["first", "second(radix-hit)"]:
        oids, text, meta = gen(ids, 128)
        outs[tag] = (oids, text, meta)
        print(f"[{tag}] completion_tokens={meta['completion_tokens']} cached_tokens={meta.get('cached_tokens')} finish={meta['finish_reason']}")
        if oids is not None:
            print(f"[{tag}] output_ids[:16]={oids[:16]}")

    a, b = outs["first"], outs["second(radix-hit)"]
    same_text = a[1] == b[1]
    same_ids = (a[0] == b[0]) if (a[0] is not None and b[0] is not None) else None
    print(f"text identical: {same_text}")
    print(f"output_ids identical: {same_ids}")

    # divergent suffix sharing the long prefix
    ids_div = ids[:-16] + ids[-8:]  # mutate tail
    oids3, text3, meta3 = gen(ids_div, 128)
    print(f"[divergent] cached_tokens={meta3.get('cached_tokens')} (expect large prefix hit)")
    print(f"[divergent] text head: {text3[:80]!r}")

    ok = same_text and (same_ids in (True, None))
    print("PROBE RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
