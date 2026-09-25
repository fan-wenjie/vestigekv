"""Needle probe over prompt lengths around the 4096-row block boundaries.

    python mexp/kimi/probe_lengths.py [--port 30000] [--lengths 3000,3800,4000,4090,4200,6000,7900,8200,12000]

A 7-digit secret at the head, filler to the target token count (Kimi tokenizer),
then the question; /v1/completions, greedy, 96 tokens. Prints tokens, whether the
secret appears and the first 60 characters, so a run that turns to garbage shows
where. Also prints the token lengths of the saved RULER prompts per (task, length).
"""

import argparse
import json
import os
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--port", default="30000")
ap.add_argument("--model", default="moonshotai/Kimi-Linear-48B-A3B-Instruct")
ap.add_argument("--lengths", default="3000,3800,4000,4090,4200,6000,7900,8200,12000")
ap.add_argument("--samples", default="")
args = ap.parse_args()

from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
if args.samples:
    s = json.load(open(args.samples))
    for task in ("niah_single_1", "ruler_cwe", "ruler_fwe", "ruler_qa_squad"):
        for L in (4096, 8192, 16384):
            docs = [d for d in s[task] if d["doc"]["max_length"] == L][:3]
            n = [len(tok(d["doc"]["input"] + (d["doc"].get("gen_prefix") or ""))["input_ids"]) for d in docs]
            print(f"{task:16s} {L:6d}: prompt tokens {n}")
SECRET = "8814019"
FILL = "The quick brown fox jumps over the lazy dog while the sun sets slowly behind the distant hills. "
for L in [int(x) for x in args.lengths.split(",")]:
    head = f"Remember this secret code: {SECRET}.\n\n"
    q = "\n\nQuestion: What is the secret code mentioned at the beginning? Answer with the digits only.\nAnswer:"
    reps = 1
    while len(tok(head + FILL * reps + q)["input_ids"]) < L:
        reps = int(reps * 1.3) + 1
    while reps > 1 and len(tok(head + FILL * reps + q)["input_ids"]) > L:
        reps -= 1
    prompt = head + FILL * reps + q
    req = {"model": args.model, "prompt": prompt, "max_tokens": 96, "temperature": 0}
    r = urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{args.port}/v1/completions", data=json.dumps(req).encode(),
        headers={"Content-Type": "application/json"}), timeout=600)
    out = json.load(r)
    text = out["choices"][0]["text"]
    print(f"L={L:6d} tokens={out['usage']['prompt_tokens']:6d} hit={SECRET in text} text={text[:60]!r}", flush=True)
