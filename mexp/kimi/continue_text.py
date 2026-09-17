"""Continue a real long document, to separate natural text from decode length.

    python mexp/kimi/continue_text.py --port 30000 --input-len 65536 \
        --output-len 14 --num-prompts 130

Every fallback measurement so far confounds two axes. The "naturalness"
ordering recorded in pre-registration 6 -- self-continuation 0.003, real
documents 0.241, needles 0.360, random tokens 0.407 -- also happens to be an
ordering by decode length: those runs generated ~250k, 1, 14 and 14 tokens.
And the decode count moves the rate by 180x where the text type moves it by
35%, so the ordering is mostly the confound.

This holds the text natural and varies only the generated length, against
`stats-stream-64k-x130` (random tokens, same context, same counts) which holds
the length and varies only the text. The four together are a 2x2.

Documents come from the LongBench v2 cache, longest first, truncated to
exactly --input-len tokens so the context is real prose rather than a random
token stream. `ignore_eos` keeps every request generating the full
--output-len: a short answer that stops early would silently shorten the very
axis under test.
"""

import argparse
import json
import os
import sys
import time

import requests

MODEL = "moonshotai/Kimi-Linear-48B-A3B-Instruct"


def load_docs(want, min_chars, subs):
    """Literary prose, longest documents first.

    Filtered to the Literary and Detective sub-domains on purpose: those are
    novels, the most natural text in the set and so the most favourable case
    for a prior built on the low-frequency structure of ordinary language. A
    government annual report or a grammar description would test something
    blurrier.
    """
    from datasets import load_dataset

    ds = load_dataset("THUDM/LongBench-v2", split="train")
    rows = [r for r in ds if r.get("sub_domain") in subs]
    rows.sort(key=lambda r: -len(r.get("context") or ""))
    seen, out = set(), []
    for row in rows:
        ctx = row.get("context") or ""
        if len(ctx) < min_chars or ctx[:200] in seen:
            continue
        seen.add(ctx[:200])
        out.append(ctx)
        if len(out) >= want:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="30000")
    ap.add_argument("--input-len", type=int, default=65536)
    ap.add_argument("--output-len", type=int, default=14)
    ap.add_argument("--num-prompts", type=int, default=130)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--sub-domains", default="Literary,Detective")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # ~4 chars per token is the usual English ratio; ask for double so the
    # truncation below always has enough to cut from.
    docs = load_docs(max(8, args.num_prompts), args.input_len * 8,
                     set(args.sub_domains.split(",")))
    if not docs:
        raise SystemExit("no LongBench-v2 context long enough; lower --input-len")
    print(f"{len(docs)} distinct documents, reusing them in order", flush=True)

    # Tokenised once per DOCUMENT, not once per request: these run to 3.6M
    # characters and the naive loop re-tokenises the same novel 130 times.
    toks = [tok(d, add_special_tokens=False)["input_ids"] for d in docs]
    prompts = []
    for i in range(args.num_prompts):
        ids = toks[i % len(toks)]
        if len(ids) < args.input_len:
            continue
        # A different window per request, so repeats are not the same prefix
        # (the radix cache would serve the second one from the first).
        off = (i // len(docs)) * 997
        prompts.append(tok.decode(ids[off : off + args.input_len]))
    if not prompts:
        raise SystemExit("no document reached --input-len tokens")

    url = f"http://127.0.0.1:{args.port}/v1/completions"
    t0 = time.time()
    n_tok = 0
    for i, p in enumerate(prompts):
        r = requests.post(
            url,
            json={
                "model": args.model,
                "prompt": p,
                "temperature": 0,
                "max_tokens": args.output_len,
                "ignore_eos": True,
            },
            timeout=args.timeout,
        )
        r.raise_for_status()
        n_tok += r.json()["usage"]["completion_tokens"]
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(prompts)}  {time.time() - t0:.0f}s", flush=True)
    print(
        f"== continue_text in={args.input_len} out={args.output_len} "
        f"n={len(prompts)} generated={n_tok} wall={time.time() - t0:.0f}s"
    )


if __name__ == "__main__":
    main()
