"""Does one top-k request reproduce the four-request loglikelihood ranking?

lm-eval's multiple_choice sends four `echo=True, logprobs=1, max_tokens=1`
requests per question and reads the last prompt token's logprob. sglang clamps
the radix prefix match to `logprob_start_len`, which echo pins at 0
(`schedule_batch.py::_compute_max_prefix_len`), so those four prefills of one
shared 120k context can never reuse cached KV. One `max_tokens=1, logprobs=K`
request asks for the same four numbers at the same position, in one prefill.

Prints both scorings side by side on the shortest documents of the 120k subset;
run_longbench2.py uses the top-k shape and this is the check behind it.

    python mexp/kimi/lb2_scoring_equiv.py [--n 3] [--port 30000] [--topk 20]
"""

import argparse
import json
import os
import sys

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
CHOICES = [" A", " B", " C", " D"]


def post(url, model, **kw):
    r = requests.post(url, json={"model": model, "temperature": 0, **kw}, timeout=3600)
    r.raise_for_status()
    return r.json()


def echo_scores(url, model, prompt):
    """lm-eval's shape: one request per choice, read the final prompt token.
    max_tokens=0 is rejected by this engine, so with max_tokens=1 the last
    prompt token is the second-to-last entry -- the same element lm-eval sums
    over (`openai_completions.py`: token_logprobs[ctxlen:-1])."""
    out = []
    for c in CHOICES:
        lp = post(url, model, prompt=prompt + c, max_tokens=1, echo=True, logprobs=1)
        lp = lp["choices"][0]["logprobs"]
        assert lp["tokens"][-2] == c, (lp["tokens"][-3:], c)
        out.append(lp["token_logprobs"][-2])
    return out


def topk_scores(url, model, prompt, topk):
    """One request: the next-token distribution at the same position."""
    d = post(url, model, prompt=prompt, max_tokens=1, logprobs=topk)
    top = d["choices"][0]["logprobs"]["top_logprobs"][0]
    return [top.get(c) for c in CHOICES], top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--port", default="30000")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(HERE, "longbench2"))
    import utils

    from datasets import load_dataset

    url = f"http://127.0.0.1:{args.port}/v1/completions"
    ds = utils.process_docs(load_dataset("THUDM/LongBench-v2", split="train"))
    docs = sorted(ds, key=lambda d: len(d["context"]))[: args.n]
    agree = worst = 0
    for doc in docs:
        prompt = utils.doc_to_text(doc)
        e = echo_scores(url, args.model, prompt)
        t, top = topk_scores(url, args.model, prompt, args.topk)
        pe = "ABCD"[max(range(4), key=lambda i: e[i])]
        present = [i for i in range(4) if t[i] is not None]
        pt = "ABCD"[max(present, key=lambda i: t[i])] if present else None
        agree += pe == pt
        worst = max(worst, max((abs(e[i] - t[i]) for i in present), default=0.0))
        print(f"chars={len(doc['context']):>7}  gold={doc['answer']}  echo={pe}  topk={pt}")
        print("   echo ", [None if v is None else round(v, 4) for v in e])
        print("   topk ", [None if v is None else round(v, 4) for v in t])
        print("   top-k keys:", json.dumps(sorted(top, key=top.get, reverse=True)[:8]))
    print(f"== argmax agreement {agree}/{len(docs)}  max |echo-topk| {worst:.4g}")


if __name__ == "__main__":
    main()
