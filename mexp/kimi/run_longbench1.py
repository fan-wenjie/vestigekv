#!/usr/bin/env python3
"""LongBench v1 summarization subsets against a live server, one arm.

    python mexp/kimi/run_longbench1.py --arm baseline [--port 30000] [--tag <job id>]

The generative long-context task the other quality gates lack: gov_report,
qmsum and multi_news need the whole document, not a needle, so tier-1
eviction plus recall has to hold up without a query that points anywhere.
Protocol is LongBench's own (THUDM/LongBench, config/dataset2prompt.json and
dataset2maxlen.json under mexp/kimi/longbench1/): the dataset's prompt
format, greedy, max_gen from the config, prompts over --max-length tokens
middle-truncated the way pred.py does it (half the budget from each end).
Raw completion, no chat template -- the same choice run_longbench2.py made,
and both arms get the identical string. ROUGE-L is scored afterwards by
make_lb1_numbers.py with py-rouge, exactly as LongBench's eval.py does.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(HERE, "longbench1")
SUBSETS = ["gov_report", "qmsum", "multi_news"]


def build_prompts(subset, tok, max_length):
    fmt = json.load(open(os.path.join(DATA, "dataset2prompt.json")))[subset]
    rows = [json.loads(l) for l in open(os.path.join(DATA, f"{subset}.jsonl")) if l.strip()]
    out = []
    for r in rows:
        prompt = fmt.format(**r)
        ids = tok(prompt, truncation=False, add_special_tokens=False)["input_ids"]
        truncated = len(ids) > max_length
        if truncated:
            half = int(max_length / 2)
            prompt = (tok.decode(ids[:half], skip_special_tokens=True)
                      + tok.decode(ids[-half:], skip_special_tokens=True))
        out.append({"_id": r["_id"], "prompt": prompt, "answers": r["answers"],
                    "tokens": len(ids), "truncated": truncated})
    return out


def generate(url, model, prompt, max_tokens, timeout):
    r = requests.post(url, json={"model": model, "prompt": prompt, "temperature": 0,
                                 "max_tokens": max_tokens}, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--port", default="30000")
    ap.add_argument("--model", default="moonshotai/Kimi-Linear-48B-A3B-Instruct")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "kimi", "longbench1"))
    ap.add_argument("--subsets", default=",".join(SUBSETS))
    ap.add_argument("--max-length", type=int, default=65536,
                    help="prompt token budget; longer prompts are middle-truncated (pred.py)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    maxlen = json.load(open(os.path.join(DATA, "dataset2maxlen.json")))
    url = f"http://127.0.0.1:{a.port}/v1/completions"
    os.makedirs(a.out, exist_ok=True)
    suffix = f"_{a.tag}" if a.tag else ""
    summary = {"arm": a.arm, "model": a.model, "max_length": a.max_length, "subsets": {}}
    for subset in a.subsets.split(","):
        items = build_prompts(subset, tok, a.max_length)
        if a.limit:
            items = items[: a.limit]
        path = os.path.join(a.out, f"pred_{a.arm}_{subset}{suffix}.jsonl")
        t0 = time.time()
        with open(path, "w") as f:
            for i, it in enumerate(items):
                text = generate(url, a.model, it["prompt"], maxlen[subset], a.timeout)
                f.write(json.dumps({"_id": it["_id"], "pred": text, "answers": it["answers"],
                                    "tokens": it["tokens"], "truncated": it["truncated"]}) + "\n")
                f.flush()
                if i % 20 == 0 or i == len(items) - 1:
                    print(f"{subset} {i + 1}/{len(items)} {time.time() - t0:.0f}s", flush=True)
        summary["subsets"][subset] = {
            "n": len(items), "truncated": sum(it["truncated"] for it in items),
            "max_gen": maxlen[subset], "wall_s": round(time.time() - t0),
            "tokens_max": max(it["tokens"] for it in items), "pred": path,
        }
        print(json.dumps({subset: summary["subsets"][subset]}), flush=True)
    with open(os.path.join(a.out, f"summary_{a.arm}{suffix}.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
