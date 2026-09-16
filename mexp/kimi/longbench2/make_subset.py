"""Write subset_120k.json: the LongBench v2 questions whose context is at most
LIMIT Kimi Linear tokens, with each question's domain, official length bucket and
difficulty (the paper reports accuracy per domain from these fields).

    python mexp/kimi/longbench2/make_subset.py [--limit 122880]

Deterministic: the dataset and tokenizer are pinned by name, the rows are in dataset
order, and the file records both plus the count.
"""

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = "THUDM/LongBench-v2"
TOKENIZER = "moonshotai/Kimi-Linear-48B-A3B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=122880, help="max context tokens (120k)")
    ap.add_argument("--out", default=os.path.join(HERE, "subset_120k.json"))
    args = ap.parse_args()
    from datasets import load_dataset
    from transformers import AutoTokenizer

    ds = load_dataset(DATASET, split="train")
    tok = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
    rows = []
    for d in ds:
        n = len(tok(d["context"], add_special_tokens=False)["input_ids"])
        if n <= args.limit:
            rows.append({"_id": d["_id"], "context_tokens": n, "domain": d["domain"],
                         "sub_domain": d["sub_domain"], "length": d["length"], "difficulty": d["difficulty"]})
    with open(args.out, "w") as f:
        json.dump({"dataset": DATASET, "tokenizer": TOKENIZER, "limit_tokens": args.limit,
                   "n": len(rows), "total": len(ds), "rows": rows}, f, indent=1)
    doms = {}
    for r in rows:
        doms[r["domain"]] = doms.get(r["domain"], 0) + 1
    print(f"{len(rows)}/{len(ds)} questions with context <= {args.limit} tokens: {doms}")


if __name__ == "__main__":
    main()
