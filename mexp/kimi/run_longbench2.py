"""LongBench v2 (<= 120k-token subset, mexp/kimi/longbench2/) against a running
sglang server: the four choices A/B/C/D scored at the "Answer:" position, greedy,
one request per question.

    python mexp/kimi/run_longbench2.py --arm baseline [--port 30000] [--tag <job id>]

Writes results/kimi/longbench2/results_<arm>[_<tag>].json (accuracy overall and per
domain / length bucket / difficulty, wall time) and samples_<arm>[_<tag>].json
(every question's four choice logprobs and the pick, so the two arms can be diffed
question by question).

Scoring shape: upstream lm-eval's own LongBench v2 task scores four one-token
choices by loglikelihood rather than generating an answer (this model reasons
first and never reaches the answer line inside a usable budget -- measured, 1% of
300 parsed). lm-eval spends four `echo=True` requests per question to do that,
and `echo` pins `logprob_start_len=0`, which clamps sglang's radix prefix match
to zero (`schedule_batch.py::_compute_max_prefix_len`), so those four prefills of
one shared 120k context can never share cached KV. For a single-token
continuation the score is the next-token logprob at one position, so one
`max_tokens=1, logprobs=K` request carries all four. Same quantity, one prefill.
Deviation recorded in mexp/kimi/prereg3_realdoc_and_fallback.md, amendment 1.
"""

import argparse
import collections
import json
import os
import sys
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
MODEL = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
TASK = "longbench2_kimi_120k"
CHOICES = [" A", " B", " C", " D"]


def score(url, model, prompt, topk, timeout):
    r = requests.post(
        url,
        json={"model": model, "prompt": prompt, "temperature": 0,
              "max_tokens": 1, "logprobs": topk},
        timeout=timeout,
    )
    r.raise_for_status()
    top = r.json()["choices"][0]["logprobs"]["top_logprobs"][0]
    return [top.get(c) for c in CHOICES]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--port", default="30000")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "kimi", "longbench2"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--limit", type=int, default=None, help="first N questions only (smoke)")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(HERE, "longbench2"))
    import utils

    from datasets import load_dataset
    from tqdm import tqdm

    docs = list(utils.process_docs(load_dataset("THUDM/LongBench-v2", split="train")))
    if args.limit:
        docs = docs[: args.limit]
    url = f"http://127.0.0.1:{args.port}/v1/completions"

    t0 = time.time()
    rows, missing = [], 0
    for doc in tqdm(docs, total=len(docs)):
        lp = score(url, args.model, utils.doc_to_text(doc), args.topk, args.timeout)
        present = [i for i in range(4) if lp[i] is not None]
        missing += 4 - len(present)
        picked = "ABCD"[max(present, key=lambda i: lp[i])] if present else None
        rows.append({"_id": doc["_id"], "domain": doc["domain"], "length": doc["length"],
                     "difficulty": doc["difficulty"], "answer": doc["answer"],
                     "picked": picked, "logprobs": lp,
                     "acc": 1.0 if picked == doc["answer"] else 0.0})

    by = {k: collections.defaultdict(list) for k in ("domain", "length", "difficulty")}
    for r, doc in zip(rows, docs):
        for k in by:
            by[k][doc[k]].append(r["acc"])
    summary = {k: {g: {"n": len(v), "acc": sum(v) / len(v)} for g, v in sorted(by[k].items())}
               for k in by}
    out = {"arm": args.arm, "task": TASK, "n": len(rows),
           "acc": sum(r["acc"] for r in rows) / max(len(rows), 1),
           "choices_absent_from_topk": missing, "topk": args.topk,
           "by": summary, "wall_s": time.time() - t0}
    os.makedirs(args.out, exist_ok=True)
    tag = args.arm + (f"_{args.tag}" if args.tag else "")
    with open(os.path.join(args.out, f"results_{tag}.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(args.out, f"samples_{tag}.json"), "w") as f:
        json.dump(rows, f)
    print(f"== LongBench v2 {args.arm} n={out['n']} acc={out['acc']:.3f} "
          f"absent={missing} wall={out['wall_s']:.0f}s")
    for k in ("domain", "length", "difficulty"):
        print(k, {g: f"{v['acc']:.3f} (n={v['n']})" for g, v in summary[k].items()})


if __name__ == "__main__":
    main()
