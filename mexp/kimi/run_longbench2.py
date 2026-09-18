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


def generate(url, model, prompt, max_tokens, timeout):
    """Answer by GENERATING, so the decode path actually runs.

    The logprob protocol below scores the four choices from the first token,
    whose logits come from the prefill's last position. VestigeKV's
    forward_extend is the unmodified base kernel over the full pool -- the
    compression only changes forward_decode -- so that protocol produces
    bit-identical output on both arms by construction and measures nothing
    about the cache policy. Generating a real answer puts max_tokens decode
    steps on the compressed path, which is the thing under test.

    Long enough to matter: the recall index needs ~18 calibration queries
    before it leaves the Z_MAX clamp, so a handful of tokens would measure the
    warm-up and nothing else."""
    r = requests.post(
        url,
        json={"model": model, "prompt": prompt, "temperature": 0,
              "max_tokens": max_tokens},
        timeout=timeout,
    )
    r.raise_for_status()
    text = r.json()["choices"][0]["text"]
    # This model reasons before answering, so a bare \b[ABCD]\b matches a letter
    # inside the reasoning. Prefer an explicit verdict, then a parenthesised
    # choice, then the last standalone letter; record misses so a protocol that
    # stops parsing is visible instead of collapsing both arms onto the floor.
    for pat in (r"(?:correct )?answer is[^A-D]{0,12}\(?([ABCD])\)?",
                r"\(([ABCD])\)(?!.*\(([ABCD])\))",
                r"\b([ABCD])\b(?!.*\b[ABCD]\b)"):
        m = re.search(pat, text, re.S | re.I)
        if m:
            return m.group(1).upper(), text
    return None, text


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
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="0 keeps the logprob protocol (prefill only, no decode); "
                         ">0 generates that many tokens so the decode path runs")
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
        if args.max_tokens:
            picked, text = generate(url, args.model, utils.doc_to_text(doc),
                                    args.max_tokens, args.timeout)
            lp = None
            missing += picked is None
        else:
            lp = score(url, args.model, utils.doc_to_text(doc), args.topk, args.timeout)
            text = None
            present = [i for i in range(4) if lp[i] is not None]
            missing += 4 - len(present)
            picked = "ABCD"[max(present, key=lambda i: lp[i])] if present else None
        rows.append({"_id": doc["_id"], "domain": doc["domain"], "length": doc["length"],
                     "difficulty": doc["difficulty"], "answer": doc["answer"],
                     "picked": picked, "logprobs": lp, "text": text,
                     "acc": 1.0 if picked == doc["answer"] else 0.0})

    by = {k: collections.defaultdict(list) for k in ("domain", "length", "difficulty")}
    for r, doc in zip(rows, docs):
        for k in by:
            by[k][doc[k]].append(r["acc"])
    summary = {k: {g: {"n": len(v), "acc": sum(v) / len(v)} for g, v in sorted(by[k].items())}
               for k in by}
    out = {"arm": args.arm, "task": TASK, "n": len(rows),
           "acc": sum(r["acc"] for r in rows) / max(len(rows), 1),
           "unparsed" if False else "choices_absent_from_topk": missing,
           "topk": args.topk, "max_tokens": args.max_tokens,
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
