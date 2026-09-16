"""LongBench v2 (<= 120k-token subset, mexp/kimi/longbench2/) against a running sglang
server: raw /v1/completions prompts, serial, greedy, 128 generated tokens, official
zero-shot template and answer extraction.

    python mexp/kimi/run_longbench2.py --arm baseline [--port 30000] [--tag <job id>]

Writes results/kimi/longbench2/results_<arm>[_<tag>].json (accuracy overall and per
domain / length bucket / difficulty, parse rate, wall time) and samples_<arm>[_<tag>].json
(every prompt's response and extracted answer, so the two arms can be diffed question
by question).
"""

import argparse
import collections
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
MODEL = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
TASK = "longbench2_kimi_120k"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--port", default="30000")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "kimi", "longbench2"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--limit", type=int, default=None, help="first N questions only (smoke)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from lm_eval import evaluator
    from lm_eval.tasks import TaskManager

    tm = TaskManager(include_path=os.path.join(HERE, "longbench2"))
    t0 = time.time()
    res = evaluator.simple_evaluate(
        model="local-completions",
        model_args={
            "model": args.model,
            "tokenizer": args.model,
            "base_url": f"http://127.0.0.1:{args.port}/v1/completions",
            "num_concurrent": 1,
            "max_retries": 3,
            "tokenized_requests": False,
            "max_length": 131072,
            "timeout": 3600,
            "trust_remote_code": True,
        },
        tasks=[TASK],
        task_manager=tm,
        batch_size=1,
        limit=args.limit,
        log_samples=True,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
    )
    samples = res["samples"][TASK]
    by = {"domain": collections.defaultdict(list), "length": collections.defaultdict(list),
          "difficulty": collections.defaultdict(list)}
    rows = []
    for s in samples:
        d = s["doc"]
        rows.append({"_id": d["_id"], "domain": d["domain"], "length": d["length"], "difficulty": d["difficulty"],
                     "answer": d["answer"], "response": s["resps"][0][0], "acc": s["acc"], "parsed": s["parsed"]})
        for k in by:
            by[k][d[k]].append(s["acc"])
    summary = {k: {g: {"n": len(v), "acc": sum(v) / len(v)} for g, v in sorted(by[k].items())} for k in by}
    out = {"arm": args.arm, "task": TASK, "n": len(rows), "results": res["results"][TASK],
           "acc": sum(r["acc"] for r in rows) / max(len(rows), 1),
           "parsed": sum(r["parsed"] for r in rows) / max(len(rows), 1),
           "by": summary, "wall_s": time.time() - t0, "seed": args.seed}
    os.makedirs(args.out, exist_ok=True)
    tag = args.arm + (f"_{args.tag}" if args.tag else "")
    with open(os.path.join(args.out, f"results_{tag}.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(args.out, f"samples_{tag}.json"), "w") as f:
        json.dump(rows, f)
    print(f"== LongBench v2 {args.arm} n={out['n']} acc={out['acc']:.3f} parsed={out['parsed']:.3f} wall={out['wall_s']:.0f}s")
    for k in ("domain", "length", "difficulty"):
        print(k, {g: f"{v['acc']:.3f} (n={v['n']})" for g, v in summary[k].items()})


if __name__ == "__main__":
    main()
