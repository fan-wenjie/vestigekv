"""RULER (lm-eval 0.4.13 `ruler` group) against a running sglang server, N samples per length.

Reproducibility: the data generators and lm-eval are seeded (--seed, default 0), the
server runs --random-seed 0 with the radix cache off, and requests are serial and greedy.

lm-eval's synthetic RULER tasks hard-code 500 samples per length; the generators are
wrapped here to cap that, since `--limit` would only take the first length's docs.
Requests go to /v1/completions (raw prompt: no chat template, so no forced reasoning),
serially, greedy, 128 generated tokens (the task yaml).
"""

import argparse
import inspect
import json
import os
import time

MODEL = "nvidia/GLM-5.3-Flash-NVFP4"
ALL_TASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multiquery", "niah_multivalue",
    "ruler_vt", "ruler_cwe", "ruler_fwe", "ruler_qa_squad", "ruler_qa_hotpot",
]


def cap_samples(n: int) -> None:
    from pathlib import Path

    import lm_eval.tasks.ruler.cwe_utils as cwe
    import lm_eval.tasks.ruler.fwe_utils as fwe
    import lm_eval.tasks.ruler.niah_utils as niah
    import lm_eval.tasks.ruler.qa_utils as qa
    import lm_eval.tasks.ruler.vt_utils as vt

    # lm-eval's yaml loader re-imports a task module from its file unless the
    # module in sys.modules carries the file's mtime (tasks/_yaml_loader.py);
    # stamp it so the tasks bind these patched module objects.
    for mod in (cwe, fwe, niah, qa, vt):
        mod.__mtime__ = Path(mod.__file__).stat().st_mtime_ns

    def cap(mod, name):
        f = getattr(mod, name)
        if "num_samples" not in inspect.signature(f).parameters:
            raise RuntimeError(f"{mod.__name__}.{name} has no num_samples to cap")

        def wrapped(*a, **k):
            k["num_samples"] = min(k.get("num_samples", 500), n)
            return f(*a, **k)

        setattr(mod, name, wrapped)

    cap(niah, "generate_samples")
    cap(cwe, "sys_word_pair_random")
    cap(fwe, "sys_kwext")
    cap(vt, "sys_vartrack_w_noise_random")
    cap(qa, "generate_samples")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--port", default="30000")
    ap.add_argument("--n", type=int, default=20, help="samples per (task, length)")
    ap.add_argument("--lengths", default="4096,8192,16384,32768,65536")
    ap.add_argument("--tasks", default=",".join(ALL_TASKS))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "ruler_glm53"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=MODEL, help="served model name (tokenizer too)")
    args = ap.parse_args()
    lengths = [int(x) for x in args.lengths.split(",")]
    tasks = args.tasks.split(",")
    cap_samples(args.n)

    from lm_eval import evaluator

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
            "max_length": max(lengths) + 512,
            "timeout": 3600,
            "trust_remote_code": True,
        },
        tasks=tasks,
        metadata={"max_seq_lengths": lengths},
        batch_size=1,
        log_samples=True,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
    )
    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.arm}_n{args.n}_{'-'.join(str(x) for x in lengths)}"
    with open(os.path.join(args.out, f"results_{tag}.json"), "w") as f:
        json.dump({"results": res["results"], "wall_s": time.time() - t0, "n_per_cell": args.n,
                   "lengths": lengths, "tasks": tasks, "seed": args.seed}, f, indent=2)
    with open(os.path.join(args.out, f"samples_{tag}.json"), "w") as f:
        json.dump(res.get("samples", {}), f)
    print(f"== RULER {args.arm} n={args.n} wall={time.time() - t0:.0f}s")
    for task in tasks:
        row = res["results"].get(task, {})
        cells = {str(L): row.get(f"{L},none", row.get(str(L))) for L in lengths}
        print(task, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in cells.items()})


if __name__ == "__main__":
    main()
