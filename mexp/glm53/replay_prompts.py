"""Replay saved RULER prompts (from a lm-eval samples json) against the local server.

/v1/completions, greedy, 64 tokens; prints tokens, wall time and whether the target
string appears. Used with SGLANG_DEBUG_VESTIGEKV_STATS=1 to read the per-build VKCAL
telemetry (recall need vs certificate fire) on real prompts.
"""

import argparse
import json
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--samples", required=True)
ap.add_argument("--tasks", default="niah_single_2,ruler_qa_squad,ruler_cwe")
ap.add_argument("--length", type=int, default=65536)
ap.add_argument("--index", type=int, default=0, help="which sample of that length per task")
ap.add_argument("--port", default="30000")
ap.add_argument("--max-tokens", type=int, default=64)
ap.add_argument("--n", type=int, default=1, help="samples per task, from --index on")
ap.add_argument("--ignore-eos", action="store_true", help="always generate max-tokens (calibration dumps need >8 steps)")
args = ap.parse_args()

samples = json.load(open(args.samples))
for task in args.tasks.split(","):
  docs = [d for d in samples[task] if d["doc"]["max_length"] == args.length]
  for d in docs[args.index : args.index + args.n]:
    prompt = d["doc"]["input"] + (d["doc"].get("gen_prefix") or "")
    req = {"model": "nvidia/GLM-5.3-Flash-NVFP4", "prompt": prompt, "max_tokens": args.max_tokens, "temperature": 0,
           "ignore_eos": args.ignore_eos}
    t0 = time.time()
    r = urllib.request.urlopen(
        urllib.request.Request(f"http://127.0.0.1:{args.port}/v1/completions", data=json.dumps(req).encode(),
                               headers={"Content-Type": "application/json"}), timeout=3600)
    out = json.load(r)
    text = out["choices"][0]["text"]
    hit = any(str(t).lower() in text.lower() for t in d["doc"]["outputs"])
    print(f"{task}: {out['usage']['prompt_tokens']} tokens, {time.time() - t0:.0f}s, hit={hit}, text={text[:80]!r}", flush=True)
