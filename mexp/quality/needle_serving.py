"""Needle-in-a-haystack retrieval on the serving path (both arms).

Serving counterpart of the harness needle (harness/e2e.py): one random
unstructured fact ("The secret passcode for {city} is {code}.") is spliced
at a random depth (10%--80%) into fineweb-edu filler; the whole document is
prefilled -- compressed BEFORE the query exists -- and the model is then
asked for the passcode. Greedy decode, exact-string judgement.

The needle placements, cities and codes are derived from a FIXED seed
identical across arms, so arm A and arm B answer byte-identical prompts;
only the cache policy differs. Run once per arm against the live quality
server (radix off, serial -- this script is serial), then compare.

Frozen bar: vestigekv intact rate >= dense intact rate (no needle the
dense arm finds may be lost), over >= 8 trials or ABORT.

  python mexp/quality/needle_serving.py gen <ctx_len> <n_trials> <out.json>
  python mexp/quality/needle_serving.py compare <vk.json> <dense.json> <verdict.json>

Cost note: one trial is one L-token prefill + ~12 decode tokens. L=131072,
n=8 is the default (~2 min/trial); L=524288 n=4 is the long-context
confirmation (~13 min/trial prefill-bound).
"""
import json
import os
import random
import sys

import requests

MODEL = os.environ.get("NEEDLE_TOKENIZER",
                       "moonshotai/Kimi-Linear-48B-A3B-Base")  # tokenizer only
URL = "http://localhost:30000/generate"
CITY = ["Reykjavik", "Montevideo", "Vientiane", "Gaborone", "Ljubljana"]
SEED = 0
GEN = 12
_TAG = MODEL.replace("/", "_")


def filler(path, need):
    try:
        buf = json.load(open(path))
        if len(buf) >= need:
            return buf
    except FileNotFoundError:
        pass
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu", name="sample-10BT",
        split="train", streaming=True,
    )
    buf = []
    for rec in ds:
        buf.extend(tok(rec["text"])["input_ids"])
        if len(buf) >= need:
            break
    json.dump(buf, open(path, "w"))
    return buf


def build_trials(ctx_len, n_trials, filler_path):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    buf = filler(filler_path, ctx_len * n_trials)
    trials = []
    for ti in range(n_trials):
        rng = random.Random(SEED * 1000 + ti)
        fill = buf[ti * ctx_len:(ti + 1) * ctx_len]
        city = CITY[ti % len(CITY)]
        code = rng.randint(10000, 99999)
        nd = tok(f"\nThe secret passcode for {city} is {code}.\n")["input_ids"]
        q = tok(f"\nQuestion: what is the secret passcode for {city}?\nAnswer:")["input_ids"]
        p = rng.randint(int(ctx_len * 0.1), int(ctx_len * 0.8))
        pre = (fill[:p] + nd + fill[p:])[:ctx_len]
        trials.append({
            "trial": ti, "city": city, "code": code, "depth": p / ctx_len,
            "input_ids": pre + q,
        })
    return trials


def gen(ctx_len, n_trials, out_path):
    filler_path = f"results/needle_filler_{ctx_len}_{_TAG}.json"
    trials = build_trials(int(ctx_len), int(n_trials), filler_path)
    rows = []
    for t in trials:
        r = requests.post(URL, json={
            "input_ids": t["input_ids"],
            "sampling_params": {"max_new_tokens": GEN, "temperature": 0.0},
        }, timeout=7200)
        r.raise_for_status()
        text = r.json()["text"]
        intact = str(t["code"]) in text
        rows.append({
            "trial": t["trial"], "city": t["city"], "code": t["code"],
            "depth": round(t["depth"], 3), "intact": bool(intact),
            "answer": text[:80],
        })
        print(f"trial {t['trial']}: depth={t['depth']:.2f} intact={intact} "
              f"answer={text[:40]!r}", flush=True)
    json.dump({"ctx_len": int(ctx_len), "n_trials": len(rows), "seed": SEED,
               "rows": rows}, open(out_path, "w"), indent=1)


def compare(vk_path, dense_path, out_path):
    vk = json.load(open(vk_path))
    dense = json.load(open(dense_path))
    assert vk["n_trials"] >= 8 and dense["n_trials"] >= 8, "ABORT: <8 trials per arm"
    vi = sum(r["intact"] for r in vk["rows"])
    di = sum(r["intact"] for r in dense["rows"])
    # per-trial pairing: same prompts, so a trial-level diff is informative
    diff = [(r_v["trial"], r_v["intact"], r_d["intact"])
            for r_v, r_d in zip(vk["rows"], dense["rows"])
            if r_v["intact"] != r_d["intact"]]
    res = {
        "ctx_len": vk["ctx_len"], "n_trials": vk["n_trials"],
        "vk_intact": vi, "dense_intact": di,
        "vk_rate": vi / vk["n_trials"], "dense_rate": di / dense["n_trials"],
        "trial_diffs": diff,
        "verdict": "PASS" if vi >= di else "FAIL",
        "protocol": "fixed-seed needle, greedy, radix off, serial",
    }
    json.dump(res, open(out_path, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "gen":
        gen(*sys.argv[2:5])
    else:
        compare(sys.argv[2], sys.argv[3], sys.argv[4])
