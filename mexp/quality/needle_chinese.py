"""Chinese-corpus needle-in-a-haystack on the serving path (both arms).

Same protocol as needle_serving.py, but haystack and needle are both
Chinese: the filler is 三国演义 (mexp/quality/sanguoyanyi.txt, simplified
script, ~492k tokens under the Kimi tokenizer) and the needle is a
modern-Chinese fact ("{place}的接头暗号是「{code}」。") spliced at a
random depth (10%--80%). Arabic digits essentially never occur in the
classical text, so the code is an unambiguous foreign token sequence.

Placements, places and codes derive from a FIXED seed identical across
arms; run once per arm against the live quality server (radix off,
serial -- this script is serial), then compare. If ctx_len * n_trials
exceeds the corpus, the filler wraps around (continuity is irrelevant:
every trial is an unrelated splice).

Frozen bar: vestigekv intact rate >= dense intact rate, >= 8 trials
or ABORT.

  python mexp/quality/needle_chinese.py gen <ctx_len> <n_trials> <out.json>
  python mexp/quality/needle_chinese.py compare <vk.json> <dense.json> <verdict.json>
"""
import json
import random
import sys

import requests

MODEL = "moonshotai/Kimi-Linear-48B-A3B-Base"  # tokenizer only; shared by both checkpoints
URL = "http://localhost:30000/generate"
CORPUS = "mexp/quality/sanguoyanyi.txt"
PLACE = ["临江县悦来客栈", "白鹭洲渡口茶棚", "青石镇王家当铺", "望云驿马厩", "赤松岗土地庙"]
SEED = 0
GEN = 16

_DIGITS = "零一二三四五六七八九"
_UNITS = ["", "十", "百", "千", "万"]


def chinese_num(n):
    """5-digit int -> Chinese reading (e.g. 48312 -> 四万八千三百一十二)."""
    s = str(n)
    out = []
    for i, d in enumerate(s):
        u = _UNITS[len(s) - 1 - i]
        if d == "0":
            if out and out[-1] != "零":
                out.append("零")
        else:
            out.append(_DIGITS[int(d)] + u)
    return "".join(out).rstrip("零")


def filler(need):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    buf = tok(open(CORPUS).read())["input_ids"]
    reps = need // len(buf) + 1
    return (buf * reps)[:need]


def build_trials(ctx_len, n_trials):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    buf = filler(ctx_len * n_trials)
    trials = []
    for ti in range(n_trials):
        rng = random.Random(SEED * 1000 + ti)
        fill = buf[ti * ctx_len:(ti + 1) * ctx_len]
        place = PLACE[ti % len(PLACE)]
        code = rng.randint(10000, 99999)
        nd = tok(f"\n{place}的接头暗号是「{code}」。\n")["input_ids"]
        q = tok(f"\n问：{place}的接头暗号是什么？\n答：")["input_ids"]
        p = rng.randint(int(ctx_len * 0.1), int(ctx_len * 0.8))
        pre = (fill[:p] + nd + fill[p:])[:ctx_len]
        trials.append({
            "trial": ti, "place": place, "code": code, "depth": p / ctx_len,
            "input_ids": pre + q,
        })
    return trials


def gen(ctx_len, n_trials, out_path):
    trials = build_trials(int(ctx_len), int(n_trials))
    rows = []
    for t in trials:
        r = requests.post(URL, json={
            "input_ids": t["input_ids"],
            "sampling_params": {"max_new_tokens": GEN, "temperature": 0.0},
        }, timeout=7200)
        r.raise_for_status()
        text = r.json()["text"]
        # accept Arabic digits or the Chinese-numeral reading of the code
        intact = str(t["code"]) in text or chinese_num(t["code"]) in text
        rows.append({
            "trial": t["trial"], "place": t["place"], "code": t["code"],
            "depth": round(t["depth"], 3), "intact": bool(intact),
            "answer": text[:80],
        })
        print(f"trial {t['trial']}: depth={t['depth']:.2f} intact={intact} "
              f"answer={text[:40]!r}", flush=True)
    json.dump({"ctx_len": int(ctx_len), "n_trials": len(rows), "seed": SEED,
               "corpus": CORPUS, "rows": rows},
              open(out_path, "w"), indent=1, ensure_ascii=False)


def compare(vk_path, dense_path, out_path):
    vk = json.load(open(vk_path))
    dense = json.load(open(dense_path))
    assert vk["n_trials"] >= 8 and dense["n_trials"] >= 8, "ABORT: <8 trials per arm"
    vi = sum(r["intact"] for r in vk["rows"])
    di = sum(r["intact"] for r in dense["rows"])
    diff = [(r_v["trial"], r_v["intact"], r_d["intact"])
            for r_v, r_d in zip(vk["rows"], dense["rows"])
            if r_v["intact"] != r_d["intact"]]
    res = {
        "ctx_len": vk["ctx_len"], "n_trials": vk["n_trials"],
        "vk_intact": vi, "dense_intact": di,
        "vk_rate": vi / vk["n_trials"], "dense_rate": di / dense["n_trials"],
        "trial_diffs": diff,
        "verdict": "PASS" if vi >= di else "FAIL",
        "protocol": "fixed-seed Chinese needle (sanguoyanyi filler), greedy, radix off, serial",
    }
    json.dump(res, open(out_path, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    if sys.argv[1] == "gen":
        gen(*sys.argv[2:5])
    else:
        compare(sys.argv[2], sys.argv[3], sys.argv[4])
