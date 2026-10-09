"""Retrieval accuracy by COMPRESSION PHASE, for a chunk-pooled index.

Why this probe exists. DSA pools kpool=4 consecutive tokens into one index entry
at a fixed stride, aligned to position 0:

    first_pos     = seq_len - q_len
    pool_start_id = first_pos // kpool          # chunk starts are asserted aligned
    n_pools       = q_len // kpool

so phase(p) = p % kpool and group(p) = p // kpool. arXiv:2609.36322 reports up to
40.2 points of retrieval difference across phases on models with this structure,
and its conclusion is that averages conceal it -- which every needle test we run
does, because the generator never controls phase.

The design controls the needle's phase EXACTLY by sending input_ids rather than
text: re-tokenising a decoded string can shift every position by one and silently
destroy the variable under test. The value is chosen to be exactly VAL_TOKENS
tokens long, so at phase 0 it sits inside ONE group and at phase 1..3 it straddles
two -- the sharpest contrast the structure allows.
"""

import argparse, json, os, random, re, sys, urllib.request

VAL_TOKENS = 4


def post(url, obj, timeout=1800):
    r = urllib.request.Request(url, json.dumps(obj).encode(),
                               {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=timeout).read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--port", default="30000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", type=int, default=65536)
    ap.add_argument("--n", type=int, default=30, help="draws PER PHASE")
    ap.add_argument("--kpool", type=int, default=4)
    ap.add_argument("--depth", type=float, default=0.5, help="needle depth in the context")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="results/glm53/phase")
    a = ap.parse_args()
    url = f"http://127.0.0.1:{a.port}"

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)

    filler = ("The grass was still wet from the morning rain and the market "
              "stalls had not yet opened along the narrow street. ")
    f_ids = tok(filler, add_special_tokens=False)["input_ids"]
    lead_ids = tok("The special passcode is ", add_special_tokens=False)["input_ids"]
    tail_ids = tok(". Remember it.", add_special_tokens=False)["input_ids"]
    q_ids = tok("\n\nQuestion: what is the special passcode? Answer with the passcode only.\nAnswer:",
                add_special_tokens=False)["input_ids"]

    rng = random.Random(a.seed)
    # values that tokenise to EXACTLY VAL_TOKENS, so length never confounds phase
    vals = []
    while len(vals) < a.n * a.kpool:
        v = "".join(rng.choice("0123456789") for _ in range(rng.choice([6, 7, 8])))
        ids = tok(v, add_special_tokens=False)["input_ids"]
        if len(ids) == VAL_TOKENS:
            vals.append((v, ids))
    print(f"value pool: {len(vals)} values, each exactly {VAL_TOKENS} tokens", flush=True)

    res = {}
    for phase in range(a.kpool):
        hits = 0
        for k in range(a.n):
            v, v_ids = vals[phase * a.n + k]
            # Choose the prefix LENGTH so the value's first token lands on `phase`.
            #   start = len(pre) + len(lead_ids)  and we need start % kpool == phase
            # so solve for len(pre) directly instead of padding after the fact.
            L0 = int(a.ctx * a.depth) - len(lead_ids)
            L = L0 + ((phase - len(lead_ids) - L0) % a.kpool)
            pre = [f_ids[i % len(f_ids)] for i in range(L)]
            start = len(pre) + len(lead_ids)
            assert start % a.kpool == phase, (start, phase, L)

            ids = pre + lead_ids + v_ids + tail_ids
            while len(ids) < a.ctx - len(q_ids):
                ids += f_ids
            ids = ids[: a.ctx - len(q_ids)] + q_ids
            out = post(url + "/generate",
                       {"input_ids": ids,
                        "sampling_params": {"max_new_tokens": 24, "temperature": 0}})
            txt = out["text"] if isinstance(out, dict) else out[0]["text"]
            ok = v in re.sub(r"[^0-9]", "", txt) or v in txt
            hits += ok
            if k == 0:
                print(f"  phase {phase}: start={start} (start%{a.kpool}={start%a.kpool}) "
                      f"val={v} got={txt.strip()[:40]!r} ok={ok}", flush=True)
        res[phase] = hits / a.n
        print(f"PHASE {phase}: {hits}/{a.n} = {hits/a.n:.4f}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    name = f"{a.arm}_ctx{a.ctx}_n{a.n}" + (f"_{a.tag}" if a.tag else "")
    with open(os.path.join(a.out, name + ".json"), "w") as f:
        json.dump({"arm": a.arm, "ctx": a.ctx, "n": a.n, "kpool": a.kpool,
                   "depth": a.depth, "seed": a.seed, "val_tokens": VAL_TOKENS,
                   "by_phase": res}, f, indent=2)
    print("PHASEDONE", json.dumps(res))


if __name__ == "__main__":
    main()
