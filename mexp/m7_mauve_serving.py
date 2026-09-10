"""M7 MAUVE on the serving path (fused final code; protocol frozen from M7).

16 fineweb-edu contexts (T=4096 prefill), 256-token generations at temp 1.0
top-p 0.95, per-context sampling_seed shared across arms (server-side
sampler, so arms differ only through the cache). Run once per arm against
the live server, then score with --score.

  python mexp/m7_mauve_serving.py gen out/m7_ctx.json out/m7_<arm>.json
  python mexp/m7_mauve_serving.py score out/m7_ctx.json out/m7_vk.json \
      out/m7_eng.json out/m7_mauve_fused.json
"""
import json
import sys

import requests

MODEL = "moonshotai/Kimi-Linear-48B-A3B-Base"
T, GEN, N_CTX = 4096, 256, 16
URL = "http://localhost:30000/generate"


def contexts(path):
    try:
        return json.load(open(path))
    except FileNotFoundError:
        from datasets import load_dataset
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu", name="sample-10BT",
            split="train", streaming=True,
        )
        buf, ctxs = [], []
        need = T + GEN
        for rec in ds:
            buf.extend(tok(rec["text"])["input_ids"])
            while len(buf) >= need and len(ctxs) < N_CTX:
                ctxs.append(buf[:need])
                buf = buf[need:]
            if len(ctxs) >= N_CTX:
                break
        json.dump(ctxs, open(path, "w"))
        return ctxs


def gen(ctx_path, out_path):
    ctxs = contexts(ctx_path)
    outs = []
    for ci, ids in enumerate(ctxs):
        r = requests.post(URL, json={
            "input_ids": ids[:T],
            "sampling_params": {
                "max_new_tokens": GEN, "temperature": 1.0, "top_p": 0.95,
                "sampling_seed": 1000 + ci,
            },
        }, timeout=600)
        r.raise_for_status()
        outs.append(r.json()["text"])
        print(f"ctx {ci}: {len(outs[-1])} chars", flush=True)
    json.dump(outs, open(out_path, "w"))


def score(ctx_path, vk_path, eng_path, out_path):
    import mauve
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    ctxs = contexts(ctx_path)
    human = [tok.decode(ids[T : T + GEN]) for ids in ctxs]
    vk = json.load(open(vk_path))
    eng = json.load(open(eng_path))
    assert len(vk) >= 12 and len(eng) >= 12, "ABORT: <12 contexts per arm"
    m_eng = mauve.compute_mauve(
        p_text=human, q_text=eng, featurize_model_name="gpt2-large",
        device_id=0, verbose=False,
    ).mauve
    m_vk = mauve.compute_mauve(
        p_text=human, q_text=vk, featurize_model_name="gpt2-large",
        device_id=0, verbose=False,
    ).mauve
    res = {
        "n_ctx": len(ctxs), "gen_tokens": GEN,
        "mauve_eng": m_eng, "mauve_vk": m_vk,
        "verdict": "PASS" if m_vk >= m_eng - 0.10 else "FAIL",
        "code": "serving path (tree recorded at score time)",
    }
    json.dump(res, open(out_path, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "gen":
        gen(sys.argv[2], sys.argv[3])
    else:
        score(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
