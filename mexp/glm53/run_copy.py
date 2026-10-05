"""Sustained verbatim copy: the failure mode every retrieval probe on this line is blind to.

Why this client exists. multikey, gsm8k and longbench1 all fail to resolve tier 1
-- tier 1 ALONE scores 0.000 on multikey at every rho while the served arm scores
1.000 -- so the kept set has no measured lower bound and rho is being chosen on
bytes alone. A needle lookup needs ONE archived row and tier 2 can always fetch
it; reproducing a long span verbatim needs a contiguous RUN of rows, a new
position every decode step, which is the work the kept set is supposed to do.

Two properties this task has and a needle does not:

1. The score is GRADED. "Copied 137 of 512 words before derailing" ranks
   configurations; 1.000-or-0.000 cannot. That is what a lower bound on kept
   needs.
2. The un-guessable part is separable. The span is natural prose carrying
   arbitrary numbers, so word accuracy can be inflated by fluent paraphrase
   while NUMBER accuracy cannot: a model that did not attend to the span cannot
   invent its digits. Both are reported; the numeric one is the sharp signal.

Design decisions that are load-bearing:

- The filler is natural text from the same corpus as the span, never synthetic.
  A NoPE cache only develops spectral structure on natural text, and the DSA
  fence owns the synthetic regime -- a synthetic filler would measure the
  fallback rather than the method.
- The span is natural prose too, NOT a random word salad. Tier 1 keeps rows by
  anomaly rank, so a salad dropped into natural filler is maximally anomalous
  and would be preferentially kept: the test would pass for a reason that has
  nothing to do with whether kept covers an ordinary span.
- Prompts are built from a seeded RNG and depend only on (seed, sample index),
  so every arm sees byte-identical prompts.
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CORPUS = os.path.join(ROOT, "mexp", "quality", "gsm8k_platinum.jsonl")

WORD = re.compile(r"[A-Za-z0-9$%.,'-]+")
NUMBER = re.compile(r"^[$]?\d[\d,.]*%?$")


def load_passages():
    """Natural English passages, dense with arbitrary numbers: the gsm8k items."""
    out = []
    with open(CORPUS) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            t = " ".join(str(r.get(k, "")) for k in ("question", "answer")).strip()
            t = " ".join(t.split())
            if len(t) > 200:
                out.append(t)
    if not out:
        raise SystemExit(f"no passages in {CORPUS}")
    return out


def build_prompt(tok, passages, rng, ctx_tokens, span_words, span_pos):
    """A context of `ctx_tokens` with a `span_words` block to copy at `span_pos`."""
    # The span: consecutive passages until it is long enough, so it reads as one
    # continuous document rather than a shuffle.
    span, i = [], rng.randrange(len(passages))
    while len(span) < span_words:
        span.extend(WORD.findall(passages[(i + len(span)) % len(passages)]))
        i += 1
    span = span[:span_words]
    span_text = " ".join(span)

    head = "DOCUMENT\n\n"
    marked = f"\n\n[BLOCK START]\n{span_text}\n[BLOCK END]\n\n"
    tail = (
        "\n\nTask: reproduce the text between [BLOCK START] and [BLOCK END] exactly, "
        "word for word, with no commentary and no summary. Output the block text only."
    )
    budget = ctx_tokens - len(tok(head + marked + tail)["input_ids"])
    if budget <= 0:
        raise SystemExit(f"--ctx {ctx_tokens} too small for a {span_words}-word span")

    # Filler drawn from passages the span did not use, split either side of it.
    filler, n = [], 0
    j = (i + 50) % len(passages)
    while n < budget:
        p = passages[j % len(passages)]
        filler.append(p)
        n += len(tok(p + " ")["input_ids"])
        j += 1
    before = max(1, int(len(filler) * span_pos))
    body = head + " ".join(filler[:before]) + marked + " ".join(filler[before:]) + tail
    return body, span


def score(out_text, span):
    """Leading verbatim run and total accuracy, overall and on numbers alone."""
    got = WORD.findall(out_text or "")
    # Align on the span's first three words: the model may preface the copy.
    start = 0
    if len(span) >= 3:
        for k in range(len(got) - 2):
            if got[k:k + 3] == span[:3]:
                start = k
                break
    got = got[start:]
    lead = 0
    while lead < len(span) and lead < len(got) and got[lead] == span[lead]:
        lead += 1
    pos = {}
    for w in got:
        pos[w] = pos.get(w, 0) + 1
    hit = 0
    for w in span:
        if pos.get(w, 0) > 0:
            pos[w] -= 1
            hit += 1
    # Numbers on their own: a multiset overlap that fluent paraphrase cannot carry.
    nums = [w for w in span if NUMBER.match(w)]
    cg = {}
    for w in got:
        if NUMBER.match(w):
            cg[w] = cg.get(w, 0) + 1
    nhit = 0
    for w in nums:
        if cg.get(w, 0) > 0:
            cg[w] -= 1
            nhit += 1
    return {
        "lead": lead / len(span),
        "lead_words": lead,
        "bag": hit / len(span),
        "num": (nhit / len(nums)) if nums else None,
        "n_num": len(nums),
        "out_words": len(got),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="30000")
    ap.add_argument("--model", default="nvidia/GLM-5.3-Flash-NVFP4")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--span-words", type=int, default=384)
    ap.add_argument("--span-pos", type=float, default=0.1)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    import random
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    passages = load_passages()
    print(f"corpus: {len(passages)} passages; ctx={a.ctx} span={a.span_words}w "
          f"pos={a.span_pos} n={a.n} seed={a.seed}", flush=True)

    rows, t0 = [], time.time()
    for s in range(a.n):
        rng = random.Random((a.seed << 20) ^ s)
        body, span = build_prompt(tok, passages, rng, a.ctx, a.span_words, a.span_pos)
        ntok = len(tok(body)["input_ids"])
        req = {"model": a.model, "messages": [{"role": "user", "content": body}],
               "max_tokens": a.max_tokens, "temperature": 0}
        r = urllib.request.urlopen(
            urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/chat/completions",
                                   data=json.dumps(req).encode(),
                                   headers={"Content-Type": "application/json"}),
            timeout=3600)
        o = json.load(r)
        m = o["choices"][0]["message"]
        # A thinking model can emit the copy inside reasoning_content and a
        # summary in content; score whichever carries the copy (needle.py hit
        # the same thing), and say which, because they are not the same event.
        c, rc = m.get("content") or "", m.get("reasoning_content") or ""
        sc, src = score(c, span), "content"
        s2 = score(rc, span)
        if s2["lead"] > sc["lead"]:
            sc, src = s2, "reasoning"
        sc.update(sample=s, prompt_tokens=ntok, usage=o["usage"], src=src)
        rows.append(sc)
        print(f"[{s+1}/{a.n}] prompt={ntok} lead={sc['lead']:.3f} ({sc['lead_words']}w) "
              f"bag={sc['bag']:.3f} num={sc['num']} src={src}", flush=True)

    def mean(k):
        v = [r[k] for r in rows if r[k] is not None]
        return sum(v) / len(v) if v else float("nan")

    print(f"\nCOPY n={len(rows)} ctx={a.ctx} span={a.span_words} "
          f"lead={mean('lead'):.4f} bag={mean('bag'):.4f} num={mean('num'):.4f} "
          f"wall={time.time()-t0:.0f}s", flush=True)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({"args": vars(a), "rows": rows,
                       "lead": mean("lead"), "bag": mean("bag"), "num": mean("num")}, f)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
