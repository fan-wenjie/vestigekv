#!/usr/bin/env python3
"""MRCR v2 against a live sglang server.

    python mexp/exp/run_mrcr.py --arm vestigekv --port 30000 --model <path> \
        --needles 2 --lengths 8192_16384,32768_65536 --n 24 --out results/kimi/mrcr

MRCR (multi-round co-reference resolution) hides several near-identical asks
in one long synthetic conversation -- "write a poem about tapirs", eight times
-- and then asks for the i-th one. It is the adversarial case for this
project's tier 2 by construction: the distractors are not merely similar to
the target, they are the SAME request answered before, so a rank-r sketch
fitted to the archive's own content has to separate rows drawn from exactly
the bulk it was fitted to. RULER's niah_multikey already showed that as the
one place VestigeKV loses; MRCR is that failure mode with the difficulty dial
turned up and a graded score instead of a hit/miss.

Requests go to /v1/completions with the raw prompt, the same path
run_ruler.py uses: the dataset ships a flattened "User: ... Assistant: ..."
transcript, so a chat template would wrap a transcript in another transcript.
Serial and greedy.

GRADING is OpenAI's, unchanged (openai/mrcr README): the model must prepend a
random string given in the question; if it does not, the score is 0; if it
does, both sides are stripped of it and compared with
difflib.SequenceMatcher(None, response, answer).ratio().

We record the two halves of that score separately as well as together, and
that is the point of this client rather than a detail of it. A zero from a
missing prefix is an instruction-following failure; a low ratio with the
prefix present is a retrieval or fidelity failure. Averaged into one number
they are indistinguishable, and the two call for opposite fixes -- the first
says the model never understood the ask, the second says the cache did not
give it the rows. Any claim this project makes about MRCR has to say which it
measured.

DATA. giulio98/MRCR_v2_common, the length-bucketed MRCR v2 packaging: its bin
boundaries are the official ones ([4096,8192], (8192,16384], ... ,
(524288,1048576]) and its rows carry the Gemini-style random-prefix ask. There
is no dataset under an openai/ or google/ namespace named v2; this is the
artifact, and the record names it so a reader knows what was run.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from difflib import SequenceMatcher

REPO = "giulio98/MRCR_v2_common"
PREPEND = re.compile(r"Prepend (\S+) to ", re.I)
# "the second essay about flamingoes" -- which instance was asked for. Two
# samples can share a context AND a prefix and differ only here, so the index
# is what separates a wrong-instance failure from a degraded-content one.
ORDINAL = re.compile(
    r"Prepend \S+ to the (\w+)\b", re.I)
# the official bins, smallest first; a --lengths value must be one of these
BUCKETS = ["4096_8192", "8192_16384", "16384_32768", "32768_65536",
           "65536_131072", "131072_262144", "262144_524288", "524288_1048576"]


def clean(response: str) -> str:
    """The completions path's equivalent of a chat message's content.

    The prompt ends in "Assistant:" and the model continues, so its first
    token is the space that would follow a colon; OpenAI's reference runs
    through chat completions, where that space is not part of the content.
    Leaving it in makes startswith() fail on a response that DID prepend the
    string, which scores an instruction-following success as a zero. Stripping
    leading whitespace restores parity with the reference; nothing else about
    the response is touched.
    """
    return response.lstrip()


def grade(response: str, answer: str, prefix: str) -> float:
    """OpenAI's MRCR grader, verbatim in behaviour, on the cleaned response."""
    response = clean(response)
    if not response.startswith(prefix):
        return 0.0
    return float(SequenceMatcher(None, response.removeprefix(prefix),
                                 answer.removeprefix(prefix)).ratio())


def take(rows, n):
    """n rows spread across the file, not the first n.

    A bin holds many questions over FEW conversations -- the 2needle 8k-16k
    bin is 96 rows over 3 contexts -- and consecutive rows are the same
    conversation asked for the first, then the second instance. Slicing the
    head would measure one haystack and call it n samples. An even stride
    spans the contexts and the instance indices both.
    """
    if n >= len(rows):
        return rows
    step = len(rows) / n
    return [rows[int(i * step)] for i in range(n)]


def load(needles: int, bucket: str):
    from huggingface_hub import list_repo_files, hf_hub_download
    import pyarrow.parquet as pq
    cfg = f"{needles}needle_in_{bucket}"
    files = [f for f in list_repo_files(REPO, repo_type="dataset")
             if f.startswith(cfg + "/")]
    if not files:
        raise SystemExit(f"ABORT: {REPO} has no config {cfg}")
    rows = []
    for f in sorted(files):
        t = pq.read_table(hf_hub_download(REPO, f, repo_type="dataset"))
        cols = t.column_names
        for i in range(t.num_rows):
            rows.append({c: t.column(c)[i].as_py() for c in cols})
    return rows


def complete(port: int, prompt: str, max_tokens: int, timeout: int):
    body = json.dumps({
        "model": "default", "prompt": prompt, "max_tokens": int(max_tokens),
        "temperature": 0.0, "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return d["choices"][0]["text"], d.get("usage", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--needles", type=int, default=2, choices=(2, 4, 8))
    ap.add_argument("--lengths", default="8192_16384",
                    help="comma list of official bins, e.g. 8192_16384,32768_65536")
    ap.add_argument("--n", type=int, default=24, help="samples per bin")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--timeout", type=int, default=1800)
    a = ap.parse_args()

    bins = [b.strip() for b in a.lengths.split(",") if b.strip()]
    bad = [b for b in bins if b not in BUCKETS]
    if bad:
        raise SystemExit(f"ABORT: {bad} are not official MRCR bins; pick from {BUCKETS}")

    os.makedirs(a.out, exist_ok=True)
    stem = f"{a.arm}_{a.needles}needle" + (f"_{a.tag}" if a.tag else "")
    rec_path = os.path.join(a.out, f"results_{stem}.json")
    smp_path = os.path.join(a.out, f"samples_{stem}.json")

    rows_out, samples = [], []
    t0 = time.time()
    for b in bins:
        data = take(load(a.needles, b), a.n)
        print(f"[mrcr] {a.needles}needle {b}: {len(data)} samples", flush=True)
        for i, r in enumerate(data):
            q = r["question"]
            m = PREPEND.search(q)
            if not m:
                raise SystemExit(
                    f"ABORT: no 'Prepend <s> to' in {b} sample {i}; the grader "
                    f"needs the random string and guessing it would score "
                    f"every arm the same way for the wrong reason")
            prefix = m.group(1)
            ans = r["answers"][0] if isinstance(r["answers"], list) else r["answers"]
            if not ans.startswith(prefix):
                raise SystemExit(
                    f"ABORT: {b} sample {i} ground truth does not start with "
                    f"its own prefix {prefix!r}; the row is malformed")
            # `examples` is already the head of `context` in this packaging;
            # concatenating it again would change the prompt under the arm's
            # feet and the length bin would stop meaning what it says.
            prompt = r["context"] + "\n\n" + q
            try:
                text, usage = complete(a.port, prompt, r["max_new_tokens"], a.timeout)
            except (urllib.error.URLError, OSError) as e:
                raise SystemExit(f"ABORT: request failed on {b} sample {i}: {e}")
            text = clean(text)
            s = grade(text, ans, prefix)
            has_prefix = text.startswith(prefix)
            om = ORDINAL.search(q)
            asked = om.group(1).lower() if om else None
            # the ratio the response WOULD have scored had the prefix been
            # right: the retrieval half of the metric, separated from the
            # instruction-following half.
            ratio = float(SequenceMatcher(
                None, text.removeprefix(prefix) if has_prefix else text,
                ans.removeprefix(prefix)).ratio())
            rows_out.append({"bin": b, "i": i, "score": s, "prefix_ok": has_prefix,
                             "ratio": ratio, "asked": asked,
                             "prompt_tokens": usage.get("prompt_tokens"),
                             "completion_tokens": usage.get("completion_tokens")})
            samples.append({"bin": b, "i": i, "prefix": prefix,
                            "response": text, "answer": ans})
            if (i + 1) % 8 == 0:
                cur = [x["score"] for x in rows_out if x["bin"] == b]
                print(f"    {i + 1}/{len(data)} mean={statistics.mean(cur):.3f}",
                      flush=True)

    by_bin = {}
    for b in bins:
        v = [x for x in rows_out if x["bin"] == b]
        if not v:
            continue
        by_bin[b] = {
            "n": len(v),
            "score": statistics.mean(x["score"] for x in v),
            "prefix_rate": statistics.mean(1.0 if x["prefix_ok"] else 0.0 for x in v),
            "ratio_given_prefix": (
                statistics.mean(x["ratio"] for x in v if x["prefix_ok"])
                if any(x["prefix_ok"] for x in v) else None),
        }
    rec = {"arm": a.arm, "model": a.model, "needles": a.needles, "bins": bins,
           "n_per_bin": a.n, "dataset": REPO, "grader": "openai/mrcr README",
           "wall_s": round(time.time() - t0, 1), "by_bin": by_bin, "rows": rows_out}
    json.dump(rec, open(rec_path, "w"))
    json.dump(samples, open(smp_path, "w"))
    print(f"\nwrote {rec_path}")
    for b, v in by_bin.items():
        rg = v["ratio_given_prefix"]
        print(f"  {b:>16}  n={v['n']:<3} score={v['score']:.3f}  "
              f"prefix={v['prefix_rate']:.2f}  "
              f"ratio|prefix={'n/a' if rg is None else f'{rg:.3f}'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
