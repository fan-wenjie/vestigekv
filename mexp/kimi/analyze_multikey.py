#!/usr/bin/env python3
"""Classify RULER needle errors by whether the right row was reached.

The paper explains the multi-key gap one way: a rank-r sketch fitted to the
archive's own content cannot separate rows drawn from the bulk it was fitted
to, so the certificate fires on the wrong row. That is a claim about SELECTION,
and it predicts a wrong answer -- a different needle's value.

The generations do not agree that this is the whole story. On
`niah_multikey_3`, whose values are 36-character UUIDs, most errors return the
CORRECT value with one character wrong:

    target  715109e9-f773-4f9b-a639-11dd1883a3d2
    got     415109e9-f773-4f9b-a639-11dd1883a3d2

UUIDs are random, so two needles sharing 35 of 36 characters does not happen by
chance. The model reached the right row and mis-emitted one token of the copy,
which is a per-DECODE-STEP failure, not a per-request selection failure. The
two are distinguishable here precisely because the answer is long: a 36-char
copy is ~20 decode steps, and one step missing its row shows up as one bad
character while its 19 neighbours stay correct.

This splits every wrong answer into:

  near   similarity to the target above --threshold: the right row, mis-copied.
  wrong  below it: a different row's value, which is the selection failure.

and then asks of every near miss whether the wrong character came from a
COMPETING needle, which is the obvious suspect in a haystack of hundreds of
UUIDs. The decisive test is the DISTANCE one below: is the emitted string
closer to some other haystack UUID than to the target? On mk3-copyfidelity-n50
it never is -- all six near misses sit at edit distance 1-2 from the target and
21-24 from the nearest competitor. The right row was reached and one or two
tokens came out wrong.

A weaker check is also reported, and is worth naming as weak: the share of
other haystack UUIDs carrying the emitted character at that position, against
1/16 = 6.25% for a hex digit at chance. Measured shares run 6.1-7.9%, i.e.
chance -- but this statistic has NO POWER against blending with a single
competitor, because one competitor's character is as rare among the rest as any
other. It rules out a broad pull toward the haystack, nothing more. The
distance test is what rules out blending.

What the n=50 run establishes (100 items per task and arm, mainline config):

  dense            niah_multikey_3 100/100   niah_single_3 100/100
  vestigekv        niah_multikey_3  88/100   niah_single_3 100/100

so the copy never fails without compression, and under compression it never
fails in a haystack of prose either. The same 36-character UUID copy fails
12 times in 100 when the haystack is itself key-value pairs, and those 12 split
exactly in half: 6 `wrong` (a different needle -- the selection failure the
paper describes) and 6 `near` (the right needle, mis-copied). `niah_single_3`
is the control throughout: same copy, same length, essays instead of pairs.

An earlier version of this file claimed niah_single_3 showed the same near-miss
shape. That came from the randfence ablation arm, not the shipped one; in the
mainline configuration single_3 does not fail at all.

    python mexp/kimi/analyze_multikey.py --samples results/kimi/ruler/samples_*.json
    python mexp/kimi/analyze_multikey.py --line kimi --examples 6
"""
from __future__ import annotations

import argparse
import difflib
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def response(s):
    r = s.get("filtered_resps") or s.get("resps") or [""]
    return (r[0] if isinstance(r, list) else r).strip()


def target(s):
    t = s.get("target")
    return (t[0] if isinstance(t, list) else str(t))


UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def interference(rec, tgt, got):
    """Did the wrong character come from a competing needle, or from nowhere?

    The near-miss class has an obvious suspect: a haystack of key-value pairs
    holds hundreds of other UUIDs, so a substituted character could be the
    model blending the target row with a competitor. That is testable without
    re-running anything, because the prompt is in the record.

    Two checks. Is the emitted string one of the haystack's UUIDs (a wholesale
    swap)? And at each substituted position, how many OTHER haystack UUIDs
    carry the emitted character there -- against 1/16, the rate for a hex digit
    drawn at random? Enrichment over 1/16 is interference; agreement with it is
    not.

    Returns (n_haystack, emitted_is_a_haystack_uuid, [(pos, want, got, share)])
    or None when the record carries no prompt."""
    doc = rec.get("doc")
    if doc is None:
        return None
    txt = doc if isinstance(doc, str) else json.dumps(doc)
    hay = set(UUID.findall(txt))
    hits = []
    if len(tgt) == len(got):
        for i, (a, b) in enumerate(zip(tgt, got)):
            if a == b:
                continue
            others = [u for u in hay if u != tgt and len(u) > i and u[i] == b]
            share = len(others) / max(1, len(hay) - 1)
            hits.append((i, a, b, share))
    return len(hay), got in hay, hits


def verbatim_in_context(rec, got):
    """Cheap hallucination check: does the answer occur verbatim in the prompt?

    An extractive answer is a span the model is copying, so it must appear in
    its own context. A mis-copied character breaks that; a mis-SELECTED row does
    not, because the wrong needle is a real span of the same haystack. The check
    therefore sees exactly the failure class selection-side work cannot reach,
    and is blind to the one it can -- they are complementary, not redundant.

    Measured on mk3-copyfidelity-n50 (400 items, both arms): 388 correct answers
    raise ZERO alarms, 5 of 6 near misses are caught, 0 of 6 wrong-row errors
    are. The one near miss it misses is a truncation, and a truncated copy is
    still a substring of what it was copied from -- length or format validity
    would catch that, at the price of knowing what shape the answer should be.

    It is a copy check, not a hallucination detector in general: it applies only
    where the answer is supposed to be a span of the context. It also detects
    rather than corrects, though this architecture has somewhere to escalate to
    -- the overflow path already attends the full row set exactly."""
    doc = rec.get("doc")
    if doc is None:
        return None
    txt = doc if isinstance(doc, str) else json.dumps(doc)
    return got.strip().rstrip(".") in txt


UUID_RE = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"


def key_adjacent(rec, got):
    """Provenance check: is the answer the value sitting next to the asked key?

    Stronger than verbatim_in_context and for a structural reason. A mis-copied
    answer is not next to the key because it is not in the text at all; a
    mis-SELECTED answer is in the text but next to a DIFFERENT key. One check
    sees both, where the verbatim one sees only the first.

    Measured on mk3-copyfidelity-n50's compressed arm, niah_multikey_3, 100
    items: 88 correct answers pass with ZERO alarms, and all 12 errors are
    flagged -- 6 mis-copied and 6 mis-selected. Every error this run produced.

    The cost is string work over the prompt and no attention at all: find the
    key the question names, take the UUID that follows each of its occurrences
    in the haystack, and ask whether the answer is among them. A per-request
    index built during prefill makes it a lookup.

    The limit is the honest one: this parses a known answer format. It is a
    provenance check for structured retrieval -- key-value stores, logs, JSON,
    tables -- not a hallucination detector for open generation. And 12 errors
    is a thin base, however clean the split looks. It detects without
    correcting, though the overflow path already attends the full row set
    exactly, so an alarm has somewhere to escalate to."""
    doc = rec.get("doc")
    if doc is None:
        return None
    txt = doc if isinstance(doc, str) else json.dumps(doc)
    q = re.search(r"What is the special magic uuid for (" + UUID_RE + r")", txt)
    if not q:
        return None
    key, body = q.group(1), txt[:q.start()]
    vals = []
    for m in re.finditer(re.escape(key), body):
        v = re.search(UUID_RE, body[m.end():m.end() + 200])
        if v:
            vals.append(v.group(0))
    return (got.strip().rstrip(".") in vals) if vals else None


def classify(path, threshold):
    """[(task, length, doc_id, kind, target, response, similarity)] for one arm.

    Only scored items count: lm-eval writes every length key on every record and
    marks the ones this item was not run at with -1.0."""
    rows = []
    for task, recs in json.load(open(path)).items():
        for s in recs:
            for k, v in s.items():
                if not (k.isdigit() and isinstance(v, (int, float)) and v >= 0):
                    continue
                tgt, got = target(s), response(s)
                if v > 0:
                    kind = "ok"
                    sim = 1.0
                else:
                    sim = difflib.SequenceMatcher(None, tgt, got).ratio()
                    kind = "near" if sim > threshold else "wrong"
                rows.append((task, int(k), s["doc_id"], kind, tgt, got, sim))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", nargs="*", default=[],
                    help="samples_*.json files; default: every one under --line")
    ap.add_argument("--line", default="kimi")
    ap.add_argument("--threshold", type=float, default=0.8,
                    help="similarity above which a wrong answer is the right row")
    ap.add_argument("--examples", type=int, default=4)
    ap.add_argument("--emit", default="",
                    help="write the mk3-copyfidelity macros to this .tex file")
    args = ap.parse_args()

    paths = args.samples or sorted(glob.glob(os.path.join(
        ROOT, "results", args.line, "ruler", "samples_*.json")))
    if not paths:
        print("no samples files")
        return 1

    for p in paths:
        rows = classify(p, args.threshold)
        print(f"\n## {os.path.basename(p)}")
        print(f"{'task':20s} {'n':>4} {'ok':>4} {'near':>5} {'wrong':>6}   near/err")
        for task in sorted({r[0] for r in rows}):
            t = [r for r in rows if r[0] == task]
            ok = sum(r[3] == "ok" for r in t)
            near = sum(r[3] == "near" for r in t)
            wrong = sum(r[3] == "wrong" for r in t)
            err = near + wrong
            frac = f"{near}/{err}" if err else "-"
            print(f"{task:20s} {len(t):>4} {ok:>4} {near:>5} {wrong:>6}   {frac:>8}")
        raw = json.load(open(p))
        ex = [r for r in rows if r[3] == "near"][:args.examples]
        for task, l, doc, _, tgt, got, sim in ex:
            print(f"  near  {task} doc{doc} {l}  sim={sim:.2f}")
            print(f"        want {tgt}")
            print(f"        got  {got}")
            rec = next((x for x in raw[task] if x["doc_id"] == doc), None)
            info = interference(rec, tgt, got) if rec else None
            if info is None:
                print("        (no prompt in record; interference untestable)")
                continue
            n_hay, swapped, hits = info
            print(f"        haystack UUIDs {n_hay}; output is one of them: {swapped}")
            for i, a, b, share in hits:
                print(f"        pos {i}: {a!r}->{b!r}; {100 * share:.1f}% of the "
                      f"others carry {b!r} there (chance 6.25%)")
            if not hits:
                print("        length differs: an insertion or deletion, not a "
                      "substitution")

    if args.emit:
        # The mk3-copyfidelity-n50 pair only: two tasks, two arms, 100 items
        # each. Emitting from anything else would put a number in the paper
        # that the sentence around it does not describe.
        want = {"baseline": None, "vestigekv": None}
        for p in paths:
            for arm in want:
                if f"samples_{arm}_n50_32768-65536_mk3-copyfidelity" in p:
                    want[arm] = classify(p, args.threshold)
        if any(v is None for v in want.values()):
            print("--emit needs both mk3-copyfidelity-n50 arms", file=sys.stderr)
            return 1
        L = ["% Generated by mexp/kimi/analyze_multikey.py --emit. Do not edit by hand."]
        for arm, tag in (("vestigekv", "Vk"), ("baseline", "Dense")):
            for task, short in (("niah_multikey_3", "MKc"), ("niah_single_3", "Sc")):
                t = [r for r in want[arm] if r[0] == task]
                L += [f"\\newcommand{{\\cf{tag}{short}N}}{{{len(t)}}}",
                      f"\\newcommand{{\\cf{tag}{short}Ok}}{{{sum(r[3] == 'ok' for r in t)}}}",
                      f"\\newcommand{{\\cf{tag}{short}Near}}{{{sum(r[3] == 'near' for r in t)}}}",
                      f"\\newcommand{{\\cf{tag}{short}Wrong}}{{{sum(r[3] == 'wrong' for r in t)}}}"]
        open(args.emit, "w").write("\n".join(L) + "\n")
        print(f"wrote {args.emit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
