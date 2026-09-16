"""Compare two RULER arms question by question.

    python mexp/kimi/ruler_diff.py <samples_a.json> <samples_b.json>

For arms that are supposed to be identical -- a stats-on twin, or a change that
moves no row, like the tier-decode router -- any differing answer is a bug, and
this says which task, which length and which document. For arms that are
supposed to differ it gives the paired counts in each direction, which is what
a quality claim rests on rather than the two means.

Prints one line per task and a summary; exit status 1 when the two disagree
anywhere, so a smoke job can gate on it.
"""

import argparse
import collections
import json
import sys


def _answers(path):
    """(task, length, doc_hash) -> (response, score). doc_hash identifies the
    question across arms; the RULER task names its metric after the length, so
    the one numeric key on the record is both the cell and the score. That
    score is -1 on every cell but the first (lm-eval scores one length per
    record and leaves the rest unfilled), so the answers are what compare
    across all 65 cells and the score only where it is >= 0."""
    out = {}
    for task, rows in json.load(open(path)).items():
        for r in rows:
            length = next(k for k in r if k.isdigit())
            out[(task, length, r["doc_hash"])] = (
                json.dumps(r["filtered_resps"]),
                float(r[length]),
            )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--show", type=int, default=5, help="differing questions to print per task")
    args = ap.parse_args()

    A, B = _answers(args.a), _answers(args.b)
    common = sorted(set(A) & set(B))
    if not common:
        print("no questions in common", file=sys.stderr)
        return 2
    only = (len(A) - len(common), len(B) - len(common))
    diff = collections.defaultdict(list)
    for k in common:
        if A[k][0] != B[k][0]:
            diff[k[0]].append(k)
    tasks = sorted({k[0] for k in common})
    width = max(len(t) for t in tasks)
    for t in tasks:
        n = sum(1 for k in common if k[0] == t)
        d = diff.get(t, [])
        scored = [k for k in d if A[k][1] >= 0 and B[k][1] >= 0]
        a_wins = sum(1 for k in scored if A[k][1] > B[k][1])
        b_wins = sum(1 for k in scored if B[k][1] > A[k][1])
        print(f"{t:{width}s}  n={n:4d}  differ={len(d):4d}  scored={len(scored):3d}"
              f"  a>b={a_wins:3d}  b>a={b_wins:3d}")
        for k in d[: args.show]:
            print(f"    {k[1]:>7s} {k[2][:12]}  a={A[k][0][:60]}  b={B[k][0][:60]}")
    total = sum(len(v) for v in diff.values())
    print(f"== {len(common)} questions compared, {total} differ"
          + (f"; a-only {only[0]}, b-only {only[1]}" if any(only) else ""))
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
