"""Per-step ITL of two arms at matched decode context.

A median over a 4k->131k sweep is the step at the MEDIAN context, not the final
one, so a median ratio answers a different question from a trace taken at 124k.
This unpacks both arms' ITL series and reports the gap as a function of context,
plus the spike structure that drives each arm's p99.
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from pack_streams import unpack_series  # the archive's own decoder, not a copy


def itls(path):
    d = [json.loads(l) for l in open(path)][-1]
    blob = (Path(path).parent / d["itls"]["__packed__"]["file"]).read_bytes()
    rows = unpack_series(blob)
    out = [v * 1000.0 for v in (rows[0] if len(rows) == 1 else sum(rows, []))]
    # The recorded median is the check that the decode is right.
    got, want = sorted(out)[len(out) // 2], d["median_itl_ms"]
    assert abs(got - want) < 0.02, f"decode mismatch: {got} vs recorded {want}"
    return d, out


def pct(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def main(a, b, in_len=4096):
    da, A = itls(a)
    db, B = itls(b)
    n = min(len(A), len(B))
    print("n=%d  %s vs %s" % (n, Path(a).name.split("126976_")[1][:34],
                              Path(b).name.split("126976_")[1][:34]))
    print("%9s %9s %9s %9s %9s" % ("ctx", "A_med", "B_med", "gap_us", "ratio"))
    edges = [0, 8192, 16384, 32768, 65536, 98304, n]
    for i in range(len(edges) - 1):
        lo, hi = edges[i], min(edges[i + 1], n)
        if hi - lo < 64:
            continue
        ma, mb = pct(A[lo:hi], 0.5), pct(B[lo:hi], 0.5)
        print("%9d %9.3f %9.3f %9.1f %9.4f"
              % (in_len + (lo + hi) // 2, ma, mb, (mb - ma) * 1000, ma / mb))
    for tag, X in (("A", A), ("B", B)):
        med = pct(X, 0.5)
        over = [i for i, v in enumerate(X) if v > med * 1.3]
        gaps = [over[i + 1] - over[i] for i in range(len(over) - 1)]
        mode = max(set(gaps), key=gaps.count) if gaps else 0
        print("%s: med %.3f p90 %.3f p99 %.3f max %.3f | steps >1.3x med: %d "
              "(%.2f%%), modal spacing %d" % (tag, med, pct(X, 0.9), pct(X, 0.99),
              max(X), len(over), 100.0 * len(over) / len(X), mode))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
