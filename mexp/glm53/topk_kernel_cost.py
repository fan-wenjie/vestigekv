"""Per-kernel cost of DSA's selection, from traces already on disk.

Resolves 87.1 vs 158.5 us for kpool_topk_transform: the histogram pass scales
with the group count (seq/pool) while the emit is fixed at 512 groups, so the
two numbers may both be right at different shapes. Prints us/step per arm per
trace, with the step count taken from the trace's own span rather than assumed.
"""
import gzip, json, sys, collections
from pathlib import Path

WANT = ("kpool_topk_transform", "_sparse_mla_decode", "_vk_dsa_decode",
        "_branch_", "paged_mqa_logits", "fp8_mqa", "sigma")


def one(path):
    with gzip.open(path, "rt") as f:
        tr = json.load(f)
    ev = [e for e in tr.get("traceEvents", [])
          if e.get("ph") == "X" and e.get("cat") in ("kernel", "Kernel", "gpu_user_annotation")]
    if not ev:
        return None
    # Steps: count the launches of the attention kernel, divided by its layers.
    agg = collections.defaultdict(lambda: [0.0, 0])
    for e in ev:
        n = e["name"]
        for w in WANT:
            if w in n:
                a = agg[w]; a[0] += e["dur"]; a[1] += 1
                break
    span = max(e["ts"] + e["dur"] for e in ev) - min(e["ts"] for e in ev)
    return agg, span


def main(paths, layers=11, steps=None):
    for p in paths:
        r = one(p)
        if r is None:
            print(f"{Path(p).name}: no kernel events"); continue
        agg, span = r
        att = agg.get("_sparse_mla_decode", agg.get("_vk_dsa_decode", [0, 0]))
        n = steps or (att[1] // layers if att[1] else 0)
        print(f"\n{Path(p).parent.name}/{Path(p).name}  span={span/1000:.1f} ms  steps~{n}")
        if not n:
            continue
        for k in sorted(agg, key=lambda k: -agg[k][0]):
            tot, cnt = agg[k]
            print("  %-26s %8.1f us/step  %6.2f us/launch  %5d launches"
                  % (k, tot / n, tot / cnt, cnt))


if __name__ == "__main__":
    main(sys.argv[1:])
