"""Diff two torch allocator snapshots (SGLANG_DEBUG_VESTIGEKV_MEM_DIR) by allocation stack.

    python mexp/glm53/memdiff.py results/glm53/memtrace/mem_req5.pickle \
                                  results/glm53/memtrace/mem_req25.pickle [--top 25] [--depth 4]

Live (active_allocated) block bytes are summed per allocation site, a site being the
innermost `depth` frames outside torch itself; the sites are listed by growth
between the two snapshots, with the reserved-but-free bytes as a separate line.
"""

import argparse
import collections
import os
import pickle


def site(frames, depth):
    own = [f for f in frames if "site-packages/torch" not in f["filename"]]
    own = own[:depth] if own else frames[:depth]
    return " < ".join(f"{os.path.basename(f['filename'])}:{f['line']} {f['name']}" for f in own)


def live_by_site(snapshot, depth):
    by = collections.Counter()
    free = 0
    for seg in snapshot["segments"]:
        for blk in seg["blocks"]:
            if blk["state"] == "active_allocated":
                by[site(blk.get("frames", []), depth)] += blk["size"]
            else:
                free += blk["size"]
    return by, free


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--depth", type=int, default=4)
    args = ap.parse_args()
    a, free_a = live_by_site(pickle.load(open(args.a, "rb")), args.depth)
    b, free_b = live_by_site(pickle.load(open(args.b, "rb")), args.depth)
    gib = 2**30
    print(f"live: {sum(a.values()) / gib:.3f} -> {sum(b.values()) / gib:.3f} GiB   "
          f"reserved-free: {free_a / gib:.3f} -> {free_b / gib:.3f} GiB")
    rows = sorted(((b[k] - a.get(k, 0), b[k], k) for k in b), reverse=True)
    print(f"{'growth MiB':>11s} {'live MiB':>9s}  site")
    for growth, live, k in rows[: args.top]:
        print(f"{growth / 2**20:11.1f} {live / 2**20:9.1f}  {k}")
    gone = sorted(((a[k], k) for k in a if k not in b), reverse=True)[:5]
    if gone:
        print("largest sites present only in the first snapshot:")
        for size, k in gone:
            print(f"{size / 2**20:11.1f}  {k}")


if __name__ == "__main__":
    main()
