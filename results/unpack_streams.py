#!/usr/bin/env python3
"""Expand the packed arrays in a results record. Ships inside results.zip.

    python unpack_streams.py kimi/latency_stream_4k-520192_vestigekv.jsonl
    python unpack_streams.py --all .

Reads a record whose array fields were replaced by

    "itls": {"__packed__": {"codec": "us-delta-lzma-1", "file": "packed/..."}}

and writes the record back with the arrays in place. Nothing outside the
standard library is needed: `lzma` ships with Python, which is the reason this
archive uses it rather than zstd. On the data here the two were within 15% of
each other and lzma won three comparisons of four -- not enough to justify
asking a reader to install anything in order to read the evidence.

THE FORMAT, in full, so this file is not the only way to read it:

    us-delta-lzma-1   lzma stream. Zigzag varints: row count, then per row a
                      value count followed by that many deltas. Values are
                      microseconds; the first delta is against zero. Divide by
                      1e6 for seconds.
    json-lzma-1       lzma stream holding UTF-8 JSON. Used for generations,
                      where the numeric transform does not apply.

Zigzag means a signed n is stored as (n << 1) ^ (n >> 63), so small negative
deltas stay one byte. Varint means seven bits a byte, high bit set while more
bytes follow.
"""
from __future__ import annotations

import argparse
import glob
import json
import lzma
import os


def read_varint(buf, off):
    n = shift = 0
    while True:
        b = buf[off]
        off += 1
        n |= (b & 0x7F) << shift
        if not b & 0x80:
            return (n >> 1) ^ -(n & 1), off
        shift += 7


def unpack_numeric(blob):
    raw = lzma.decompress(blob)
    nrows, off = read_varint(raw, 0)
    rows = []
    for _ in range(nrows):
        n, off = read_varint(raw, off)
        vals, acc = [], 0
        for _ in range(n):
            d, off = read_varint(raw, off)
            acc += d
            vals.append(acc / 1e6)
        rows.append(vals)
    return rows


def expand(record, base):
    out = dict(record)
    for key, value in record.items():
        meta = value.get("__packed__") if isinstance(value, dict) else None
        if not meta:
            continue
        blob = open(os.path.join(base, meta["file"]), "rb").read()
        if meta["codec"] == "json-lzma-1":
            out[key] = json.loads(lzma.decompress(blob))
        else:
            rows = unpack_numeric(blob)
            out[key] = rows[0] if meta.get("shape") == "flat" else rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="a record, or a directory with --all")
    ap.add_argument("--all", action="store_true", help="every record under path")
    ap.add_argument("--in-place", action="store_true",
                    help="rewrite the records; default prints one to stdout")
    a = ap.parse_args()

    paths = (sorted(glob.glob(os.path.join(a.path, "**", "*.jsonl"), recursive=True))
             if a.all else [a.path])
    for path in paths:
        lines = [l for l in open(path).read().splitlines() if l.strip()]
        base = os.path.dirname(path)
        out = [json.dumps(expand(json.loads(l), base)) for l in lines]
        if a.in_place:
            open(path, "w").write("\n".join(out) + "\n")
            print(f"expanded {path}")
        else:
            print(out[0][:2000])
            if len(paths) > 1:
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
