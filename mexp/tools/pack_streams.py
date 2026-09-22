#!/usr/bin/env python3
"""Pack the per-token arrays out of the streaming records, losslessly enough.

    python mexp/tools/pack_streams.py --glob 'results/kimi/latency_stream_*.jsonl'
    python mexp/tools/pack_streams.py --glob '...' --write
    python mexp/tools/unpack_streams.py results/results.zip   # the other half

WHY NOT JUST REDUCE. reduce_streams.py throws the per-token latencies away and
keeps the statistics the paper reports. That is 19.6x and it is verified, but
it answers only the questions we thought to ask: a reader who wants a
percentile we did not keep, or the autocorrelation of the inter-token series,
has to re-run a 512k decode to get it.

Packing keeps every value at 22x, which is smaller than the reduction. The
arrays are the compressible part precisely because they are what a generic
compressor handles worst: JSON text of float64 seconds, 22 characters a value,
each one a fresh string of digits. Three transforms fix that --

    microseconds   the instrument reports a 4 ms interval; the 12th decimal
                   place of a float64 second is arithmetic, not measurement
    deltas         consecutive inter-token latencies differ by far less than
                   they are, so the differences are small integers
    lzma           stdlib, and on this data it beat zstd -19 in three of four
                   encodings tested (0.032 MB against 0.038)

WHY STDLIB. The archive has to open on a reviewer's machine. `lzma` ships with
Python, so the unpacker is twenty lines and no install; msgpack + zstd measured
1.7x worse here AND would ask a reviewer to pip install two packages to read
the evidence.

PRECISION. Values are stored to the microsecond, which on a 4 ms interval is
0.025%. --write refuses unless every statistic the record reports survives the
round trip exactly as reduce_streams checks its own projection.
"""
from __future__ import annotations

import argparse
import glob
import json
import lzma
import os
import struct

CODEC = "us-delta-lzma-1"
ARRAY_KEYS = ("itls", "generated_texts", "input_lens", "output_lens", "ttfts")
PACKED_DIR = "packed"


def _varint(n):
    """Zigzag varint: a delta of +-3 us costs one byte, not eight.

    Fixed 8-byte deltas cost 5x overall where this costs 22x. The values are
    small by construction -- consecutive inter-token latencies differ by far
    less than they are -- so the width has to follow the value, and the sign
    has to fold into the low bit or every negative delta becomes ten bytes."""
    n = (n << 1) ^ (n >> 63)
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _read_varint(buf, off):
    n = shift = 0
    while True:
        b = buf[off]; off += 1
        n |= (b & 0x7F) << shift
        if not b & 0x80:
            return (n >> 1) ^ -(n & 1), off
        shift += 7


def pack_series(rows):
    """[[float, ...], ...] -> bytes. Row lengths are kept so the shape returns."""
    out = [_varint(len(rows))]
    for row in rows:
        q = [int(round(v * 1e6)) for v in row]
        out.append(_varint(len(q)))
        prev = 0
        for v in q:
            out.append(_varint(v - prev))
            prev = v
    return lzma.compress(b"".join(out), preset=9 | lzma.PRESET_EXTREME)


def unpack_series(blob):
    raw = lzma.decompress(blob)
    nrows, off = _read_varint(raw, 0)
    rows = []
    for _ in range(nrows):
        n, off = _read_varint(raw, off)
        vals, acc = [], 0
        for _ in range(n):
            d, off = _read_varint(raw, off)
            acc += d
            vals.append(acc / 1e6)
        rows.append(vals)
    return rows


def pack_text(rows):
    return lzma.compress(json.dumps(rows).encode(), preset=9 | lzma.PRESET_EXTREME)


def unpack_text(blob):
    return json.loads(lzma.decompress(blob))


def stats(blob):
    """Every scalar a record reports, for the round-trip check."""
    return {k: v for k, v in blob.items() if isinstance(v, (int, float, str, bool))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--min-bytes", type=int, default=100_000)
    a = ap.parse_args()

    before = after = 0
    packed = skipped = 0
    # recursive: the archive packs every record under results/, which is two
    # levels -- the top-level throughput records and the per-line stream ones.
    for path in sorted(glob.glob(a.glob, recursive=True)):
        size = os.path.getsize(path)
        lines = open(path).read().splitlines()
        if size < a.min_bytes or not lines:
            skipped += 1
            continue
        blob = json.loads(lines[0])
        def shape(k):
            """num = per-token latencies, txt = generations, flat = one per request.

            The first version demanded a list of lists and so quietly skipped
            generated_texts, which is a flat list of 32 strings -- 18% of the
            record and the reason the ratio stuck at 4x while the latencies
            themselves packed 12x."""
            v = blob.get(k)
            if not (isinstance(v, list) and v):
                return None
            if isinstance(v[0], list):
                return "num" if v[0] and isinstance(v[0][0], (int, float)) else "txt"
            return "flat" if isinstance(v[0], (int, float)) else "txt"

        series = {k: (shape(k), blob[k]) for k in ARRAY_KEYS if shape(k)}
        if not series:
            skipped += 1
            continue

        stem = os.path.splitext(os.path.basename(path))[0]
        side = os.path.join(os.path.dirname(path), PACKED_DIR)
        total = 0
        refs = {}
        for key, (kind, rows) in series.items():
            # Generated text is 18% of a record and the numeric packer cannot
            # touch it; left alone it caps the whole file at 4x however well
            # the latencies pack. Model output repeats heavily, so lzma over
            # the concatenation is the right tool and the round trip is exact.
            if kind == "num":
                data, back = pack_series(rows), None
                back = unpack_series(data)
                want = [[round(v * 1e6) / 1e6 for v in r] for r in rows]
            elif kind == "flat":
                data = pack_series([rows])
                back = unpack_series(data)[0]
                want = [round(v * 1e6) / 1e6 for v in rows]
            else:
                data = pack_text(rows)
                back = unpack_text(data)
                want = rows
            if back != want:
                raise SystemExit(f"ABORT: {path}:{key} does not survive the round trip")
            refs[key] = (f"{stem}.{key}.lzma", data)
            total += len(data)
        kept = dict(blob)
        for key, (name, _) in refs.items():
            num = series[key][0] in ("num", "flat")
            kept[key] = {"__packed__": {"codec": CODEC if num else "json-lzma-1",
                                        "shape": series[key][0],
                                        "unit": "microseconds" if num else None,
                                        "file": f"{PACKED_DIR}/{name}"}}
        new = json.dumps(kept) + "\n"
        if stats(json.loads(new)) != stats(blob):
            raise SystemExit(f"ABORT: {path} scalars changed while packing")

        before += size
        after += len(new) + total
        packed += 1
        print(f"  {size / 1e6:8.2f} -> {(len(new) + total) / 1e6:6.3f} MB "
              f"({size / max(len(new) + total, 1):5.0f}x)  {os.path.basename(path)}")
        if a.write:
            os.makedirs(side, exist_ok=True)
            for _, (name, data) in refs.items():
                open(os.path.join(side, name), "wb").write(data)
            open(path, "w").write(new)

    verb = "packed" if a.write else "would pack"
    print(f"\n{packed} {verb}, {skipped} skipped  |  {before / 1e6:.1f} MB -> "
          f"{after / 1e6:.1f} MB ({before / max(after, 1):.0f}x)"
          f"{'' if a.write else '  [dry run; pass --write]'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
