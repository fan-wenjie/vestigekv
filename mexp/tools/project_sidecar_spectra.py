#!/usr/bin/env python3
"""Reduce the sidecar dumps to the numbers the appendix actually quotes.

    ~/.conda/envs/sglang-dev/bin/python mexp/tools/project_sidecar_spectra.py --write

The four `sidecardump*` directories hold 10.4 GB of whole pool rows, and the
archive that ships with the paper is 10.57 GB because of them -- 98.6% of it,
for a claim that is six numbers long. This writes those six numbers, per
request, so the appendix's spectral claim can be checked without the dumps and
the dumps can be excluded the way `caldump/` already is.

It is the same rule the RULER samples follow: keep the originals on disk, ship
the projection. What makes a projection legitimate rather than a convenient
deletion is that it carries what the claim rests on. The claim is that RULER's
sidecar has an isolated component above the low-pass cutoff which moves a large
share of tier 1's kept set, and that real prose does not. So the projection
carries, per request: the peak bin above the cutoff, its height against the
median bin there, and the share of the top-3% sigma set that survives notching
it out. A reader who wants the spectra themselves re-runs the registered dump
job named in the README.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "results", "kimi")
OUT = os.path.join(RESULTS, "sidecar_spectra.json")
sys.path.insert(0, os.path.join(ROOT, "mexp", "kimi"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    import make_spectrum_numbers as spec

    conds = {"Ruler": ["sidecardump", "sidecardump5"],
             "Prose": ["sidecardump_prose", "sidecardump_prose5"]}
    out = {"layer": spec.LAYER, "kappa": spec.KAPPA, "block": spec.BLOCK,
           "note": "per-request (peak bin above cutoff, peakiness, kept share "
                   "surviving a notch); raw dumps excluded from the archive, "
                   "regenerate with the registered sidecardump jobs",
           "conditions": {}}
    for mac, dirs in conds.items():
        rows = [r for d in dirs for r in spec.measure(os.path.join(RESULTS, d))]
        if not rows:
            raise SystemExit(f"ABORT: no dumps for {mac}; nothing to project")
        out["conditions"][mac] = [
            {"peak_bin": b, "peakiness": round(p, 1), "kept_share": round(k, 4)}
            for b, p, k in rows]
        print(f"  {mac:6} {len(rows)} requests  bins "
              f"{sorted({r['peak_bin'] for r in out['conditions'][mac]})}")
    if a.write:
        with open(OUT, "w") as f:
            json.dump(out, f, indent=1)
        print(f"wrote {OUT} ({os.path.getsize(OUT)} bytes) against "
              f"{sum(os.path.getsize(p) for p in glob.glob(os.path.join(RESULTS, 'sidecardump*', '*.pt'))) / 1e9:.2f} GB of dumps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
