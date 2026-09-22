#!/usr/bin/env python3
"""Do the two lines' operators still compute the same thing at Kimi geometry?

    python mexp/tools/compare_operator_trees.py
    python mexp/tools/compare_operator_trees.py --other ~/vestigekv-wt/engine-glm53

The GLM line parameterised the operators by `Geometry` and the commit that did
it says the validated Kimi Linear path is unchanged. That was an assertion.
`test_vestigekv_geometry.py` checks that a Kimi config derives the KIMI_LINEAR
values, which is a different claim -- it says the constants agree, not that the
kernels produce the same numbers.

This runs one fixed input through both checkouts and compares the outputs bit
for bit. A comparison against a stored golden file would answer a weaker
question: goldens drift with whoever last regenerated them, while the other
tree is the thing actually being claimed equivalent.

Why it matters beyond hygiene. The two lines have diverged enough that a
cherry-pick does not carry work between them, so there is a standing
temptation to keep two copies of the same operators -- and then a fix lands in
one. If this stays green the Kimi line can move onto the parameterised
operators, at which point the DCP work is written once instead of twice. If it
goes red, that migration is unsafe and the reason will be in the diff this
prints.

Each case is run in a subprocess per tree, because both define the same module
names and a single interpreter can only import one of them.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

PY = os.path.expanduser("~/.conda/envs/sglang-dev/bin/python")
HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SIGMA = '''
from sglang.srt.layers.attention.vestigekv.sigma_fused import sigma_fused_from_pool
from sglang.srt.layers.attention.vestigekv import defaults as D
BLOCK = 512
row = D.KV_LORA_RANK + D.SIDECAR_DIM
g = torch.Generator(device="cuda").manual_seed(7)
kbuf = torch.randn(BLOCK, row, generator=g, device="cuda", dtype=torch.bfloat16)
slots = torch.arange(BLOCK, device="cuda", dtype=torch.int64)
sig, hist = sigma_fused_from_pool(kbuf, slots, BLOCK)
OUTPUT = {"sigma": sig, "histogram": hist}
'''

PROLOGUE = '''
from sglang.srt.layers.attention.vestigekv.fused_prologue import fused_prologue
H, R, KV, WIDTH, P, NKm = 32, 64, 512, 576, 6, 768
torch.manual_seed(3)
q = torch.randn(P, H, WIDTH, device="cuda")
kr = torch.randn(P, NKm, WIDTH, device="cuda", dtype=torch.bfloat16)
v = torch.stack([torch.linalg.qr(torch.randn(KV, R, device="cuda"))[0].T.contiguous()
                 for _ in range(P)])
nk = torch.randint(NKm // 2, NKm, (P,), device="cuda", dtype=torch.int64)
thr = torch.rand(P, device="cuda") * 3
m1, qside, qsk, qres = fused_prologue(q, kr, v, nk, thr, 1 / 24.0)
OUTPUT = {"max1g": m1, "qside": qside, "qsk": qsk, "qres": qres}
'''

CASES = {"sigma_fused": SIGMA, "fused_prologue": PROLOGUE}

RUNNER = '''
import sys, torch
sys.path.insert(0, {tree!r} + "/python")
{body}
torch.save({{k: v.cpu() for k, v in OUTPUT.items()}}, {out!r})
print("ok")
'''


def run(tree, body, out):
    r = subprocess.run([PY, "-c", RUNNER.format(tree=tree, body=body, out=out)],
                       capture_output=True, text=True)
    if r.returncode != 0 or "ok" not in r.stdout:
        raise SystemExit(f"ABORT: {tree} failed to produce the case:\n"
                         + (r.stderr.strip()[-1500:] or r.stdout.strip()[-500:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--this", default=os.path.join(HERE, "engine"),
                    help="the tree whose operators are the reference")
    ap.add_argument("--other", default=os.path.expanduser("~/vestigekv-wt/engine-glm53"),
                    help="the tree claimed equivalent at Kimi geometry")
    a = ap.parse_args()

    import torch

    bad = 0
    for name, body in CASES.items():
        with tempfile.TemporaryDirectory() as d:
            pa, pb = os.path.join(d, "a.pt"), os.path.join(d, "b.pt")
            run(a.this, body, pa)
            run(a.other, body, pb)
            ta, tb = torch.load(pa), torch.load(pb)
            for key in ta:
                same = torch.equal(ta[key], tb[key])
                mark = "same" if same else "DIFFERS"
                extra = ""
                if not same:
                    bad += 1
                    d_ = (ta[key].float() - tb[key].float()).abs()
                    extra = f"  max |diff| {d_.max():.3e} over {int((d_ > 0).sum())} values"
                print(f"  {name:16} {key:11} {mark}{extra}")
    print()
    if bad:
        print(f"{bad} output(s) differ: the parameterised operators are NOT a "
              "drop-in at Kimi geometry, and migrating the Kimi line onto them "
              "would move published numbers.")
        return 1
    print("every output bit-identical: the parameterised operators are a "
          "drop-in at Kimi geometry.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
