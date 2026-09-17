"""Disassemble the forked stage 1 with the fence compiled in and compiled out.

    python mexp/kimi/fence_disasm.py [--out <dir>]

`FENCE` is a `tl.constexpr`, so the two settings are two separate compilations
of one source. The claim under test is that they differ by exactly one branch:
FENCE=False should drop the `fetch_ovf` load and the page-table arm, and change
nothing else. Anything else in the diff -- a different register count, a
different tiling, a spill -- is a cost the source does not show and would
explain a gap the timings cannot attribute.

Prints the register and instruction counts for both, then writes the PTX and
SASS of each and a unified diff next to them.
"""

import argparse
import os
import re
import subprocess
import sys

POOL, R1, CAP, FW, R2T = 4096, 4, 512, 128, 1024
LK, LV, H, SPLITS = 576, 512, 16, 8


def _rows(fence):
    import torch

    from sglang.srt.layers.attention.vestigekv.decode_fork import VestigeKVRows

    g = torch.Generator(device="cuda").manual_seed(0)
    z = lambda *s, d=torch.int32: torch.zeros(*s, dtype=d, device="cuda")  # noqa: E731
    return VestigeKVRows(
        slots=torch.zeros(1, dtype=torch.int64, device="cuda"),
        kept_buf=torch.randint(0, POOL, (R1, CAP), dtype=torch.int32, device="cuda", generator=g),
        kept_len=torch.full((R1,), 300, dtype=torch.int32, device="cuda"),
        fetch_buf=torch.randint(0, POOL, (R1, FW), dtype=torch.int32, device="cuda", generator=g),
        fetch_len=torch.full((R1,), 37, dtype=torch.int32, device="cuda"),
        fetch_ovf=z(R1),
        r2t=torch.randint(0, POOL, (R1 - 1, R2T), dtype=torch.int32, device="cuda", generator=g),
        seq=torch.full((1,), 777, dtype=torch.int64, device="cuda"),
        loc=torch.full((1,), POOL - 1, dtype=torch.int64, device="cuda"),
        fence=fence,
    )


def _compile(fence):
    """Run the launcher once and hand back the kernel Triton compiled for it."""
    import torch

    from sglang.srt.layers.attention.vestigekv import decode_fork as df

    pool = torch.randn(POOL, 1, LK, dtype=torch.bfloat16, device="cuda")
    q = torch.randn(1, H, LK, dtype=torch.bfloat16, device="cuda")
    out = torch.zeros(1, H, SPLITS, LV, dtype=torch.float32, device="cuda")
    lse = torch.zeros(1, H, SPLITS, dtype=torch.float32, device="cuda")
    indices = torch.zeros(1024, dtype=torch.int64, device="cuda")
    indptr = torch.tensor([0, 337], dtype=torch.int32, device="cuda")
    splits = torch.full((1,), SPLITS, dtype=torch.int32, device="cuda")

    # triton 3.7: JITFunction.device_caches[dev] = (compiled_by_key, ...)
    cache = df._vk_fwd_grouped_kernel_stage1.device_caches[torch.cuda.current_device()][0]
    before = set(cache)
    df.decode_grouped_att_m_fwd(
        q, pool, pool[:, :, :LV], out, lse, indptr, indices, _rows(fence),
        splits, SPLITS, 1.0 / (LK**0.5), 0.0, has_mla=True,
    )
    torch.cuda.synchronize()
    new = set(cache) - before
    if not new:
        raise RuntimeError(f"no new compilation for fence={fence}; clear ~/.triton/cache")
    return cache[new.pop()]


def _sass(cubin, path):
    open(path + ".cubin", "wb").write(cubin)
    r = subprocess.run(["nvdisasm", "-c", path + ".cubin"], capture_output=True, text=True)
    if r.returncode:  # a driver-newer-than-nvdisasm mismatch is not fatal
        return None
    # drop addresses and scheduling comments so the diff is about instructions
    body = [re.sub(r"/\*[0-9a-f]{4}\*/|;\s*/\*.*?\*/", "", l).rstrip()
            for l in r.stdout.splitlines()]
    text = "\n".join(l for l in body if l.strip())
    open(path + ".sass", "w").write(text + "\n")
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/fence_disasm")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    kinds = {}
    for fence in (False, True):
        k = _compile(fence)
        name = "fence_on" if fence else "fence_off"
        md = k.metadata
        ptx = k.asm["ptx"]
        open(os.path.join(args.out, name + ".ptx"), "w").write(ptx)
        sass = _sass(k.asm["cubin"], os.path.join(args.out, name))
        kinds[name] = (md, ptx, sass)
        print(f"{name:9s} regs={getattr(md, 'num_regs', '?'):>4} "
              f"spills={getattr(md, 'num_spills', '?')} "
              f"shared={getattr(md, 'shared', '?')} "
              f"warps={getattr(md, 'num_warps', '?')} "
              f"ptx_lines={len(ptx.splitlines())} "
              f"sass_lines={len(sass.splitlines()) if sass else 'n/a'}")

    for what, idx in (("ptx", 1), ("sass", 2)):
        a, b = kinds["fence_off"][idx], kinds["fence_on"][idx]
        if a is None or b is None:
            print(f"\n== {what}: not available")
            continue
        pa = os.path.join(args.out, f"fence_off.{what}")
        pb = os.path.join(args.out, f"fence_on.{what}")
        d = subprocess.run(["diff", "-u", pa, pb], capture_output=True, text=True).stdout
        adds = sum(1 for l in d.splitlines() if l.startswith("+") and not l.startswith("+++"))
        dels = sum(1 for l in d.splitlines() if l.startswith("-") and not l.startswith("---"))
        hunks = sum(1 for l in d.splitlines() if l.startswith("@@"))
        open(os.path.join(args.out, f"{what}.diff"), "w").write(d)
        print(f"\n== {what}: {hunks} hunks, +{adds}/-{dels} lines "
              f"-> {os.path.join(args.out, what + '.diff')}")


if __name__ == "__main__":
    sys.exit(main())
