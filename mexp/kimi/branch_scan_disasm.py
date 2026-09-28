"""Disassemble the branch-only scan against the served scan it replaces.

    python mexp/kimi/branch_scan_disasm.py [--out <dir>]

`branch_scan._branch_scan_kernel` is a fork of `batched_step._scan_batched_kernel`
with the sketch load and the second dot deleted. The fork exists precisely
because deleting work cannot be done as a predicated arm inside the row loop --
both arms' operands stay live across the body and the allocator spills, which
taxes every step rather than the ones the feature fires on
(.claude/rules/disassemble-check-for-spills.md).

That rule asks two questions of an arm written to be cheaper, and this answers
both: STACK 0 with no LDL/STL, AND the instruction the cheap arm was written
for -- the TMA gather of the sidecar -- still present. A fork that dropped the
sketch and also lost its asynchronous loads would be a wash the timings could
not explain.
"""

import argparse
import os
import re
import subprocess

import torch
import triton

P, AM, H, R = 2, 8192, 32, 64


def _inputs(dev):
    from sglang.srt.layers.attention.vestigekv import defaults as D
    from sglang.srt.layers.attention.vestigekv.fused_prologue import (
        _pool_read_mode,
        _set_tma_allocator,
    )

    _set_tma_allocator(dev)
    arena = P * AM
    g = torch.Generator(device=dev).manual_seed(0)
    return dict(
        mode=_pool_read_mode(dev),
        arena=arena,
        pool=torch.randn(arena, D.LATENT_DIM, device=dev, dtype=torch.bfloat16),
        arch=torch.randint(0, arena, (arena,), device=dev, dtype=torch.int32,
                           generator=g),
        csk=torch.randn(arena, R, device=dev, dtype=torch.float16),
        aidx=torch.randint(0, arena, (arena,), device=dev, dtype=torch.int32,
                           generator=g),
        rho=torch.rand(arena, device=dev),
        qside_t=torch.randn(P, D.SIDECAR_DIM, H, device=dev, dtype=torch.bfloat16),
        qsk_t=torch.randn(P, R, H, device=dev, dtype=torch.float16),
        qres=torch.rand(P, H, device=dev),
        max1g=torch.zeros(P, H, device=dev),
        a_len=torch.full((P,), AM, dtype=torch.int64, device=dev),
        a_off=torch.arange(P, dtype=torch.int64, device=dev) * AM,
        cc=torch.full((P,), 0.2, device=dev),
        hit=torch.zeros(arena, dtype=torch.int8, device=dev),
    )


def _compile(which, warps, block, multi, x):
    """Launch once and return the CompiledKernel that launch produced."""
    from sglang.srt.layers.attention.vestigekv import defaults as D
    from sglang.srt.layers.attention.vestigekv.batched_step import (
        _scan_batched_kernel,
    )
    from sglang.srt.layers.attention.vestigekv.branch_scan import (
        _branch_scan_kernel,
    )

    jit = _branch_scan_kernel if which == "branch" else _scan_batched_kernel
    # triton 3.7: JITFunction.device_caches[dev] = (compiled_by_key, ...)
    cache = jit.device_caches[torch.cuda.current_device()][0]
    before = set(cache)
    nb = triton.cdiv(AM, block * multi)
    counts = torch.zeros(P, nb, dtype=torch.int32, device=x["pool"].device)
    grid = (min(nb, D.SCAN_GRID_CAP), P)
    common = dict(
        H=H, DD=D.SIDECAR_DIM, SIDE_POOL=x["mode"], ROW=D.LATENT_DIM,
        KV_OFF=D.KV_LORA_RANK, POOL_ROWS=x["arena"],
        BLOCK_A=block, MULTI=multi, num_warps=warps,
    )
    kbase = torch.full((P,), x["pool"].data_ptr(), dtype=torch.int64,
                       device=x["pool"].device)
    cbase = torch.full((P,), x["csk"].data_ptr(), dtype=torch.int64,
                       device=x["pool"].device)
    if which == "branch":
        jit[grid](x["qside_t"], x["qres"], x["max1g"], None, x["arch"], kbase,
                  x["rho"], x["a_len"], x["a_off"], x["cc"], x["hit"], counts,
                  AM, D.ATTN_SCALE, **common)
    else:
        jit[grid](x["qside_t"], x["qsk_t"], x["qres"], x["max1g"], None,
                  x["arch"], kbase, None, x["aidx"], cbase, x["rho"],
                  x["a_len"], x["a_off"], x["cc"], x["hit"], counts, AM,
                  D.ATTN_SCALE, R=R, CSK_TIER=x["mode"], CSK_ROWS=x["arena"],
                  **common)
    torch.cuda.synchronize()
    new = set(cache) - before
    if not new:
        raise RuntimeError(f"no new compilation for {which} warps={warps}")
    return cache[new.pop()]


def _sass(cubin, path):
    open(path + ".cubin", "wb").write(cubin)
    r = subprocess.run(["nvdisasm", "-c", path + ".cubin"],
                       capture_output=True, text=True)
    if r.returncode:  # a driver-newer-than-nvdisasm mismatch is not fatal
        return None
    body = [re.sub(r"/\*[0-9a-f]{4}\*/|;\s*/\*.*?\*/", "", l).rstrip()
            for l in r.stdout.splitlines()]
    text = "\n".join(l for l in body if l.strip())
    open(path + ".sass", "w").write(text + "\n")
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/branch_scan_disasm")
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--warps", default="2,4")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")
    x = _inputs(dev)
    print(f"pool-read mode={x['mode']} (0 packed, 1 indirect, 2 TMA)")

    kinds = {}
    for which in ("served", "branch"):
        for w in (int(v) for v in a.warps.split(",")):
            k = _compile(which, w, a.block, 1024 // a.block, x)
            name = f"{which}_w{w}"
            md = k.metadata
            sass = _sass(k.asm["cubin"], os.path.join(a.out, name))
            kinds[name] = sass
            # triton 3.7 carries these on the CompiledKernel, not its metadata
            regs = getattr(k, "n_regs", getattr(md, "num_regs", "?"))
            spills = getattr(k, "n_spills", getattr(md, "num_spills", "?"))
            print(f"{name:12s} regs={regs:>4} "
                  f"spills={spills} "
                  f"shared={getattr(md, 'shared', '?')} "
                  f"sass_lines={len(sass.splitlines()) if sass else 'n/a'}")

    for name, sass in kinds.items():
        if not sass:
            continue
        n = {op: sum(1 for l in sass.splitlines() if op in l)
             for op in ("LDL", "STL", "LDGSTS", "UBLKCP", "UTMALDG", "HMMA")}
        print(f"{name:12s} " + " ".join(f"{k}={v}" for k, v in n.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
