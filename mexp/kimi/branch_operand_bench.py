"""The branch operand build against the sketch build it forks, and its SASS.

    python mexp/kimi/branch_operand_bench.py [--rows 65536] [--sweep] [--out <dir>]

Equivalence first, at the bar operand_fused.py itself set: side and csk must
match bit for bit, and rho to ~1 ulp with the FIRE SET unchanged. Bit equality
on rho is not available and not the right ask -- deleting the residual's live
`recon` tensor changes how Triton lays out the same reduction, which moves the
last bit (and, measured against fp64, moves it toward the true value). What
matters is whether any row changes side of its threshold, so that is measured
rather than argued: both operand builds are scored by the same scan and the
hit masks compared.

Then what it buys: the sketch kernel reads the content twice (once to project,
once for the residual) and loads the [R, KV] basis in both loops; this one
reads it once and loads no basis. And then BA is swept, and the two kernels
are disassembled -- .claude/rules/disassemble-check-for-spills.md asks both
whether the fork spills and whether the cheaper arm kept the asynchronous
loads it was written for.
"""

import argparse
import os
import re
import subprocess

import torch
import triton


def _sass(cubin, path):
    open(path + ".cubin", "wb").write(cubin)
    r = subprocess.run(["nvdisasm", "-c", path + ".cubin"],
                       capture_output=True, text=True)
    if r.returncode:
        return None
    body = [re.sub(r"/\*[0-9a-f]{4}\*/|;\s*/\*.*?\*/", "", l).rstrip()
            for l in r.stdout.splitlines()]
    text = "\n".join(l for l in body if l.strip())
    open(path + ".sass", "w").write(text + "\n")
    return text


def _fire(rho, side, A, r):
    """Score the archive the way the branch scan does and return the hit mask.

    rho enters the certificate multiplicatively, so a last-bit change can only
    flip a row whose score sits within that bit of the threshold. Whether any
    row does is a property of the data, not of the argument, so it is measured.
    """
    from sglang.srt.layers.attention.vestigekv import defaults as D

    dev = rho.device
    g = torch.Generator(device=dev).manual_seed(0)
    H = 32
    q = torch.randn(D.SIDECAR_DIM, H, device=dev, dtype=torch.bfloat16, generator=g)
    qres = torch.rand(H, device=dev, generator=g)
    acc = (side.float() @ q.float()) * D.ATTN_SCALE
    score = acc + 0.2 * rho[:, None] * qres[None, :]
    # a threshold in the middle of the distribution, where flips are possible
    thr = score.median(0).values
    return (score > thr[None, :]).any(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=65536)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--out", default="/tmp/branch_operand")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from sglang.srt.layers.attention.vestigekv import defaults as D
    from sglang.srt.layers.attention.vestigekv import branch_operand as BO
    from sglang.srt.layers.attention.vestigekv.operand_fused import (
        _operand_fused_kernel,
        build_operands_fused,
    )

    dev = torch.device("cuda")
    torch.manual_seed(0)
    A, r = a.rows, a.rank
    pool = torch.randn(A + 1024, D.LATENT_DIM, device=dev, dtype=torch.bfloat16)
    slots = torch.randint(0, A, (A,), device=dev, dtype=torch.int64)
    Vz = torch.zeros(r, D.KV_LORA_RANK, device=dev, dtype=torch.float32)

    csk_a, rho_a, side_a = build_operands_fused(pool, slots, Vz)
    csk_b, rho_b, side_b = BO.build_operands_branch(pool, slots, Vz)
    torch.cuda.synchronize()
    dside = int((side_a != side_b).sum())
    dcsk = int((csk_a != csk_b).sum())
    rel = ((rho_a - rho_b).abs() / rho_a.abs().clamp_min(1e-9)).max().item()
    ulp = rel / torch.finfo(torch.float32).eps
    print(f"rows={A} rank={r}  side={dside} csk={dcsk} bitwise diffs; "
          f"rho max rel={rel:.3e} ({ulp:.2f} ulp)")
    if dside or dcsk:
        raise SystemExit("ABORT: side/csk must be bit-identical and are not")
    if ulp > 4.0:
        raise SystemExit(f"ABORT: rho moved {ulp:.1f} ulp, past a reduction-order "
                         "difference; the fork is computing something else")
    fa, fb = _fire(rho_a, side_a, A, r), _fire(rho_b, side_b, A, r)
    flips = int((fa != fb).sum())
    print(f"fire set: {int(fa.sum())}/{A} fired, {flips} rows change side of "
          f"the threshold ({100.0 * flips / A:.4f}%)")
    if flips:
        raise SystemExit(
            "ABORT: the operand fork moves the fire set, so it is a different "
            "recall and the branch arm's quality numbers do not carry over")

    def timeit(fn, iters):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        e = torch.cuda.Event(True), torch.cuda.Event(True)
        e[0].record()
        for _ in range(iters):
            fn()
        e[1].record()
        torch.cuda.synchronize()
        return e[0].elapsed_time(e[1]) * 1e3 / iters

    t_sk = timeit(lambda: build_operands_fused(pool, slots, Vz), a.iters)
    t_br = timeit(lambda: BO.build_operands_branch(pool, slots, Vz), a.iters)
    # content read twice + csk store vs content read once
    BY_SK = D.KV_LORA_RANK * 2 * 2 + D.SIDECAR_DIM * 2 + r * 2 + 4
    BY_BR = D.KV_LORA_RANK * 2 + D.SIDECAR_DIM * 2 + 4
    print(f"\n{'variant':>20} {'us':>9} {'GB/s':>9} {'B/row':>6}")
    for nm, t, by in (("sketch build", t_sk, BY_SK), ("branch fork", t_br, BY_BR)):
        print(f"{nm:>20} {t:>9.1f} {A*by/t/1e3:>9.1f} {by:>6}")
    print(f"{'speedup':>20} {t_sk/t_br:>9.2f}x   (traffic ceiling "
          f"{BY_SK/BY_BR:.2f}x)")

    if a.sweep:
        print(f"\nsweep BA (baseline {BO.BRANCH_BA} = {t_br:.1f} us)")
        keep = BO.BRANCH_BA
        best = (t_br, keep)
        for ba in (16, 32, 64, 128, 256):
            BO.BRANCH_BA = ba
            try:
                t = timeit(lambda: BO.build_operands_branch(pool, slots, Vz), a.iters)
            except Exception as e:
                print(f"  BA={ba:<4} {type(e).__name__}: {str(e)[:60]}")
                continue
            print(f"  BA={ba:<4} {t:>8.1f} us  {A*BY_BR/t/1e3:>8.1f} GB/s  "
                  f"{t_br/t:>5.2f}x")
            if t < best[0]:
                best = (t, ba)
        BO.BRANCH_BA = keep
        print(f"  best BA={best[1]} -> {best[0]:.1f} us")

    print()
    for nm, jit in (("sketch", _operand_fused_kernel),
                    ("branch", BO._operand_branch_kernel)):
        cache = jit.device_caches[torch.cuda.current_device()][0]
        for k in list(cache)[:1]:
            ck = cache[k]
            sass = _sass(ck.asm["cubin"], os.path.join(a.out, nm))
            n = {op: (sass.count(op) if sass else -1)
                 for op in ("LDL", "STL", "LDGSTS", "UBLKCP", "HMMA")}
            print(f"{nm:>8} regs={getattr(ck, 'n_regs', '?'):>4} "
                  f"spills={getattr(ck, 'n_spills', '?')} "
                  f"sass={0 if not sass else len(sass.splitlines()):>5} " +
                  " ".join(f"{k}={v}" for k, v in n.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
