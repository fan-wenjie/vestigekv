"""BranchRecallTier against the flag form it replaces, state and fire set.

    python mexp/kimi/branch_tier_equiv.py [--rows 65536] [--close 4096]

The flag form (SGLANG_DEBUG_VESTIGEKV_BRANCH_ONLY) reaches branch-only recall
by fitting a rank-r basis and overwriting it with zeros, so a plain RecallTier
handed a zero basis as v_init computes exactly what it computed. That is the
reference here, and BranchRecallTier has to match it -- not approximately, but
to the ulp budget the operand fork already established, WITH THE FIRE SET
UNCHANGED, because every branch-only quality number in this project was taken
with the flag form.

What is being verified is a deletion, so the test states what was deleted and
what must survive it:

  deleted   the eigendecomposition (v_init takes build's given-basis arm)
            the operand kernel's projection loop and its basis loads
            the sketch cache's STORAGE -- _csk_all is one zero row viewed as
            [closed, r], because a zero matrix only has to be stored once
            extend_closed's two matmuls per closed block

  survives  rho, the certificate's per-row norm, which is now ||content||
            _pos_all and arch, the row tables
            the [closed, r] SHAPE of _csk_all, which three readers index by
            the fire set

A shape that still reads right while the storage is gone is the whole point,
so the storage is measured and printed rather than asserted away.
"""

import argparse

import torch


def _tier(cls, rows, ncal, rank, dev, v_zero):
    from sglang.srt.layers.attention.vestigekv import defaults as D

    g = torch.Generator(device=dev).manual_seed(0)
    pool = torch.randn(rows + 4096, D.LATENT_DIM, device=dev,
                       dtype=torch.bfloat16, generator=g)
    row_slots = torch.arange(rows, device=dev, dtype=torch.int32)
    keep = torch.zeros(rows, dtype=torch.bool, device=dev)
    keep[torch.randperm(rows, device=dev, generator=g)[: rows // 32]] = True
    q_cal = torch.randn(ncal, 32, D.LATENT_DIM, device=dev, generator=g)
    q_pos = torch.arange(rows - ncal, rows, device=dev, dtype=torch.long)
    t = cls(r=rank)
    # The reference is handed the zero basis explicitly; the subclass supplies
    # its own, which is the deletion under test.
    t.build(pool, row_slots, keep, q_cal, q_pos, conservative=False,
            v_init=v_zero if v_zero is not None else None)
    return t, pool


def _fire(rho, side, dev):
    from sglang.srt.layers.attention.vestigekv import defaults as D

    g = torch.Generator(device=dev).manual_seed(1)
    H = 32
    q = torch.randn(D.SIDECAR_DIM, H, device=dev, dtype=torch.bfloat16, generator=g)
    qres = torch.rand(H, device=dev, generator=g)
    score = (side.float() @ q.float()) * D.ATTN_SCALE + 0.2 * rho[:, None] * qres[None, :]
    return (score > score.median(0).values[None, :]).any(1)


def _bytes(t):
    if t is None:
        return 0
    return t.untyped_storage().nbytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=65536)
    ap.add_argument("--close", type=int, default=4096)
    ap.add_argument("--ncal", type=int, default=64)
    ap.add_argument("--rank", type=int, default=64)
    a = ap.parse_args()

    from sglang.srt.layers.attention.vestigekv import defaults as D
    from sglang.srt.layers.attention.vestigekv.branch_tier import BranchRecallTier
    from sglang.srt.layers.attention.vestigekv.recall_tier import RecallTier

    dev = torch.device("cuda")
    vz = torch.zeros(a.rank, D.KV_LORA_RANK, device=dev, dtype=torch.float32)
    ref, pool = _tier(RecallTier, a.rows, a.ncal, a.rank, dev, vz)
    new, _ = _tier(BranchRecallTier, a.rows, a.ncal, a.rank, dev, None)

    eps = torch.finfo(torch.float32).eps
    fails = []

    def check(name, x, y, ulp_budget=0.0):
        if x is None or y is None:
            ok = x is y
            print(f"  {name:<22} {'both None' if ok else 'ONE IS None'}")
            if not ok:
                fails.append(name)
            return
        if x.shape != y.shape:
            print(f"  {name:<22} SHAPE {tuple(x.shape)} vs {tuple(y.shape)}")
            fails.append(name)
            return
        if not x.is_floating_point():
            n = int((x != y).sum())
            print(f"  {name:<22} {n} differing of {x.numel()}")
            if n:
                fails.append(name)
            return
        d = (x.float() - y.float()).abs()
        rel = (d / x.float().abs().clamp_min(1e-9)).max().item()
        print(f"  {name:<22} max rel {rel:.3e} ({rel / eps:.2f} ulp)")
        if rel / eps > ulp_budget:
            fails.append(f"{name} ({rel / eps:.2f} ulp > {ulp_budget})")

    print(f"after build ({a.rows} rows, rank {a.rank})")
    check("_rho_all", ref._rho_all, new._rho_all, ulp_budget=4.0)
    check("_pos_all", ref._pos_all, new._pos_all)
    check("arch", ref.arch, new.arch)
    check("_arch_idx", ref._arch_idx, new._arch_idx)
    check("V", ref.V, new.V, ulp_budget=0.0)
    check("csk (selection)", ref.csk, new.csk, ulp_budget=0.0)
    check("zp", torch.tensor([ref.zp]), torch.tensor([new.zp]), ulp_budget=4.0)

    fr, fn = _fire(ref.rho, ref.side, dev), _fire(new.rho, new.side, dev)
    flips = int((fr != fn).sum())
    print(f"  {'fire set':<22} {flips} rows change side "
          f"({int(fr.sum())}/{fr.numel()} fired)")
    if flips:
        fails.append("fire set")

    blk = pool[a.rows : a.rows + a.close]
    slots = torch.arange(a.rows, a.rows + a.close, device=dev, dtype=torch.int32)
    ref.extend_closed(blk, slots)
    new.extend_closed(blk, slots)
    print(f"\nafter extend_closed (+{a.close} rows)")
    check("_rho_all", ref._rho_all, new._rho_all, ulp_budget=4.0)
    check("_pos_all", ref._pos_all, new._pos_all)
    check("_csk_all shape/vals", ref._csk_all, new._csk_all, ulp_budget=0.0)

    br, bn = _bytes(ref._csk_all), _bytes(new._csk_all)
    print(f"\n_csk_all storage   flag form {br / 2**20:8.2f} MiB   "
          f"stripped {bn / 2**20:8.4f} MiB   "
          f"({br / max(bn, 1):.0f}x)")
    print(f"                   shape both {tuple(ref._csk_all.shape)}")

    if fails:
        raise SystemExit("ABORT: " + "; ".join(fails))
    print("\nequivalent: every surviving field matches, the fire set is unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
