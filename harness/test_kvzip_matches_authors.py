#!/usr/bin/env python3
"""Does our KVzip scoring compute the authors' statistic? Ask their code.

    KVZIP_SRC=/path/to/KVzip python harness/test_kvzip_matches_authors.py

The KVzip comparison in the paper runs our transcription of the authors'
scoring rule, because their implementation monkeypatches LlamaAttention /
Qwen2Attention / Qwen3Attention / Gemma3Attention and has no MLA path at all --
it cannot run on this checkpoint, and porting it would mean writing the MLA
attention ourselves, which is the part a reader would want independently
checked. So the check has to happen one level down: on the same tensors, does
our function return what their function returns?

Their `KVScore._get_score` (attention/score.py) does, per layer:

    keys    = cat([sinks, chunk, last q_len keys])
    logits  = q @ keys^T / sqrt(head_dim)
    logits += causal mask over the trailing q_len x q_len block
    w       = softmax(logits, -1)[..., sink : sink+ctx_len]
    score   = w.amax over (group, query)        -> one score per KV head

Ours reduces the head axis too, because an MLA cache row is a single latent
shared by every head -- per-head eviction does not exist in that architecture.
So the comparison is their score reduced over heads against ours, and to make
the key sets identical the test uses sink=0 with one chunk spanning the whole
context, which is exactly the configuration our scoring pass runs in.

Exits non-zero on any mismatch. CPU only; no model, no GPU, no download.
"""
from __future__ import annotations

import math
import os
import sys

import torch


def load_authors_scorer(src):
    """Instantiate the authors' KVScore with the attributes _get_score reads."""
    sys.path.insert(0, src)
    from attention.score import KVScore  # noqa: E402

    return KVScore


def theirs(KVScore, q, k, sink, start, end, n_layers=1):
    """Run the authors' _get_score verbatim and return its score tensor."""
    s = KVScore()
    s.n_heads_kv = k.shape[1]
    s.n_layers = n_layers
    s.dtype = q.dtype
    s.device = q.device
    s.sink, s.start_idx, s.end_idx = sink, start, end
    s.init_score()
    s._get_score(q, k, 0)
    return s.score[0]  # [1, n_heads_kv, ctx_len]


def ours(kvzip_position_scores, q, k, n_ctx, q0, head_dim):
    """Run our extracted function on the same tensors.

    Their _get_score scales by 1/sqrt(head_dim) inside; ours takes the scale as
    an argument, so it is passed here rather than folded in."""
    # ours wants [H, nq, D] queries and [K, D] keys, one head group flattened
    qh = q[0].reshape(-1, q.shape[2], q.shape[3])          # [H, q, D]
    kk = k[0, 0]                                            # [K, D], MQA-shared
    return kvzip_position_scores(qh, kk, head_dim**-0.5, n_ctx, q0)


def main():
    src = os.environ.get("KVZIP_SRC")
    if not src or not os.path.isdir(src):
        print("set KVZIP_SRC to a checkout of https://github.com/snu-mllab/KVzip",
              file=sys.stderr)
        return 2
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from e2e import kvzip_position_scores

    KVScore = load_authors_scorer(src)
    torch.manual_seed(0)

    bad = 0
    # One KV head keeps the two reductions comparable: with several, theirs
    # returns one row per head and ours returns the max over heads, so the
    # test would be comparing a max against its members. Head-count variation
    # is covered by the group axis, which theirs reduces and ours does too.
    for n_ctx, q_len, n_heads, head_dim in [(16, 4, 2, 8), (64, 16, 4, 16),
                                            (128, 32, 8, 32)]:
        q = torch.randn(1, n_heads, q_len, head_dim, dtype=torch.float32)
        k = torch.randn(1, 1, n_ctx + q_len, head_dim, dtype=torch.float32)

        t = theirs(KVScore, q, k, sink=0, start=0, end=n_ctx)[0]   # [1, n_ctx]
        t = t.amax(dim=0)                                          # reduce head
        o = ours(kvzip_position_scores, q, k, n_ctx, q0=n_ctx, head_dim=head_dim)

        d = (t - o).abs().max().item()
        ok = d < 1e-5
        bad += not ok
        print(f"  ctx={n_ctx:>4} q={q_len:>3} heads={n_heads:>2} dim={head_dim:>3}"
              f"   max|theirs-ours| = {d:.3e}   {'OK' if ok else 'MISMATCH'}")

    print("\nagree" if not bad else f"\n{bad} case(s) disagree")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
