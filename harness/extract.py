"""Extract MLA cache latents + queries from Kimi-Linear (NoPE MLA).

Captures per MLA layer:
  compressed_kv : (L, 576)  = [c_s (512, pre-norm) | k_rot_s (64, UNROTATED)]
  q             : (Q, 32, 192) = [q_pass (128) | q_rot (64)] at sampled query positions
Plus kv_b_proj / kv_a_layernorm weights, so scores are reconstructed offline.

Every capture is asserted. A hook that fires zero times is a hard abort.
"""
import argparse, json, math, os, sys, time
import torch

ARCH = {
    "kimi": dict(
        path="~/.cache/huggingface/hub/models--moonshotai--Kimi-Linear-48B-A3B-Base/"
             "snapshots/3b171c17bfc4ee348599b6781a2ca8715c21c8dc",
        attn_cls="KimiMLAAttention", pe="nope"),
    "kimi_instruct": dict(
        path="/workspace/hf_instruct/Kimi-Linear-48B-A3B-Instruct",
        attn_cls="KimiMLAAttention", pe="nope"),
    # Base weights with modeling code byte-identical to the Instruct arm's
    # (symlinked safetensors; the only variable across the two is weights).
    "kimi_base_mirror": dict(
        path="/workspace/base_mirror",
        attn_cls="KimiMLAAttention", pe="nope"),
    "dsv2": dict(
        path="~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite/"
             "snapshots/604d5664dddd88a0433dbae533b7fe9472482de0",
        attn_cls="DeepseekV2Attention", pe="rope"),
}
# same layer indices in both so the two models are compared at matched depth
MLA_LAYERS_FIXED = [3, 7, 11, 15, 19, 23, 26]


def check_hooks(got_kv, got_q, expected):
    """Predicate, extracted so it can be driven directly with known-bad input."""
    if got_kv != expected or got_q != expected:
        raise SystemExit(
            f"ABORT hooks fired kv={got_kv} q={got_q}, expected "
            f"{expected}/{expected}. A capture that captures nothing "
            f"yields conclusions about nothing.")


def build_device_map(n_layers, dense_layers, gpu_expert_layers):
    dm = {"model.embed_tokens": 0, "model.norm": 0, "lm_head": 0}
    moe_seen = 0
    for i in range(n_layers):
        dm[f"model.layers.{i}.input_layernorm"] = 0
        dm[f"model.layers.{i}.post_attention_layernorm"] = 0
        dm[f"model.layers.{i}.self_attn"] = 0
        if i in dense_layers:
            dm[f"model.layers.{i}.mlp"] = 0
        else:
            dm[f"model.layers.{i}.block_sparse_moe.gate"] = 0
            dm[f"model.layers.{i}.block_sparse_moe.shared_experts"] = 0
            dm[f"model.layers.{i}.block_sparse_moe.experts"] = (
                0 if moe_seen < gpu_expert_layers else "cpu")
            moe_seen += 1
    return dm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, required=True)
    ap.add_argument("--n-docs", type=int, required=True)
    ap.add_argument("--n-queries", type=int, required=True,
                    help="query positions sampled per doc, from the last half")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--arch", choices=list(ARCH), required=True)
    ap.add_argument("--gpu-expert-layers", type=int, required=True,
                    help="how many MoE layers keep their experts on GPU")
    ap.add_argument("--min-query-positions", type=int, default=512)
    ap.add_argument("--corpus-file", default="", help="local text file instead of fineweb")
    ap.add_argument("--max-lm-loss", type=float, required=True,
                    help="abort if mean CE on real text exceeds this; the "
                         "uniform-random baseline is ln(vocab)=12.0")
    a = ap.parse_args()

    # transformers 5.x moved OutputRecorder out of utils.generic; the checkpoint's
    # remote code targets 4.57. Re-expose the REAL object (not a stub) so the
    # dynamic module imports it unchanged.
    import transformers.utils.generic as _g
    if not hasattr(_g, "OutputRecorder"):
        from transformers.utils.output_capturing import OutputRecorder as _OR
        _g.OutputRecorder = _OR
        print("[shim] transformers.utils.generic.OutputRecorder <- "
              "transformers.utils.output_capturing", flush=True)
    assert hasattr(_g, "check_model_inputs"), "check_model_inputs also moved"

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    A = ARCH[a.arch]; MODEL = os.path.expanduser(A["path"])
    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    if a.arch.startswith("kimi"):
        mla_idx = sorted(i - 1 for i in cfg.linear_attn_config["full_attn_layers"])
        assert cfg.mla_use_nope is True, "checkpoint is not NoPE -- wrong subject"
    else:
        mla_idx = list(MLA_LAYERS_FIXED)     # dsv2: every layer is MLA; subsample
        assert not getattr(cfg, "mla_use_nope", False)
    assert mla_idx == MLA_LAYERS_FIXED, f"depth mismatch: {mla_idx}"
    n_layers = cfg.num_hidden_layers
    dense = set(range(cfg.first_k_dense_replace))
    print(f"[cfg] layers={n_layers} MLA(0-based)={mla_idx} "
          f"kv_lora={cfg.kv_lora_rank} rope_dim={cfg.qk_rope_head_dim} "
          f"nope_dim={cfg.qk_nope_head_dim} heads={cfg.num_attention_heads} "
          f"pe={A['pe']}", flush=True)
    assert len(mla_idx) == 7, f"expected 7 MLA layers, got {len(mla_idx)}"

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    L_MAX = a.seq_len

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, dtype=torch.bfloat16,
        device_map={"": 0},                       # 30GB, fits whole on GPU
        attn_implementation="eager",
        low_cpu_mem_usage=True) if a.arch == "dsv2" else \
        AutoModelForCausalLM.from_pretrained(
            MODEL, trust_remote_code=True, dtype=torch.bfloat16,
            device_map=build_device_map(n_layers, dense, a.gpu_expert_layers),
            low_cpu_mem_usage=True)
    model.eval()
    print(f"[load] {time.time()-t0:.0f}s  gpu_alloc="
          f"{torch.cuda.memory_allocated()/2**30:.2f}GiB", flush=True)

    # ---- verify the modules we are about to hook are the real MLA ones ----
    layers = model.model.layers
    for i in mla_idx:
        sa = layers[i].self_attn
        assert type(sa).__name__ == A["attn_cls"], \
            f"layer {i} is {type(sa).__name__}, not {A['attn_cls']}"
        assert sa.kv_lora_rank == cfg.kv_lora_rank
        if a.arch.startswith("kimi"):
            assert sa.use_nope is True
    if a.arch.startswith("kimi"):
        for i in range(n_layers):
            if i not in mla_idx:
                assert type(layers[i].self_attn).__name__ == "KimiDeltaAttention", \
                    f"layer {i} unexpectedly {type(layers[i].self_attn).__name__}"
    print("[check] MLA/KDA layer identity verified", flush=True)


    # ---- hooks ----
    grab, fire = {}, {k: 0 for k in ("kv", "q", "o")}
    cur_qpos = {"v": None}

    def mk(kind, li):
        def h(mod, inp, out):
            fire[kind] += 1
            grab[(kind, li)] = out.detach().to(torch.float32).cpu()
        return h

    def mk_o(li):
        def h(mod, args):
            fire["o"] += 1
            x = args[0].detach()
            if cur_qpos["v"] is not None:
                x = x[:, cur_qpos["v"]]
            grab[("o", li)] = x.to(torch.float32).cpu()
        return h

    handles = []
    for i in mla_idx:
        sa = layers[i].self_attn
        handles.append(sa.kv_a_proj_with_mqa.register_forward_hook(mk("kv", i)))
        handles.append(sa.q_proj.register_forward_hook(mk("q", i)))
        handles.append(sa.o_proj.register_forward_pre_hook(mk_o(i)))

    # The checkpoint's __init__ hard-forces flash_attention_2 (line 839-846),
    # ignoring what we pass. flash_attn is not installed. _use_flash_attention_2
    # is assigned but never read, and _attn_implementation is read only inside
    # KimiMLAAttention.forward -- so overriding it post-construction is complete.
    def set_attn(kind):
        type(model)._supports_sdpa = True
        model.config._attn_implementation = kind
        for m in model.modules():
            c = getattr(m, "config", None)
            if c is not None and c is not model.config:
                c._attn_implementation = kind

    # MEASURED 2026-09-01: eager and sdpa disagree by 1.4e-1 relative on this
    # checkpoint (v_head_dim=128 != qk_head_dim=192 is handled differently by
    # the sdpa wrapper). eager is the module's own reference implementation and
    # is validated below against an independent offline reconstruction.
    if a.arch.startswith("kimi"):
        set_attn("eager")
    print(f"[attn] backend = {model.config._attn_implementation}", flush=True)

    # ---- corpus: local file (PREREG8 dedup regimes) or fineweb-edu ----
    if a.corpus_file:
        ds = [{"text": open(a.corpus_file).read()}]
    else:
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                          split="train", streaming=True)
    g = torch.Generator().manual_seed(a.seed)

    L, docs_done = a.seq_len, 0
    os.makedirs(a.out, exist_ok=True)
    meta = {"arch": a.arch, "pe": A["pe"],
            "seq_len": L, "n_queries": a.n_queries, "seed": a.seed,
            "mla_layers": mla_idx, "model": MODEL,
            "config": {k: getattr(cfg, k) for k in
                       ("kv_lora_rank", "qk_rope_head_dim", "qk_nope_head_dim",
                        "v_head_dim", "num_attention_heads")},
            "rope_theta": getattr(cfg, "rope_theta", None),
            "args": {k: v for k, v in sorted(vars(a).items())}}

    # static weights needed to rebuild scores offline
    # Take cos/sin and the softmax scale FROM THE MODEL rather than
    # reimplementing them. DeepSeek-V2-Lite uses YaRN rope_scaling (factor 40,
    # mscale 0.707), so neither plain theta^(-2i/d) nor d^(-1/2) is correct.
    W = {}
    for i in mla_idx:
        sa = layers[i].self_attn
        W[f"kv_b_proj.{i}"] = sa.kv_b_proj.weight.detach().float().cpu()
        W[f"kv_a_ln.{i}"] = sa.kv_a_layernorm.weight.detach().float().cpu()
        sc = getattr(sa, "softmax_scale", None)
        if sc is None:
            sc = getattr(sa, "scaling")
        W[f"softmax_scale.{i}"] = float(sc)
    rope = {}
    for i in mla_idx:
        re_ = getattr(layers[i].self_attn, "rotary_emb", None)
        if re_ is not None:
            dummy = torch.zeros(1, 1, L_MAX, 1, device="cuda",
                                dtype=torch.float32)
            co, si = re_(dummy, seq_len=L_MAX)
            rope[f"cos.{i}"] = co.detach().float().cpu()[:L_MAX]
            rope[f"sin.{i}"] = si.detach().float().cpu()[:L_MAX]
            break
    print(f"[weights] softmax_scale={W[f'softmax_scale.{mla_idx[0]}']:.6f}  "
          f"rotary_emb={'captured' if rope else 'absent (NoPE)'}", flush=True)
    torch.save({"W": W, "eps": cfg.rms_norm_eps, "rope": rope},
               f"{a.out}/weights.pt")

    buf, ntok_seen, losses = [], 0, []
    for rec in ds:
        ids = tok(rec["text"], return_tensors=None)["input_ids"]
        buf.extend(ids)
        if len(buf) < L:
            continue
        chunk = torch.tensor(buf[:L], dtype=torch.long)[None].cuda()
        buf = buf[L:]

        qpos = (torch.randperm(L - L // 2, generator=g)[:a.n_queries]
                + L // 2).sort().values
        cur_qpos["v"] = qpos
        grab.clear()
        before = dict(fire)
        with torch.no_grad():
            o = model(chunk, use_cache=False, labels=chunk)
        check_hooks(fire["kv"] - before["kv"], fire["q"] - before["q"], 7)
        assert fire["o"] - before["o"] == 7, \
            f"ABORT o_proj hook fired {fire['o']-before['o']}/7"

        # A broken load (wrong kernel semantics, unmatched weights) still emits
        # tensors of the right shape. Only the loss can tell us it is broken.
        loss = float(o.loss)
        import math
        rand = math.log(cfg.vocab_size)
        if not (0.05 < loss < a.max_lm_loss):
            raise SystemExit(
                f"ABORT LM loss {loss:.3f} outside (0.05, {a.max_lm_loss}); "
                f"uniform-random baseline is {rand:.2f}. The model is not "
                f"computing what we think it is; captures would be garbage.")
        losses.append(loss)

        out = {"qpos": qpos}
        for i in mla_idx:
            kv = grab[("kv", i)][0]                       # (L, 576)
            q = grab[("q", i)][0]                         # (L, 32*192)
            assert kv.shape == (L, cfg.kv_lora_rank + cfg.qk_rope_head_dim), kv.shape
            assert q.shape == (L, cfg.num_attention_heads *
                               (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim)), q.shape
            assert torch.isfinite(kv).all() and torch.isfinite(q).all(), \
                f"non-finite capture at layer {i}"
            out[f"kv.{i}"] = kv.to(torch.float16)
            out[f"q.{i}"] = q[qpos].view(
                len(qpos), cfg.num_attention_heads, -1).to(torch.float16)
            o = grab[("o", i)][0]                         # (Q, 32*128)
            assert o.shape == (len(qpos), cfg.num_attention_heads *
                               cfg.v_head_dim), o.shape
            out[f"o.{i}"] = o.to(torch.float16)
        torch.save(out, f"{a.out}/doc{docs_done:03d}.pt")
        docs_done += 1
        ntok_seen += L
        print(f"[doc {docs_done}/{a.n_docs}] loss={loss:.3f} {time.time()-t0:.0f}s "
              f"gpu={torch.cuda.memory_allocated()/2**30:.2f}GiB", flush=True)
        if docs_done >= a.n_docs:
            break

    for h in handles:
        h.remove()
    assert docs_done == a.n_docs, f"ABORT only {docs_done}/{a.n_docs} docs"
    meta["n_docs"] = docs_done
    meta["total_query_positions"] = docs_done * a.n_queries
    meta["lm_loss_mean"] = sum(losses) / len(losses)
    meta["lm_loss_all"] = losses
    meta["lm_loss_random_baseline"] = math.log(cfg.vocab_size)
    assert meta["total_query_positions"] >= a.min_query_positions, \
        f"ABORT only {meta['total_query_positions']} query positions"
    json.dump(meta, open(f"{a.out}/meta.json", "w"), indent=2)
    print(f"[done] {docs_done} docs -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
