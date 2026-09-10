"""End-to-end: put the COMPRESSED prefix cache back into the model and measure
real cross-entropy + needle retrieval. Per PREREG2.md.

Intervention point: [RMSNorm(c) | k_rot] inside KimiMLAAttention.forward, which
is exactly what an MLA deployment caches -- both K and V are derived from it.
"""
import argparse, json, math, os, torch, torch.nn.functional as F
from extract import ARCH, build_device_map, MLA_LAYERS_FIXED

STATE = {"op": None, "T": None, "rows": 0, "layers": 0, "keep": None,
         "gauge": None, "eastats": None, "pca": None, "tbasis": None}


def install_lm_head_slicer(model):
    """The checkpoint has no logits_to_keep; full logits are 2 bytes x L x 164k,
    which is the second memory wall for long context. Slice lm_head's input."""
    def pre(mod, args):
        k = STATE["keep"]
        return (args[0][:, -k:],) + args[1:] if k else None
    model.lm_head.register_forward_pre_hook(pre)


def lowpass(C, k):
    Fq = torch.fft.rfft(C, dim=0)
    if k < Fq.shape[0]:
        Fq[k:] = 0
    return torch.fft.irfft(Fq, n=C.shape[0], dim=0)


def make_op(name, m, frac=0.25):
    """-> f(C[T,576]) = (C_tilde, keep_mask|None). All query-independent."""
    def hybrid(C):
        s = max(1, int(round(m * frac))); k = max(0, (m - s) // 2)
        R = lowpass(C, k) if k > 0 else torch.zeros_like(C)
        idx = (C - R).norm(dim=-1).topk(min(s, C.shape[0])).indices
        R = R.clone(); R[idx] = C[idx]; return R, None
    def imp_only(C):
        k = max(0, m // 2)
        R = lowpass(C, k) if k > 0 else torch.zeros_like(C)
        idx = (C - R).norm(dim=-1).topk(min(m, C.shape[0])).indices
        keep = torch.zeros(C.shape[0], dtype=torch.bool, device=C.device)
        keep[idx] = True; return C, keep
    def _imp(C, score):
        keep = torch.zeros(C.shape[0], dtype=torch.bool, device=C.device)
        keep[score.topk(min(m, C.shape[0])).indices] = True
        return C, keep
    def imp_mean(C):     # crudest possible smooth model: the global mean
        return _imp(C, (C - C.mean(0, keepdim=True)).norm(dim=-1))
    def sel_gauge(C):
        """Re-gauged selector: ||A (C-mu)|| with A = (sum_h E[q q^T])^{1/2}.
        Equivalent to E_q[(q.delta)^2]; A is absorbable into W_UK/W_UV offline,
        so at inference this is a plain norm in the re-gauged basis."""
        A = STATE["gauge"]
        assert A is not None, "gauge not loaded"
        d = C - C.mean(0, keepdim=True)
        return _imp(C, (d @ A.to(C.dtype)).norm(dim=-1))
    def _ea(C, centered):
        """Expected Attention (arXiv:2510.00636). centered=False reproduces the
        published estimator; centered=True removes the 2 kbar^T Sigma delta
        cross term that the published form leaves in."""
        st = STATE["eastats"]; assert st is not None, "EA stats not loaded"
        mu, sig, Wv = st["mu"], st["sig"], st["Wv"]
        d_ = C.shape[1]
        X = (C - C.mean(0, keepdim=True)) if centered else C
        z = torch.zeros(C.shape[0], device=C.device)
        for h in range(mu.shape[0]):
            z = z + torch.softmax(X @ mu[h] / d_ ** .5
                                  + ((X @ sig[h]) * X).sum(-1) / (2 * d_), 0)
        vn = torch.einsum('hvd,td->th', Wv, C[:, :Wv.shape[-1]]).norm(dim=-1)
        return _imp(C, z * vn)
    def ea_orig(C):      return _ea(C, False)
    def ea_centered(C):  return _ea(C, True)

    def sel_gauge_lp(C):
        """Re-gauge AND a low-pass detector: the last place FFT could still
        earn its keep -- does bandwidth add anything on top of the gauge?"""
        A = STATE["gauge"]; assert A is not None
        d = C - lowpass(C, 16)
        return _imp(C, (d @ A.to(C.dtype)).norm(dim=-1))
    def _branch(C, dc, mr, mc, k=8):
        """Per-branch budget. The 64-d head-shared branch carries ~68% of the
        score variance and nearly all of the selection signal, yet is 1/9 of
        the cache width; the 512-d per-head content branch carries the rest.
        Spend the budget accordingly instead of treating all 576 dims alike."""
        out = C.clone()
        for lo, hi, m in ((0, dc, mc), (dc, C.shape[1], mr)):
            X = C[:, lo:hi]
            R = lowpass(X, k)
            idx = (X - R).norm(dim=-1).topk(min(m, X.shape[0])).indices
            Y = R.clone(); Y[idx] = X[idx]
            out[:, lo:hi] = Y
        return out, None
    def _lowrank(C, r, mc):
        """Every token keeps an r-dim code in a fixed offline basis, so EVERY
        record stays approximately recoverable -- unlike a low-pass fill,
        where ~T/k neighbouring tokens share one reconstruction and a
        low-attention token is not recoverable at all. On top of that, the
        m most anomalous tokens are stored exactly."""
        P = STATE["pca"]; assert P is not None, "pca basis not loaded"
        mu, V = P["mu"], P["V"][:, :r]
        d = C - mu
        rec = (d @ V) @ V.T + mu                       # per-token rank-r code
        idx = (C - rec).norm(dim=-1).topk(min(mc, C.shape[0])).indices
        out = rec.clone(); out[idx] = C[idx]
        return out, None
    def _lbasis(C, m, w, k=16):
        """Local temporal basis. span-rank found spans are strongly low-rank
        in time (r ~ w/3) but NOT rank-1, and no fixed kernel fits well, so the
        shapes are calibrated from data. At matched cost this buys 3x more
        spans than storing them exactly."""
        P = STATE["tbasis"]; assert P is not None, "temporal basis not loaded"
        phi = P[w]["phi"]                      # (w, r)
        r = phi.shape[1]; T_ = C.shape[0]; half = w // 2
        R = lowpass(C, k)
        d = C - R
        sc = d.norm(dim=-1).clone()
        n_anchor = max(1, m // r)              # r vectors per span, not w
        out = R.clone()
        for _ in range(n_anchor):
            i = int(sc.argmax())
            if sc[i] <= -1e29 or not (half <= i < T_ - half):
                sc[max(0, i - w):min(T_, i + w + 1)] = -1e30
                if sc.max() <= -1e29:
                    break
                continue
            B = d[i - half:i + half + 1]
            out[i - half:i + half + 1] = R[i - half:i + half + 1] + phi @ (phi.T @ B)
            sc[max(0, i - w):min(T_, i + w + 1)] = -1e30
        return out, None

    def _span(C, m, w, k=16):
        # identifiability (spikes-and-sines): the low-pass resolves T/k, a
        # width-w spike has spectral support below T/w, so the two bases stay
        # separable only while w*k << T. Past that the low-pass absorbs the
        # spike and the residual detector cannot see it.
        """Keep m/w anchors as SPANS of width w instead of m isolated tokens.
        Same total budget. Motivation: an anchor's neighbours are bound to it
        (fixed phrases, collocations), and a query usually matches the phrase,
        not the isolated answer token. Anchors are picked with non-maximum
        suppression so spans do not overlap. cf. ChunkKV (2502.00299), which
        keeps semantic chunks in standard KV caches."""
        T_ = C.shape[0]
        R = lowpass(C, k)
        sc = (C - R).norm(dim=-1).clone()
        half = w // 2
        n_anchor = max(1, m // w)
        keep = torch.zeros(T_, dtype=torch.bool, device=C.device)
        for _ in range(n_anchor):
            i = int(sc.argmax())
            if sc[i] <= -1e29:
                break
            lo, hi = max(0, i - half), min(T_, i + half + 1)
            keep[lo:hi] = True
            sc[max(0, i - w):min(T_, i + w + 1)] = -1e30      # NMS
        out = R.clone(); out[keep] = C[keep]
        return out, None
    def _invbranch(C, m, mode, r=64, k=16):
        """Inverted branch allocation (NoPE-only).

        Irminsul (2605.05696) gives the 512-d position-free latent the higher
        precision and compresses the 64-d branch harder, because after RoPE
        that branch scores at AUC 0.43 -- "rotation destroys position-free
        structure". Under NoPE it is not rotated, and we measure the inverse:
        the 64-d branch alone retains top-1 at 0.9869 while the 512-d content
        alone gives 0.0001. So invert the allocation.

        Every token keeps its digest (the 64-d branch, optionally PCA'd to r).
        Only the selected m tokens keep their 512-d content. What happens to a
        non-selected token is the open question:
          drop    - excluded from attention entirely (the digest is then unused)
          score   - digest scores it, its value falls back to the mean value
          fill    - content reconstructed from the low-pass (known to fail)
        """
        DCw = C.shape[1] - 64
        cont, dig = C[:, :DCw], C[:, DCw:]
        d = dig
        if r < dig.shape[1]:                            # narrow the digest to r dims
            X = dig.double(); mu = X.mean(0, keepdim=True); Xc = X - mu
            V = torch.linalg.svd(Xc, full_matrices=False)[2][:r]
            d = ((Xc @ V.T) @ V + mu).to(dig.dtype)
        sel = (d - lowpass(d, k)).norm(dim=-1)          # select ON THE DIGEST
        idx = sel.topk(min(m, C.shape[0])).indices
        keep = torch.zeros(C.shape[0], dtype=torch.bool, device=C.device)
        keep[idx] = True
        if mode == "drop":
            return C, keep
        out = C.clone()
        if mode == "score":
            out[~keep, :DCw] = cont.mean(0)             # value falls back
        elif mode == "fill":
            out[~keep, :DCw] = lowpass(cont, k)[~keep]
        return out, None

    def _kcenter(C, m2):
        """Gonzalez farthest-point: MAX-distortion covering. Outliers become
        their own reps (kept exactly); the dense bulk shares reps. Keeping T
        rows makes the multiplicity correction implicit and exact (NoPE+MLA
        exchangeability). Deterministic start: farthest from the mean."""
        T = C.shape[0]; m2 = min(m2, T)
        d = (C - C.mean(0, keepdim=True)).norm(dim=-1)
        idx = torch.empty(m2, dtype=torch.long, device=C.device)
        idx[0] = d.argmax()
        d = (C - C[idx[0]]).norm(dim=-1)
        for i in range(1, m2):
            idx[i] = d.argmax()
            d = torch.minimum(d, (C - C[idx[i]]).norm(dim=-1))
        reps = C[idx]
        a2 = torch.cdist(C, reps).argmin(1)
        return reps[a2], None

    def _kmeans(X, k, iters=10):
        N = X.shape[0]
        C0 = X[torch.randint(0, N, (1,), device=X.device)]
        for _ in range(k - 1):                      # k-means++ seeding
            d = torch.cdist(X, C0).min(1).values ** 2
            C0 = torch.cat([C0, X[torch.multinomial(d.clamp_min(1e-12), 1)]], 0)
        for _ in range(iters):
            a = torch.cdist(X, C0).argmin(1)
            oh = torch.zeros(N, k, device=X.device, dtype=X.dtype)
            oh[torch.arange(N, device=X.device), a] = 1.0
            cnt = oh.sum(0).clamp_min(1)
            C0 = (oh.T @ X) / cnt[:, None]
        return torch.cdist(X, C0).argmin(1), C0

    def _cluster(C, n):
        """Key-space clustering: replace every token by its cluster centroid.

        This is the operation RoPE forbids. Under NoPE the score of a centroid
        is the centroid of the scores for EVERY query, and because v = W_UV c
        is linear in c the centroid latent already carries the members' mean
        value -- no separate value merge is needed. Softmax over all T slots
        then reproduces weighted-centroid attention exactly, with each cluster
        carrying its multiplicity.

        NO TOKEN IS DISCARDED: every token is represented by its centroid.
        Cost: n*576 floats + T*log2(n) bits of assignment.
        """
        a, cent = _kmeans(C, min(n, C.shape[0]))
        return cent[a], None
    def nuke(C):
        """Keep 4 sink tokens and destroy everything else. If retrieval still
        survives, the layers we did NOT compress are carrying it, and the
        partial-layer condition is not a valid control."""
        keep = torch.zeros(C.shape[0], dtype=torch.bool, device=C.device)
        keep[:4] = True
        return C, keep
    def _sel_k(k):
        """Detector bandwidth k DECOUPLED from the storage budget m.
        k=1 is exactly mean-centering (bin 0 of the rFFT) and needs no FFT."""
        def f(C):
            R = C.mean(0, keepdim=True) if k <= 1 else lowpass(C, k)
            return _imp(C, (C - R).norm(dim=-1))
        return f
    def imp_norm(C):     # raw magnitude
        return _imp(C, C.norm(dim=-1))
    def imp_random(C):   # does selection do anything at all?
        return _imp(C, torch.rand(C.shape[0], device=C.device))
    def fft(C):  return lowpass(C, max(1, m // 2)), None
    def avgpool(C):
        T = C.shape[0]; r = max(1, T // m); n = (T // r) * r
        h = C[:n].view(-1, r, C.shape[1]).mean(1).repeat_interleave(r, 0)
        return torch.cat([h, C[n:]], 0), None
    def recent(C):
        keep = torch.zeros(C.shape[0], dtype=torch.bool, device=C.device)
        keep[-max(1, m - 4):] = True; keep[:4] = True; return C, keep
    def stride(C):
        T = C.shape[0]; r = max(1, T // m)
        keep = torch.zeros(T, dtype=torch.bool, device=C.device)
        keep[::r] = True; return C, keep
    table = dict(hybrid=hybrid, imp_only=imp_only, fft=fft, avgpool=avgpool,
                 recent=recent, stride=stride, imp_mean=imp_mean,
                 imp_norm=imp_norm, imp_random=imp_random,
                 sel_coupled=imp_only, sel_gauge=sel_gauge,
                 sel_gauge_lp=sel_gauge_lp, nuke=nuke, ea_orig=ea_orig,
                 ea_centered=ea_centered)
    for kk in (1, 4, 16, 64):
        table[f"sel_k{kk}"] = _sel_k(kk)
    # The pure linear-compression arm: keep EVERY token, compress each latent
    # to r dims with this context's own PCA basis. Storage matched to eviction
    # (r = 576*m/T) and the basis is not charged, both in the linear arm's
    # favour. Answers "is linear latent compression already optimal" directly
    # rather than by argument.
    def _proj(C):
        T, D = C.shape
        r = max(1, int(round(D * m / T)))
        X = C.double(); mu = X.mean(0, keepdim=True); Xc = X - mu
        V = torch.linalg.svd(Xc, full_matrices=False)[2][:r]
        return ((Xc @ V.T) @ V + mu).to(C.dtype), None
    table["proj_r"] = _proj
    # split_<rot rho denom>_<content rho denom>; m is the reference budget
    T_ref = None
    for n in (64, 128, 256, 512, 1024):
        def mkc(n=n):
            def f(C):
                return _cluster(C, n)
            return f
        table[f"cluster_n{n}"] = mkc()
    def kcenter(C):
        return _kcenter(C, m)
    def kmeans_m(C):
        a3, cent = _kmeans(C, min(m, C.shape[0]))
        return cent[a3], None
    table["kcenter"] = kcenter
    table["kmeans_m"] = kmeans_m
    def kcover(C):
        # half the budget exact (needle side, by construction), half covering
        # the remainder (denominator side, with implicit multiplicity)
        T2 = C.shape[0]; ke = max(1, m // 2)
        R = C.mean(0, keepdim=True) if 16 <= 1 else lowpass(C, 16)
        sel = (C - R).norm(dim=-1)
        top = sel.topk(min(ke, T2)).indices
        keep = torch.zeros(T2, dtype=torch.bool, device=C.device)
        keep[top] = True
        rest = (~keep).nonzero().flatten()
        out = C.clone()
        if rest.numel() > 0:
            out[rest] = _kcenter(C[rest], max(1, m - ke))[0]
        return out, None
    table["kcover"] = kcover
    def kcover_dig(C):
        # selection on the 64-d sidecar ONLY (the salience channel per the
        # static dissection; 11% read cost; rotated under RoPE -> must fail
        # there, which is the acceptance criterion)
        T2 = C.shape[0]; ke = max(1, m // 2)
        dig = C[:, C.shape[1] - 64:]
        sel = (dig - lowpass(dig, 16)).norm(dim=-1)
        top = sel.topk(min(ke, T2)).indices
        keep = torch.zeros(T2, dtype=torch.bool, device=C.device)
        keep[top] = True
        rest = (~keep).nonzero().flatten()
        out = C.clone()
        if rest.numel() > 0:
            out[rest] = _kcenter(C[rest], max(1, m - ke))[0]
        return out, None
    table["kcover_dig"] = kcover_dig
    def learned_sel(C):
        # PREREG9: trained static scorer (576->32->1 per layer), model untouched.
        # Scores computed on the raw cache row; eviction of everything below top-m.
        global _SELNETS
        try:
            _SELNETS
        except NameError:
            _SELNETS = torch.load("out/selector.pt", weights_only=False)
        li = STATE.get("cur_layer")
        ent = _SELNETS.get(li)
        if ent is None:
            raise SystemExit(f"ABORT learned_sel: no scorer for layer {li}; "
                             f"scoring with a missing net would silently rank garbage")
        net = torch.nn.Sequential(torch.nn.Linear(576, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
        net.load_state_dict(ent["state"]); net = net.to(C.device)
        with torch.no_grad():
            sc = net((C - ent["mu"].to(C.device)) / ent["sd"].to(C.device)).squeeze(-1)
        return _imp(C, sc)
    table["learned_sel"] = learned_sel
    def h2o(C):
        imp = STATE.get("imp_h2o"); assert imp is not None, "ABORT h2o: importance not computed"
        return _imp(C, imp.to(C.device))
    def snapkv(C):
        imp = STATE.get("imp_snap"); assert imp is not None, "ABORT snapkv: importance not computed"
        return _imp(C, imp.to(C.device))
    table["h2o"] = h2o
    table["snapkv"] = snapkv
    def h2o_recent(C):
        # the faithful H2O budget split: half heavy hitters, half recent window
        imp = STATE.get("imp_h2o"); assert imp is not None, "ABORT h2o_recent"
        T2 = C.shape[0]; keep = torch.zeros(T2, dtype=torch.bool, device=C.device)
        keep[-max(1, m // 2):] = True
        keep[imp.to(C.device).topk(min(m - m // 2, T2)).indices] = True
        return C, keep
    table["h2o_recent"] = h2o_recent
    def twotier(C):
        mk = STATE.get("mask2d"); assert mk is not None, "ABORT twotier: cascade not computed"
        return C, mk
    twotier.m2t = m
    table["twotier"] = twotier
    for rr in (4, 8, 16, 32, 64):
        def mkr(rr=rr):
            def f(C):
                return _invbranch(C, m, "drop", r=rr)
            return f
        table[f"dig_r{rr}"] = mkr()
    def digk64(C):
        # sidecar-only selection with detector bandwidth 64 -- the bandwidth
        # the long-context sweeps found optimal at L>=32768
        return _invbranch(C, m, "drop", r=64, k=64)
    table["digk64"] = digk64
    for md in ("drop", "score", "fill"):
        def mki(md=md):
            def f(C):
                return _invbranch(C, m, md)
            return f
        table[f"inv_{md}"] = mki()
    for w in (5, 9, 17, 33):
        def mkb(w=w):
            def f(C):
                return _lbasis(C, m, w)
            return f
        table[f"lbasis_w{w}"] = mkb()
    for w in (1, 3, 5, 9, 17):
        def mks(w=w):
            def f(C):
                return _span(C, m, w)
            return f
        table[f"span_w{w}"] = mks()
    # joint (w, k) grid: the failure boundary should track w*k ~ T
    for w in (1, 5, 17, 65):
        for kk in (4, 16, 64, 256):
            def mkj(w=w, kk=kk):
                def f(C):
                    return _span(C, m, w, kk)
                return f
            table[f"wk_w{w}_k{kk}"] = mkj()
    for r, cc in ((16, 64), (16, 32), (32, 64), (32, 32), (64, 64), (64, 128),
                  (128, 128)):
        def mkl(r=r, cc=cc):
            def f(C):
                return _lowrank(C, r, max(2, C.shape[0] // cc))
            return f
        table[f"lowrank_r{r}_c{cc}"] = mkl()
    for rr, cc in ((1, 16), (1, 32), (1, 64), (1, 128), (2, 16), (2, 32),
                   (4, 32), (4, 64), (8, 32), (16, 4)):
        def mk(rr=rr, cc=cc):
            def f(C):
                T_ = C.shape[0]; dc = C.shape[1] - 64
                return _branch(C, dc, max(2, T_ // rr), max(2, T_ // cc))
            return f
        table[f"split_r{rr}_c{cc}"] = mk()
    return table[name]


def patched_forward(self, hidden_states, attention_mask=None, past_key_values=None, **kw):
    b, s = hidden_states.shape[:-1]
    qs = self.q_proj(hidden_states).view(b, s, -1, self.q_head_dim).transpose(1, 2)
    q_pass, q_rot = torch.split(
        qs, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

    ckv = self.kv_a_proj_with_mqa(hidden_states)
    kp_raw, k_rot = torch.split(
        ckv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    c = self.kv_a_layernorm(kp_raw)

    op, T = STATE["op"], STATE["T"]
    STATE["cur_layer"] = self.layer_idx                     # learned_sel needs it
    STATE["gauge"] = getattr(self, "_gauge", None)
    STATE["eastats"] = getattr(self, "_eastats", None)
    STATE["pca"] = getattr(self, "_pca", None)
    STATE["tbasis"] = getattr(self, "_tbasis", None)
    keeps = None
    if op is not None:
        # op may be a single callable (applied to every batch element) or a
        # list of b callables -- the latter lets one forward evaluate b
        # different operators on the same sequence, which is what makes
        # batching pay off when experts are CPU-offloaded.
        ops = op if isinstance(op, (list, tuple)) else [op] * b
        assert len(ops) == b, f"{len(ops)} ops for batch {b}"
        C = torch.cat([c, k_rot], -1)                        # (b, s, 576)
        if STATE.get("need_twotier") and getattr(
                (op if not isinstance(op, (list, tuple)) else op[0]), "m2t", None) is not None:
            # PREREG19 Part A: self-calibrated cascade over the archive. All of
            # this is computable at compression time in deployment: labels from
            # the prefix's own queries, thresholds from those labels, sketch
            # basis from those queries. Nothing references the future.
            assert b == 1
            Tc = STATE["T"]
            m_ = getattr((op if not isinstance(op, (list, tuple)) else op[0]), "m2t")
            sc_ = getattr(self, "softmax_scale", None) or getattr(self, "scaling")
            Hh = self.num_heads; dn = self.qk_nope_head_dim
            Wb_ = self.kv_b_proj.weight.view(Hh, -1, self.kv_lora_rank)[:, :dn, :]
            qe_all = torch.cat([torch.einsum('hsd,hdc->hsc', q_pass[0].float(), Wb_.float()),
                                q_rot[0].float()], -1)               # (H, s, 576)
            Cf = torch.cat([c, k_rot], -1)[0, :Tc].float()
            rr_ = Cf[:, -64:]
            Fq = torch.fft.rfft(rr_, dim=0); Fq[16:] = 0
            sig = (rr_ - torch.fft.irfft(Fq, n=Tc, dim=0)).norm(dim=-1)
            keep0 = torch.zeros(Tc, dtype=torch.bool, device=Cf.device)
            keep0[sig.topk(min(m_, Tc)).indices] = True
            arch = (~keep0).nonzero().flatten()
            qe = qe_all.permute(1, 0, 2).reshape(-1, 576)            # (s*H, 576)
            skept = (qe @ Cf[keep0].T) * sc_
            p1 = torch.softmax(skept, -1)
            ent = -(p1 * p1.clamp_min(1e-12).log()).sum(-1)
            max1 = skept.max(-1).values
            del skept, p1
            qcal = qe_all[:, Tc // 2:Tc].reshape(-1, 576)[:, :512]
            qcal = qcal - qcal.mean(0, keepdim=True)
            V = torch.linalg.svd(qcal, full_matrices=False)[2][:64]
            csk = Cf[arch, :512] @ V.T
            rho_ = (Cf[arch, :512] - csk @ V).norm(dim=-1)
            qc_ = qe[:, :512]; qsk = qc_ @ V.T
            qres = (qc_ - qsk @ V).norm(dim=-1)
            idxs = (qe[:, 512:] @ Cf[arch, 512:].T + qsk @ csk.T) * sc_
            scale_ = (qres[:, None] * rho_[None, :]) * sc_ / (512 - 64) ** 0.5
            # calibration: prefix query rows [Tc//2, Tc), full-cache labels.
            # Causality: query at prefix position p sees keys <= p, all in the
            # prefix, so a causal mask over :Tc suffices.
            calrows = torch.arange((Tc // 2) * Hh, Tc * Hh, device=Cf.device)
            sfull = (qe[calrows] @ Cf.T) * sc_
            qpos_cal = (calrows // Hh)
            sfull = sfull.masked_fill(
                torch.arange(Tc, device=Cf.device)[None, :] > qpos_cal[:, None],
                torch.finfo(torch.float32).min)
            tgtc = sfull.argmax(-1); del sfull
            hardc = ~keep0[tgtc]
            entc = ent[calrows]
            thr_g = float(entc[hardc].quantile(0.03)) if int(hardc.sum()) > 5 else float("-inf")
            if float((entc > thr_g).float().mean()) > 0.60:
                thr_g = float("-inf")                                 # gate off: scan always
            posc = torch.searchsorted(arch.contiguous(), tgtc)
            zp = 8.0
            for z in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0):
                if int(hardc.sum()) == 0:
                    zp = 2.0; break
                fire_c = (idxs[calrows][hardc] + z * scale_[calrows][hardc])                     > max1[calrows][hardc][:, None]
                rec = float(fire_c.gather(1, posc[hardc][:, None]).float().mean())
                if rec >= 0.90:
                    zp = z; break
            gate = ent > thr_g
            score = idxs + zp * scale_
            fire = (score > max1[:, None]) & gate[:, None]
            topj = score.topk(min(16, score.shape[1]), dim=-1).indices
            fetch = torch.zeros_like(fire)
            fetch.scatter_(1, topj, True)
            fetch &= fire
            fetch_q = fetch.view(-1, Hh, arch.numel()).any(1)         # (s, |arch|)
            mask2d = keep0.unsqueeze(0).expand(fetch_q.shape[0], Tc).clone()
            rows_ = fetch_q.nonzero()
            mask2d[rows_[:, 0], arch[rows_[:, 1]]] = True
            STATE["mask2d"] = mask2d
            STATE["tt_stats"] = {"gate_off": thr_g == float("-inf"), "zp": zp,
                                 "gate_rate": float(gate.float().mean()),
                                 "fetch_per_q": float(fetch_q.sum(1).float().mean())}
            print(f"[tt] layer={self.layer_idx} gate_off={STATE['tt_stats']['gate_off']} "
                  f"zp={zp:.1f} gate_rate={STATE['tt_stats']['gate_rate']:.3f} "
                  f"fetch/q={STATE['tt_stats']['fetch_per_q']:.1f} "
                  f"mask_extra={float(mask2d.float().mean()) - float(keep0.float().mean()):.4f}",
                  flush=True)
            del idxs, scale_, fire, fetch, score, qe, ent, max1
        if STATE.get("need_imp"):
            # H2O / SnapKV baselines need prefill attention weights. Column sums
            # of the causal softmax, in query chunks; SnapKV uses only the last
            # WOBS queries, kernel-7 pooled along keys.
            assert b == 1, "importance baselines assume batch 1"
            WOBS = 32
            sc_ = getattr(self, "softmax_scale", None) or getattr(self, "scaling")
            Hh = self.num_heads; dn = self.qk_nope_head_dim
            Wb_ = self.kv_b_proj.weight.view(Hh, -1, self.kv_lora_rank)[:, :dn, :]
            # Only the compressed region [0, Tc): queries beyond the
            # compression point are the FUTURE and must not leak into the
            # baseline's importance (real H2O sees only the prefill so far).
            Tc = STATE["T"]
            qe = torch.cat([torch.einsum('hsd,hdc->hsc', q_pass[0, :, :Tc].float(),
                                         Wb_.float()), q_rot[0, :, :Tc].float()], -1)
            Cf = C[0, :Tc].float()
            acc = torch.zeros(Tc, device=C.device); accw = torch.zeros(Tc, device=C.device)
            neg = torch.finfo(torch.float32).min
            kidx = torch.arange(Tc, device=C.device)
            for i0 in range(0, Tc, 256):
                j0 = min(i0 + 256, Tc)
                w = torch.einsum('hqc,tc->hqt', qe[:, i0:j0], Cf) * sc_
                w = w.masked_fill(kidx[None, None, :] >
                                  torch.arange(i0, j0, device=C.device)[None, :, None], neg)
                w = torch.softmax(w, -1)
                acc += w.sum((0, 1))
                lo = max(i0, Tc - WOBS)
                if lo < j0:
                    accw += w[:, lo - i0:j0 - i0].sum((0, 1))
                del w
            total = float(acc.sum())
            assert abs(total - Hh * Tc) / (Hh * Tc) < 1e-3, f"imp colsums {total} != H*Tc"
            STATE["imp_h2o"] = acc
            STATE["imp_snap"] = torch.nn.functional.avg_pool1d(
                accw.view(1, 1, -1), 7, stride=1, padding=3).view(-1)
            del qe, Cf
        outs, keeps = [], []
        for bi in range(b):
            Cc, kp = ops[bi](C[bi, :T].float())
            assert Cc.shape == (T, C.shape[-1]), Cc.shape
            outs.append(torch.cat([Cc.to(C.dtype), C[bi, T:]], 0))
            keeps.append(kp)
            STATE["rows"] += T
        C = torch.stack(outs, 0)
        STATE["layers"] += 1
        c, k_rot = C[..., :self.kv_lora_rank], C[..., self.kv_lora_rank:]
        if all(k is None for k in keeps):
            keeps = None

    kp = self.kv_b_proj(c).view(
        b, s, -1, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
    k_pass, value_states = torch.split(
        kp, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    k_rot = k_rot.view(b, 1, s, self.qk_rope_head_dim).expand(*k_pass.shape[:-1], -1)

    query_states = torch.cat((q_pass, q_rot), dim=-1)
    key_states = torch.cat((k_pass, k_rot), dim=-1)


    # Tiled exact attention. Mathematically identical to the module's eager
    # kernel but never materialises the (H, L, L) score matrix, which is what
    # caps the context length. Validated against eager in gate 1.
    o = tiled_attention(query_states, key_states, value_states,
                        self.scaling, keeps, T)
    return self.o_proj(o.reshape(b, s, -1).contiguous())


QBLOCK_BYTES = 512 << 20     # target for one query block's score tile


def qblock_for(b, H, s, itemsize):
    """Pick the query-block size so one score tile stays near a fixed budget.
    A constant block size is itself a long-context wall: at L=131072 a
    512-row tile is 4.3 GB."""
    per = max(1, b * H * s * itemsize)
    return int(max(32, min(512, (QBLOCK_BYTES // per) // 32 * 32)))


def tiled_attention(q, k, v, scaling, keeps=None, T=None):
    """Tiled exact causal attention. The (L, L) mask is never materialised --
    at L=131072 that tensor alone is 68 GB, which is the real wall for long
    context, well before the weights are. Causality and eviction are applied
    per query block instead. Bit-identical to the module's eager kernel
    (validated by gate 1)."""
    b, H, s, _ = q.shape
    out = torch.empty(b, H, s, v.shape[-1], dtype=v.dtype, device=q.device)
    neg = torch.finfo(q.dtype).min
    kidx = torch.arange(s, device=q.device)
    qb = qblock_for(b, H, s, q.element_size())
    for i in range(0, s, qb):
        j = min(i + qb, s)
        w = torch.matmul(q[:, :, i:j], k.transpose(2, 3)) * scaling
        add = torch.where(kidx[None, :] > torch.arange(i, j, device=q.device)[:, None],
                          torch.tensor(neg, dtype=q.dtype, device=q.device),
                          torch.tensor(0.0, dtype=q.dtype, device=q.device))
        w = w + add
        if keeps is not None:
            for bi, kp in enumerate(keeps):
                if kp is not None:
                    if kp.dim() == 2:      # per-query mask (s, T): two-tier fetch
                        w[bi, :, :, :T] = w[bi, :, :, :T].masked_fill(
                            ~kp[i:j].unsqueeze(0), neg)
                    else:
                        w[bi, :, :, :T] = w[bi, :, :, :T].masked_fill(
                            ~kp.view(1, 1, -1), neg)
        w = torch.softmax(w, dim=-1, dtype=torch.float32).to(v.dtype)
        out[:, :, i:j] = torch.matmul(w, v)
        del w, add
    return out.transpose(1, 2).contiguous()


def ce_on_tail(model, ids, T):
    STATE["keep"] = ids.shape[1] - T + 1
    with torch.no_grad():
        lg = model(ids, use_cache=False).logits[0]
    STATE["keep"] = None
    assert lg.shape[0] == ids.shape[1] - T + 1, lg.shape
    return float(F.cross_entropy(lg[:-1].float(), ids[0, T:]))


def nll_of_answer_batched(model, ids, ans_start):
    """ids (b, L), same ans_start for every row -> list of b NLL sums."""
    STATE["keep"] = ids.shape[1] - ans_start + 1
    with torch.no_grad():
        lg = model(ids, use_cache=False).logits
    STATE["keep"] = None
    assert lg.shape[1] == ids.shape[1] - ans_start + 1, lg.shape
    out = []
    for bi in range(ids.shape[0]):
        out.append(float(F.cross_entropy(lg[bi, :-1].float(),
                                         ids[bi, ans_start:], reduction="sum")))
    return out


def nll_of_answer(model, ids, ans_start):
    STATE["keep"] = ids.shape[1] - ans_start + 1
    with torch.no_grad():
        lg = model(ids, use_cache=False).logits[0]
    STATE["keep"] = None
    assert lg.shape[0] == ids.shape[1] - ans_start + 1, lg.shape
    return float(F.cross_entropy(lg[:-1].float(), ids[0, ans_start:],
                                 reduction="sum"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=list(ARCH), required=True)
    ap.add_argument("--seq-len", type=int, required=True)
    ap.add_argument("--n-docs", type=int, required=True)
    ap.add_argument("--gpu-expert-layers", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--needle-trials", type=int, required=True)
    ap.add_argument("--ops", default="", help="comma list to override the op set")
    ap.add_argument("--gauge", default="", help="path to per-layer gauge matrices")
    ap.add_argument("--eastats", default="", help="per-layer query mu/sigma")
    ap.add_argument("--pca", default="", help="offline PCA basis per layer")
    ap.add_argument("--tbasis", default="", help="offline temporal span basis")
    ap.add_argument("--rhos", default="8,32,128",
                    help="comma list of rho DENOMINATORS; used to cost-match "
                         "uniform budgets against per-branch splits")
    ap.add_argument("--batch", type=int, default=1,
                    help="operators evaluated per forward; amortises the "
                         "CPU-offloaded expert weight streaming")
    a = ap.parse_args()

    import transformers.utils.generic as _g
    if not hasattr(_g, "OutputRecorder"):
        from transformers.utils.output_capturing import OutputRecorder as _OR
        _g.OutputRecorder = _OR
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    A = ARCH[a.arch]; MODEL = os.path.expanduser(A["path"])
    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    assert cfg.mla_use_nope is True, "PREREG2 targets the NoPE checkpoint"
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, dtype=torch.bfloat16,
        device_map=build_device_map(cfg.num_hidden_layers,
                                    set(range(cfg.first_k_dense_replace)),
                                    a.gpu_expert_layers),
        low_cpu_mem_usage=True)
    model.eval()
    install_lm_head_slicer(model)
    model.config._attn_implementation = "eager"
    for mm in model.modules():
        c = getattr(mm, "config", None)
        if c is not None and c is not model.config:
            c._attn_implementation = "eager"

    L, T = a.seq_len, a.seq_len // 2
    STATE["T"] = T
    GAUGE = torch.load(a.gauge, weights_only=True) if a.gauge else None
    EAST = torch.load(a.eastats, weights_only=True) if a.eastats else None
    PCA = torch.load(a.pca, weights_only=True) if a.pca else None
    TB = torch.load(a.tbasis, weights_only=True) if a.tbasis else None
    mla = [i for i in MLA_LAYERS_FIXED]
    origs = {}
    for i in mla:
        sa = model.model.layers[i].self_attn
        assert type(sa).__name__ == "KimiMLAAttention"
        origs[i] = sa.forward

    # ---------- corpus ----------
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    g = torch.Generator().manual_seed(a.seed)
    buf, docs = [], []
    for rec in ds:
        buf.extend(tok(rec["text"])["input_ids"])
        while len(buf) >= L and len(docs) < a.n_docs + a.needle_trials:
            docs.append(torch.tensor(buf[:L], dtype=torch.long)); buf = buf[L:]
        if len(docs) >= a.n_docs + a.needle_trials:
            break

    def run(ids, op, fn):
        STATE["op"] = op; STATE["rows"] = 0; STATE["layers"] = 0
        v = fn(ids)
        if op is not None and STATE["layers"] != len(mla):
            raise SystemExit(f"ABORT intervention hit {STATE['layers']} layers, "
                             f"expected {len(mla)}: it did not actually run")
        nb = 1 if not isinstance(op, (list, tuple)) else len(op)
        if op is not None and STATE["rows"] != len(mla) * STATE["T"] * nb:
            raise SystemExit(f"ABORT replaced {STATE['rows']} rows, expected "
                             f"{len(mla)*STATE['T']*nb}")
        return v

    # gate 1 runs at a short length so the UNPATCHED eager kernel (which
    # materialises the LxL score matrix) can still run; the tiled kernel is
    # then used at every length.
    gl = min(2048, L)
    ids0 = docs[0][:gl][None].cuda()
    STATE["T"] = gl // 2
    ce_unpatched = ce_on_tail(model, ids0, gl // 2)
    # The model allocates a full (b,1,L,L) causal mask before the layers.
    # Our tiled kernel builds causality per block, and KDA layers use a
    # separate 2-D linear_attn_mask, so this allocation is pure waste.
    import sys as _s
    _mk = _s.modules[type(model.model.layers[mla[0]].self_attn).__module__]
    _mk.create_causal_mask = lambda **kw: None
    print("[mem] model-level causal mask short-circuited", flush=True)

    for i in mla:                                   # install the patch
        sa = model.model.layers[i].self_attn
        if GAUGE is not None:
            assert i in GAUGE, f"no gauge for layer {i}"
            sa._gauge = GAUGE[i].cuda()
        if TB is not None:
            assert i in TB, f"no temporal basis for layer {i}"
            sa._tbasis = {w: {"phi": TB[i][w]["phi"].cuda()} for w in TB[i]}
        if PCA is not None:
            assert i in PCA, f"no pca for layer {i}"
            sa._pca = {"mu": PCA[i]["mu"].cuda(), "V": PCA[i]["V"].cuda()}
        if EAST is not None:
            assert i in EAST, f"no EA stats for layer {i}"
            sa._eastats = {"mu": EAST[i]["mu"].cuda(),
                           "sig": EAST[i]["sig"].cuda(),
                           "Wv": EAST[i]["Wv"].cuda()}
        sa.forward = patched_forward.__get__(sa, type(sa))
    ce_patched = run(ids0, None, lambda x: ce_on_tail(model, x, gl // 2))
    print(f"[gate1] patch fidelity: unpatched {ce_unpatched:.6f} vs patched "
          f"{ce_patched:.6f}  |d|={abs(ce_patched-ce_unpatched):.2e}", flush=True)
    if abs(ce_patched - ce_unpatched) > 1e-3:
        raise SystemExit("ABORT the patched forward changed the model")

    ce_id = run(ids0, lambda C: (C, None), lambda x: ce_on_tail(model, x, gl // 2))
    STATE["T"] = T
    print(f"[gate2] identity operator: dCE = {ce_id-ce_unpatched:+.2e}", flush=True)
    if abs(ce_id - ce_unpatched) > 1e-3:
        raise SystemExit("ABORT identity compression is not identity -- plumbing bug")

    RHOS = [1.0 / float(x) for x in a.rhos.split(",")]
    OPS = [("hybrid25", "hybrid", .25), ("hybrid50", "hybrid", .50),
           ("hybrid75", "hybrid", .75), ("imp_only", "imp_only", 0),
           ("fft", "fft", 0), ("recent", "recent", 0), ("stride", "stride", 0)]
    if a.ops:
        STATE["need_imp"] = any(o.split("@")[0] in ("h2o", "snapkv", "h2o_recent") for o in a.ops.split(","))
        STATE["need_twotier"] = any(o.split("@")[0] == "twotier" for o in a.ops.split(","))
        OPS = [(o, o.split("@")[0], float(o.split("@")[1]) if "@" in o else 0)
               for o in a.ops.split(",")]
    res = []
    STATE["T"] = T
    for di in range(a.n_docs):
        ids = docs[di][None].cuda()
        base = run(ids, None, lambda x: ce_on_tail(model, x, T))
        for rho in RHOS:
            m = max(2, int(round(T * rho)))
            for nm, base_op, fr in OPS:
                ce = run(ids, make_op(base_op, m, fr),
                         lambda x: ce_on_tail(model, x, T))
                res.append(dict(kind="ce", doc=di, rho=rho, m=m, op=nm,
                                base=base, val=ce, d=ce - base))
        print(f"[ce doc {di+1}/{a.n_docs}] base={base:.4f}  " + "  ".join(
            f"{r['op']}@1/{round(1/r['rho'])}:{r['d']:+.3f}"
            for r in res[-len(OPS):]), flush=True)

    eff = [r for r in res if r["rho"] == 1/128]
    if a.n_docs and eff and max(abs(r["d"]) for r in eff) < 1e-6:
        raise SystemExit("ABORT no operator changed CE at rho=1/128; the "
                         "intervention is not reaching the model")

    # ---------- needle ----------
    CITY = ["Reykjavik", "Montevideo", "Vientiane", "Gaborone", "Ljubljana"]
    for ti in range(a.needle_trials):
        filler = docs[a.n_docs + ti].tolist()
        city = CITY[ti % len(CITY)]
        code = 10000 + int(torch.randint(0, 89999, (1,), generator=g))
        nd = tok(f"\nThe secret passcode for {city} is {code}.\n")["input_ids"]
        q = tok(f"\nQuestion: what is the secret passcode for {city}?\nAnswer: {code}")["input_ids"]
        ans_n = len(tok(f"{code}")["input_ids"])
        p = int(torch.randint(int(T * .1), int(T * .8), (1,), generator=g))
        pre = filler[:p] + nd + filler[p:]
        pre = pre[:T]
        ids = torch.tensor(pre + q, dtype=torch.long)[None].cuda()
        ans_start = len(pre) + len(q) - ans_n
        STATE["T"] = len(pre)
        base = run(ids, None, lambda x: nll_of_answer(model, x, ans_start))
        jobs = [(rho, max(2, int(round(len(pre) * rho))), nm, bo, fr)
                for rho in RHOS for nm, bo, fr in OPS]

        # one-off gate: batching must not change a single result.
        # Only meaningful when batching is actually in use -- batch=1 vs
        # batch=2 differ at ~1e-3 CE even in the UNPATCHED model (kernel
        # selection), which is why batching was abandoned for metric runs.
        if ti == 0 and a.batch > 1:
            rho0, m0, nm0, bo0, fr0 = jobs[0]
            v1 = run(ids, make_op(bo0, m0, fr0),
                     lambda x: nll_of_answer(model, x, ans_start))
            idsB = ids.expand(2, -1).contiguous()
            vB = run(idsB, [make_op(bo0, m0, fr0)] * 2,
                     lambda x: nll_of_answer_batched(model, x, ans_start))
            dev = max(abs(x - v1) for x in vB)
            print(f"[gate3] batching equivalence on {nm0}: "
                  f"unbatched {v1:.6f} vs batched {vB} |d|={dev:.2e}", flush=True)
            if dev > 1e-3:
                raise SystemExit("ABORT batching changes the result")

        for i0 in range(0, len(jobs), a.batch):
            chunk = jobs[i0:i0 + a.batch]
            nb = len(chunk)
            idsB = ids.expand(nb, -1).contiguous() if nb > 1 else ids
            opsB = [make_op(bo, m, fr) for _, m, _, bo, fr in chunk]
            vals = run(idsB, opsB if nb > 1 else opsB[0],
                       lambda x: (nll_of_answer_batched(model, x, ans_start)
                                  if nb > 1 else
                                  [nll_of_answer(model, x, ans_start)]))
            for (rho, m, nm, _, _), v in zip(chunk, vals):
                res.append(dict(kind="needle", trial=ti, rho=rho, m=m, op=nm,
                                base=base, val=v, d=v - base))
        print(f"[needle {ti+1}/{a.needle_trials}] {city} base_nll={base:.3f}  " +
              "  ".join(f"{r['op']}:{r['d']:+.2f}" for r in res[-len(OPS):]), flush=True)
        STATE["T"] = T

    for i in mla:
        model.model.layers[i].self_attn.forward = origs[i]
    json.dump({"arch": a.arch, "L": L, "T": T, "n_docs": a.n_docs,
               "needle_trials": a.needle_trials, "ce_unpatched": ce_unpatched,
               "gate_identity_dCE": ce_id - ce_unpatched,
               "args": {k: v for k, v in sorted(vars(a).items())}, "rows": res},
              open(a.out, "w"))
    print(f"[done] {len(res)} rows -> {a.out}")


if __name__ == "__main__":
    main()
