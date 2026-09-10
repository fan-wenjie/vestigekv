"""Static anatomy of MLA score/value pathways. Weights only, no forward pass.

Per layer:
  score-path latent spectrum: eig of  sum_h G_h^T G_h,  G_h = W_q_nope^{hT} W_UK^h
  value-path latent spectrum: eig of  sum_h W_UV^{hT} W_UV^h
  rot channel: row norms of the kv_a rot block vs content block (is the sidecar
    up-weighted?), ER of each block, and cross-head overlap of W_q_rot row spaces.
"""
import json, sys, glob, torch
from safetensors import safe_open

MODEL_DIR, OUT = sys.argv[1], sys.argv[2]
cfg = json.load(open(glob.glob(MODEL_DIR + "/config.json")[0]))
H = cfg["num_attention_heads"]; DN = cfg["qk_nope_head_dim"]; DR = cfg["qk_rope_head_dim"]
DC = cfg["kv_lora_rank"]; DV = cfg["v_head_dim"]
LAYERS = [3, 7, 11, 15, 19, 23, 26]
idx = json.load(open(glob.glob(MODEL_DIR + "/model.safetensors.index.json")[0]))["weight_map"]

def get(name):
    with safe_open(MODEL_DIR + "/" + idx[name], framework="pt") as f:
        return f.get_tensor(name).float().cuda()

def er(ev):
    ev = ev.clamp_min(0)
    return float(ev.sum() ** 2 / (ev ** 2).sum())

def r90(ev):
    ev = ev.clamp_min(0).flip(0)  # eigvalsh ascending -> descending
    c = ev.cumsum(0) / ev.sum()
    return int((c < 0.90).sum()) + 1

res = []
print(f"{'layer':>5} | {'score ER':>9} {'score r90':>10} | {'value ER':>9} {'value r90':>10} | "
      f"{'rot/cont rownorm':>16} {'qrot overlap':>13}")
for li in LAYERS:
    p = f"model.layers.{li}.self_attn."
    Wq = get(p + "q_proj.weight").view(H, DN + DR, -1)          # (H,192,dm)
    Wb = get(p + "kv_b_proj.weight").view(H, DN + DV, DC)       # (H,256,512)
    Wa = get(p + "kv_a_proj_with_mqa.weight")                   # (576,dm)
    Gs = torch.zeros(DC, DC, device="cuda"); Gv = torch.zeros(DC, DC, device="cuda")
    subs = []
    for h in range(H):
        G = Wq[h, :DN, :].T @ Wb[h, :DN, :]                     # (dm,512)
        Gs += G.T @ G
        V = Wb[h, DN:, :]                                       # (128,512)
        Gv += V.T @ V
        qr = Wq[h, DN:, :]                                      # (64,dm) rot query
        subs.append(torch.linalg.svd(qr, full_matrices=False)[2][:16])
    es = torch.linalg.eigvalsh(Gs); ev = torch.linalg.eigvalsh(Gv)
    rn_rot = Wa[DC:].norm(dim=-1).mean(); rn_cont = Wa[:DC].norm(dim=-1).mean()
    S = torch.stack(subs)
    ov = []
    for i in range(H):
        for j in range(i + 1, H):
            ov.append(float((S[i] @ S[j].T).pow(2).sum() / 16))
    ovm = sum(ov) / len(ov)
    row = dict(layer=li, score_er=er(es), score_r90=r90(es), value_er=er(ev),
               value_r90=r90(ev), rot_over_content=float(rn_rot / rn_cont), qrot_overlap=ovm)
    res.append(row)
    print(f"{li:>5} | {row['score_er']:>9.1f} {row['score_r90']:>10} | {row['value_er']:>9.1f} "
          f"{row['value_r90']:>10} | {row['rot_over_content']:>16.3f} {row['qrot_overlap']:>13.4f}")
    del Wq, Wb, Wa, Gs, Gv; torch.cuda.empty_cache()
json.dump(res, open(OUT, "w"), indent=1)
