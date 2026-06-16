"""
═══════════════════════════════════════════════════════════════════════════════
 RESULTADO NEGATIVO — REPROVADO em 2026-06-14. NÃO reintroduzir em produção.
   rembg/u2net puro venceu (foto branco-no-branco IMG_20260610, 12 obj):
     vazamento  rembg 0.05  vs  trimap 0.25  (5× pior)
     resíduo    rembg 0.09  vs  trimap 0.33  (4× pior)
   O BG-seed cromático COME o branco do objeto (paradoxo same-color) e vaza papel.
   Mantido como artefato reproduzível. Ver memória [[bg-vector-rejected]].
═══════════════════════════════════════════════════════════════════════════════

Protótipo: trimap matting (FG = heatmap DINOv2 · BG = vetor de fundo 3-vias)
vs. rembg/u2net puro — para decidir se a separação objeto/fundo melhora.

Ideia (acordada com o usuário):
  - FG seed   ← heatmap DINOv2 (object-ness, sinal SEMÂNTICO já existente)
  - BG seed   ← estimador de fundo robusto em 3 vias, CALCULADO em código
                (mediana Lab · cluster dominante k-means · anel de borda),
                sobre os pixels FORA de todas as bboxes (= fundo conhecido).
  - unknown   ← o resto → alpha-matting closed-form (pymatting) resolve a borda.
  - gate      ← separabilidade dist(obj,fundo)/spread decide se a via vale.

Não toca produção. Roda em poc e mede:
  - vazamento de fundo no anel da bbox (quanto fundo sobreviveu) — menor é melhor
  - cobertura no centroide (quanto do objeto sobrou)            — maior é melhor
  - resíduo sobre pixels "cara de fundo" (bgdist<=tau)          — menor é melhor
  - banda suave (transição de borda) e runtime

Uso:
  conda run -n casaiq python poc_dinov2/demo_trimap.py [foto] [cache_s1.json]
"""
import sys
import time
import json
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from core.gazetteer import GazetteerMatcher, _COLLECTIVE_SZ, _PATCH  # noqa: E402

FOTO  = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/cuco/Downloads/IMG_20260610_112623028.jpg")
CACHE = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "poc_dinov2/output/piloto_analise_cache.json"
OUT   = ROOT / "poc_dinov2/output/demo_trimap"
OUT.mkdir(parents=True, exist_ok=True)

MATTE_MAX = 384      # downscale do ROI p/ o solver closed-form (velocidade)
PAD       = 0.10     # padding do ROI (igual produção _gerar_crop_final)

# ── DINOv2 (replica _encode_* do gazetteer, sem precisar do índice FAISS) ──────
print("[init] carregando DINOv2-small...")
_proc  = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
_model = AutoModel.from_pretrained("facebook/dinov2-small").eval()
_DIM   = _model.config.hidden_size


def encode_global(img):
    img_r = img.resize((448, 448), Image.LANCZOS)
    with torch.no_grad():
        inp = _proc(images=img_r, return_tensors="pt")
        cls = F.normalize(_model(**inp).last_hidden_state[:, 0, :], dim=-1)
    return cls[0].numpy().astype("float32")


def encode_patches(img):
    tile, ntile = 224, _COLLECTIVE_SZ // 224
    img_r = img.resize((_COLLECTIVE_SZ, _COLLECTIVE_SZ), Image.LANCZOS)
    tiles = [img_r.crop((c*tile, r*tile, (c+1)*tile, (r+1)*tile))
             for r in range(ntile) for c in range(ntile)]
    with torch.no_grad():
        feats = _model(**_proc(images=tiles, return_tensors="pt")).last_hidden_state[:, 1:, :]
        feats = F.normalize(feats, dim=-1)
    ppt = int(feats.shape[1] ** 0.5)
    grid = feats.reshape(ntile, ntile, ppt, ppt, _DIM)
    grid = grid.permute(0, 2, 1, 3, 4).reshape(ntile*ppt, ntile*ppt, _DIM)
    return grid.numpy().astype("float32")   # (64,64,384)


# ── Vetor de fundo 3-vias (em código, sobre pixels fora das bboxes) ────────────
def _kmeans_dominante(X, k=3, iters=12, seed=0):
    """k-means numpy puro; devolve centro do maior cluster."""
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), size=min(k, len(X)), replace=False)].copy()
    lab = np.zeros(len(X), int)
    for _ in range(iters):
        d = ((X[:, None, :] - C[None]) ** 2).sum(-1)
        lab = d.argmin(1)
        for j in range(len(C)):
            if (lab == j).any():
                C[j] = X[lab == j].mean(0)
    counts = np.bincount(lab, minlength=len(C))
    return C[int(counts.argmax())]


def modelo_fundo_3vias(lab_img, mask_obj):
    """Retorna (estimativas[3x3], mapa_bgdist HxW, tau)."""
    H, W = lab_img.shape[:2]
    bg = lab_img[~mask_obj]                          # pixels de fundo conhecido
    # subamostra p/ k-means
    sub = bg[np.random.default_rng(0).choice(len(bg), size=min(6000, len(bg)), replace=False)]
    est_med     = np.median(bg, axis=0)              # 1) mediana (robusta a outliers)
    est_cluster = _kmeans_dominante(sub, k=3)        # 2) cluster dominante (fundo bimodal)
    ring = np.zeros((H, W), bool)                    # 3) anel de borda
    m = int(0.06 * min(H, W))
    ring[:m, :] = ring[-m:, :] = ring[:, :m] = ring[:, -m:] = True
    est_border  = np.median(lab_img[ring & ~mask_obj], axis=0)
    ests = np.stack([est_med, est_cluster, est_border])           # (3,3)

    # mapa: distância ao fundo = min sobre as 3 estimativas (baixo = cara de fundo)
    bgdist = np.min(np.stack([np.linalg.norm(lab_img - e, axis=2) for e in ests]), axis=0)
    # tau: limiar "consistente com fundo" = p85 das distâncias na região de fundo
    tau = float(np.percentile(bgdist[~mask_obj], 85))
    return ests, bgdist, tau


# ── Trimap + matting ──────────────────────────────────────────────────────────
def construir_trimap(heat_roi, bgdist_roi, tau):
    """heat_roi e bgdist_roi já recortados no ROI. Retorna trimap uint8 {0,128,255}."""
    h = heat_roi.copy()
    h = (h - h.min()) / (np.ptp(h) + 1e-9)
    fg_thr = np.percentile(h, 80)         # top 20% object-ness = FG definido
    bg_thr = np.percentile(h, 40)         # baixo object-ness +
    fg = h >= fg_thr
    bg = (bgdist_roi <= tau) & (h <= bg_thr)
    # limpeza morfológica leve p/ tirar sal-e-pimenta
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, kern).astype(bool)
    bg = cv2.morphologyEx(bg.astype(np.uint8), cv2.MORPH_OPEN, kern).astype(bool)
    tri = np.full(h.shape, 128, np.uint8)
    tri[bg] = 0
    tri[fg] = 255                          # FG vence empate
    return tri


def alpha_trimap(roi_rgb, trimap):
    from pymatting import estimate_alpha_cf
    H, W = roi_rgb.shape[:2]
    sc = MATTE_MAX / max(H, W) if max(H, W) > MATTE_MAX else 1.0
    if sc < 1.0:
        small = cv2.resize(roi_rgb, (int(W*sc), int(H*sc)), interpolation=cv2.INTER_AREA)
        trism = cv2.resize(trimap, (int(W*sc), int(H*sc)), interpolation=cv2.INTER_NEAREST)
    else:
        small, trism = roi_rgb, trimap
    a = estimate_alpha_cf(small.astype(np.float64)/255.0, trism.astype(np.float64)/255.0)
    if sc < 1.0:
        a = cv2.resize(a.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    return np.clip(a, 0, 1).astype(np.float32)


_rembg_sess = None
def alpha_rembg(roi_rgb):
    global _rembg_sess
    from rembg import new_session, remove
    from io import BytesIO
    if _rembg_sess is None:
        _rembg_sess = new_session("u2net")
    buf = BytesIO(); Image.fromarray(roi_rgb).save(buf, "PNG")
    m = np.array(Image.open(BytesIO(remove(buf.getvalue(), session=_rembg_sess))).convert("RGBA"))
    return (m[:, :, 3].astype(np.float32) / 255.0)


# ── Métricas (sem ground-truth, independentes do método) ──────────────────────
def metricas(alpha, bgdist_roi, tau, centroide_roi):
    H, W = alpha.shape
    ring = np.zeros((H, W), bool)
    m = max(1, int(0.12 * min(H, W)))
    ring[:m, :] = ring[-m:, :] = ring[:, :m] = ring[:, -m:] = True
    leak = float(alpha[ring].mean())                                   # menor melhor
    cy, cx = centroide_roi
    r = max(2, int(0.10 * min(H, W)))
    cov = float(alpha[max(0, cy-r):cy+r, max(0, cx-r):cx+r].mean())    # maior melhor
    bgmask = bgdist_roi <= tau
    resid = float(alpha[bgmask].mean()) if bgmask.any() else float("nan")  # menor melhor
    soft = float(((alpha > 0.05) & (alpha < 0.95)).mean())            # banda de borda
    return leak, cov, resid, soft


def aplica_escuro(roi_rgb, alpha):
    esc = (roi_rgb.astype(np.float32) * 0.10)
    out = roi_rgb.astype(np.float32) * alpha[..., None] + esc * (1 - alpha[..., None])
    return out.astype(np.uint8)


# ════════════════════════════════════════════════════════════════════════════
def main():
    d = json.load(open(CACHE))
    objs = d["objetos"]
    img_pil = ImageOps.exif_transpose(Image.open(FOTO)).convert("RGB")
    rgb = np.array(img_pil)
    H, W = rgb.shape[:2]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    print(f"[foto] {FOTO.name}  {W}x{H}  fundo='{d.get('fundo','')}'  n_obj={len(objs)}")

    # máscara de fundo conhecido = fora de todas as bboxes
    mask_obj = np.zeros((H, W), bool)
    for o in objs:
        b = o["bbox_normalizada"]
        mask_obj[int(b["y1"]*H):int(b["y2"]*H), int(b["x1"]*W):int(b["x2"]*W)] = True

    print("[bg] estimando vetor de fundo 3-vias...")
    ests, bgdist, tau = modelo_fundo_3vias(lab, mask_obj)
    print("     estimativas Lab (med/cluster/borda):",
          *[f"({e[0]:.0f},{e[1]:.0f},{e[2]:.0f})" for e in ests], " tau=%.1f" % tau)

    print("[dino] encodando patches da foto inteira (1 forward)...")
    grid = encode_patches(img_pil)
    S = grid.shape[0]

    print(f"\n{'#':>2} {'objeto':22} {'sep':>5} | {'   rembg (leak/cov/resid)':>26} | {'   trimap (leak/cov/resid)':>26}")
    print("-" * 92)

    linhas = []
    for i, o in enumerate(objs):
        nome = o["nome"][:22]
        b = o["bbox_normalizada"]
        cores = o.get("cores_dominantes") or []
        cen = o.get("centroide_normalizado") or {"cx": (b["x1"]+b["x2"])/2, "cy": (b["y1"]+b["y2"])/2}

        # heatmap DINOv2 (auto-similaridade) × peso de cor (mesmo da produção)
        mx, my = PAD, PAD
        cx1 = max(0.0, b["x1"]-mx); cy1 = max(0.0, b["y1"]-my)
        cx2 = min(1.0, b["x2"]+mx); cy2 = min(1.0, b["y2"]+my)
        crop = img_pil.crop((cx1*W, cy1*H, cx2*W, cy2*H))
        qv = encode_global(crop)
        heat = grid @ qv                                    # (64,64)
        if cores:
            heat = heat * GazetteerMatcher._color_weight(None, img_pil, cores, S)
        heat_full = cv2.resize(heat.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)

        # ROI (bbox + pad)
        bw, bh = (b["x2"]-b["x1"])*W, (b["y2"]-b["y1"])*H
        x1 = max(0, int(b["x1"]*W - bw*PAD)); y1 = max(0, int(b["y1"]*H - bh*PAD))
        x2 = min(W, int(b["x2"]*W + bw*PAD)); y2 = min(H, int(b["y2"]*H + bh*PAD))
        roi = rgb[y1:y2, x1:x2]
        heat_roi = heat_full[y1:y2, x1:x2]
        bgd_roi  = bgdist[y1:y2, x1:x2]
        cen_roi  = (int(cen["cy"]*H - y1), int(cen["cx"]*W - x1))

        # separabilidade: cor do objeto (mediana Lab dos top-heat) vs fundo
        rl = lab[y1:y2, x1:x2].reshape(-1, 3)
        hflat = heat_roi.reshape(-1)
        top = hflat >= np.percentile(hflat, 85)
        obj_lab = np.median(rl[top], axis=0)
        sep = float(np.min(np.linalg.norm(ests - obj_lab, axis=1)) / (bgd_roi.std() + 1e-6))

        # alphas
        t0 = time.time(); a_rb = alpha_rembg(roi);                    t_rb = time.time()-t0
        tri = construir_trimap(heat_roi, bgd_roi, tau)
        t0 = time.time(); a_tm = alpha_trimap(roi, tri);             t_tm = time.time()-t0

        m_rb = metricas(a_rb, bgd_roi, tau, cen_roi)
        m_tm = metricas(a_tm, bgd_roi, tau, cen_roi)
        linhas.append((nome, sep, m_rb, m_tm, t_rb, t_tm))
        print(f"{i:>2} {nome:22} {sep:5.2f} | "
              f"{m_rb[0]:.2f}/{m_rb[1]:.2f}/{m_rb[2]:.2f}  ({t_rb:.1f}s) | "
              f"{m_tm[0]:.2f}/{m_tm[1]:.2f}/{m_tm[2]:.2f}  ({t_tm:.1f}s)")

        # painel visual
        tri_viz = np.zeros_like(roi)
        tri_viz[tri == 255] = (0, 200, 0); tri_viz[tri == 0] = (200, 0, 0); tri_viz[tri == 128] = (90, 90, 90)
        panel = np.hstack([roi[:, :, ::-1], aplica_escuro(roi, a_rb)[:, :, ::-1],
                           aplica_escuro(roi, a_tm)[:, :, ::-1], tri_viz[:, :, ::-1]])
        cv2.imwrite(str(OUT / f"obj_{i:02d}_{nome.replace(' ', '_')}.jpg"), panel)

    # ── agregado + veredito ──────────────────────────────────────────────────
    print("-" * 92)
    arr = lambda j, k: np.array([l[j][k] for l in linhas])
    print("MÉDIAS        leak↓        cov↑        resid↓")
    print(f"  rembg    {arr(2,0).mean():.3f}      {arr(2,1).mean():.3f}      {arr(2,2).mean():.3f}")
    print(f"  trimap   {arr(3,0).mean():.3f}      {arr(3,1).mean():.3f}      {arr(3,2).mean():.3f}")
    print(f"\nPainéis salvos em {OUT}  (ROI | rembg | trimap | trimap-viz)")
    print("Legenda trimap-viz: verde=FG  vermelho=BG  cinza=incerto")


if __name__ == "__main__":
    main()
