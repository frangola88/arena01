"""
poc_dinov2/demo_analise_cena_multipass.py — 3 experimentos pedidos pelo usuário
sobre core/analise_cena.py:

  1) pass1 em resolução mais alta (block_sz/sigma menores, thresholds reajustados)
     comparado ao baseline atual — testa se dá pra separar objetos próximos
     sem explodir falso-positivo de textura.
  2) pass2 (zoom local em cada blob) com amostragem H+V+DIAGONAL direto na
     imagem crua do recorte (não rotacionando um mapa já suavizado, que foi
     o erro do experimento anterior demo_analise_cena_rotada.py).
  3) grid heterogêneo: nº de seções do pass2 escalado pela forma do recorte
     (não reusa 48x48 fixo num recorte fino/alongado).

Saída: nº de objetos por etapa, pra cada foto-mestre.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, label, binary_closing, binary_opening, find_objects

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.analise_cena import (  # noqa: E402
    BLOCK_SZ, N_SECOES, M_SECOES, PASSO_SAMP, SIGMA_SUAV, SIGMA_SUP,
    SIGMOID_K, SIGMOID_MU, TH_OBJ_PCT, AREA_MIN,
    _variancia_local, _normalizar, _amostrar_secoes, _sigmoid, _carregar_cinza,
)


# ─── 1. pass1 com resolução parametrizável ───────────────────────────────────

def reconstruir(shape, pts, vals, sigma_pos):
    from scipy.interpolate import griddata
    Hb, Wb = shape
    gx = np.linspace(0, 1, Wb)
    gy = np.linspace(0, 1, Hb)
    GX, GY = np.meshgrid(gx, gy)
    sup = griddata(pts, vals, (GX, GY), method="cubic", fill_value=0.0)
    sup = np.clip(sup, 0, None).astype(np.float32)
    if sigma_pos > 0:
        sup = gaussian_filter(sup, sigma=sigma_pos)
    return _normalizar(sup)


def detectar_bbox(superficie, th_obj_pct, area_min):
    """Como _detectar_objetos, mas devolve bbox (em coords de bloco) de cada objeto."""
    th = np.percentile(superficie, th_obj_pct)
    mask = superficie > th
    mask = binary_closing(mask, iterations=2)
    mask = binary_opening(mask, iterations=1)
    labeled, n = label(mask)
    slices = find_objects(labeled)
    objs = []
    for i, sl in enumerate(slices, start=1):
        area = int((labeled[sl] == i).sum())
        if area < area_min:
            continue
        y0, y1 = sl[0].start, sl[0].stop
        x0, x1 = sl[1].start, sl[1].stop
        objs.append({"bbox": (y0, y1, x0, x1), "area": area})
    return objs, mask


def pass1(img_gray, *, block_sz, sigma_pre, sigma_pos, n_sec, m_sec, passo,
          th_obj_pct, area_min):
    t0 = time.time()
    var_map = _variancia_local(img_gray, block_sz)
    var_norm = _normalizar(gaussian_filter(var_map, sigma=sigma_pre)) if sigma_pre > 0 \
        else _normalizar(var_map)
    Hb, Wb = var_norm.shape
    pts, vals = _amostrar_secoes(var_norm, n_sec, m_sec, passo)
    sup = reconstruir((Hb, Wb), pts, vals, sigma_pos)
    sup = _sigmoid(sup, SIGMOID_K, SIGMOID_MU)
    objs, _ = detectar_bbox(sup, th_obj_pct, area_min)
    return {
        "n_obj": len(objs), "objs": objs, "shape_blocos": (Hb, Wb),
        "block_sz": block_sz, "tempo_s": round(time.time() - t0, 3),
    }


# ─── 2+3. pass2 local: heterogêneo + diagonal, amostrado direto no recorte ──

def _amostrar_diagonais(mapa, n_diag, passo):
    """Amostra pontos ao longo de linhas a +-45 graus, direto no mapa denso
    (não rotaciona nada — o mapa já é a variância crua do recorte)."""
    Hb, Wb = mapa.shape
    pts, vals = [], []
    diag_len = int(np.hypot(Hb, Wb)) + 1
    ts = np.arange(0, diag_len, passo)
    offsets = np.linspace(-Wb, Hb, max(4, n_diag // 2), dtype=int)
    for c in offsets:
        for sinal in (1, -1):
            xs = ts
            ys = sinal * ts + c
            ok = (xs >= 0) & (xs < Wb) & (ys >= 0) & (ys < Hb)
            for x, y in zip(xs[ok], ys[ok]):
                pts.append((x / (Wb - 1 + 1e-9), y / (Hb - 1 + 1e-9)))
                vals.append(float(mapa[int(y), int(x)]))
    return pts, vals


def grid_dims_para_recorte(Hc, Wc, espaco_px=3, min_sec=8, max_sec=64):
    """Nº de seções H/V escalado pela forma do recorte (grid heterogêneo —
    não reusa 48x48 fixo num recorte fino e alongado)."""
    n_h = int(np.clip(round(Hc / espaco_px), min_sec, max_sec))
    m_v = int(np.clip(round(Wc / espaco_px), min_sec, max_sec))
    return n_h, m_v


def refinar_blob(img_gray, bbox_px, *, pad_px=40, block_sz_fino=4,
                  sigma_pre=0.6, sigma_pos=0.6, th_obj_pct=60, area_min=3,
                  usar_diagonais=True):
    H, W = img_gray.shape
    y0, y1, x0, x1 = bbox_px
    y0 = max(0, y0 - pad_px); x0 = max(0, x0 - pad_px)
    y1 = min(H, y1 + pad_px); x1 = min(W, x1 + pad_px)
    crop = img_gray[y0:y1, x0:x1]
    if crop.shape[0] < block_sz_fino * 4 or crop.shape[1] < block_sz_fino * 4:
        return {"n_sub": 1, "motivo": "recorte pequeno demais p/ refinar"}

    var_map = _variancia_local(crop, block_sz_fino)
    var_norm = _normalizar(gaussian_filter(var_map, sigma=sigma_pre)) if sigma_pre > 0 \
        else _normalizar(var_map)
    Hb, Wb = var_norm.shape

    n_h, m_v = grid_dims_para_recorte(Hb, Wb)  # heterogêneo (item 3)
    pts_hv, vals_hv = _amostrar_secoes(var_norm, n_h, m_v, passo=1)
    pts, vals = list(pts_hv), list(vals_hv)
    if usar_diagonais:
        pts_d, vals_d = _amostrar_diagonais(var_norm, n_diag=max(n_h, m_v), passo=1)
        pts += pts_d
        vals += vals_d

    sup = reconstruir((Hb, Wb), np.array(pts), np.array(vals), sigma_pos)
    sup = _sigmoid(sup, SIGMOID_K, SIGMOID_MU)
    objs, _ = detectar_bbox(sup, th_obj_pct, area_min)
    return {"n_sub": max(1, len(objs)), "n_h": n_h, "m_v": m_v, "shape_recorte": crop.shape}


# ─── runner ──────────────────────────────────────────────────────────────────

def avaliar(caminho: str):
    print(f"\n=== {Path(caminho).name} ===")
    img_gray = _carregar_cinza(caminho)

    base = pass1(img_gray, block_sz=BLOCK_SZ, sigma_pre=SIGMA_SUAV, sigma_pos=SIGMA_SUP,
                 n_sec=N_SECOES, m_sec=M_SECOES, passo=PASSO_SAMP,
                 th_obj_pct=TH_OBJ_PCT, area_min=AREA_MIN)
    print(f"  [1] baseline (block={BLOCK_SZ}, sigma={SIGMA_SUAV})       : "
          f"n_obj={base['n_obj']:2d}  tempo={base['tempo_s']:.2f}s")

    hires = pass1(img_gray, block_sz=8, sigma_pre=1.5, sigma_pos=1.5,
                  n_sec=64, m_sec=64, passo=2, th_obj_pct=70, area_min=6)
    print(f"  [1] hires    (block=8,  sigma=1.5)        : "
          f"n_obj={hires['n_obj']:2d}  tempo={hires['tempo_s']:.2f}s")

    # pass2: refina cada blob do hires (mais provável de já estar mais separado)
    escala = BLOCK_SZ  # pass1 usa block_sz=16 só p/ converter bbox base->pixel; hires usa 8
    fonte_objs = hires
    bsz = fonte_objs["block_sz"]
    print(f"  [2+3] refino local de cada blob do hires (H+V+diagonal, grid heterogêneo):")
    total_sub_diag = 0
    total_sub_hv = 0
    for i, o in enumerate(fonte_objs["objs"], start=1):
        y0, y1, x0, x1 = [v * bsz for v in o["bbox"]]
        r_diag = refinar_blob(img_gray, (y0, y1, x0, x1), usar_diagonais=True)
        r_hv = refinar_blob(img_gray, (y0, y1, x0, x1), usar_diagonais=False)
        total_sub_diag += r_diag["n_sub"]
        total_sub_hv += r_hv["n_sub"]
        print(f"      blob {i}: área={o['area']:3d}blk  H+V={r_hv['n_sub']}  "
              f"H+V+diag={r_diag['n_sub']}")
    print(f"  --- soma após refino local ---")
    print(f"  hires bruto         : {hires['n_obj']}")
    print(f"  refino só H+V       : {total_sub_hv}")
    print(f"  refino H+V+diagonal : {total_sub_diag}")


if __name__ == "__main__":
    alvos = sys.argv[1:] or [
        "data/fotos_mestras/IMG_mestra04semflash.jpg",
        "data/fotos_mestras/IMG_mestra04comflash.jpg",
    ]
    for a in alvos:
        avaliar(a)
