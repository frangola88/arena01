"""
demo_analise_cena.py — reconstrução de superfície de object-ness por seções ortogonais + RBF.

Princípio: N seções horizontais + M seções verticais na imagem bruta →
thin-plate spline → superfície 2D de "agitação local" → bg_mask, picos, complexidade.

Nenhum modelo ML, nenhuma API. Custo: ~1-3s CPU, zero $$.

Uso:
    conda run -n casaiq python poc_dinov2/demo_analise_cena.py

Saída: poc_dinov2/output/analise_cena_*.png
"""

import numpy as np
import cv2
from scipy.interpolate import griddata
from scipy.ndimage import label, maximum_filter, gaussian_filter
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time

# ─── parâmetros ────────────────────────────────────────────────────────────────
IMG_PATH   = Path("/home/cuco/Downloads/IMG_20260610_112623028.jpg")
BLOCK_SZ   = 16       # px por bloco para variância local
N_SECOES   = 48       # seções horizontais amostradas
M_SECOES   = 48       # seções verticais amostradas
PASSO_SAMP = 4        # pegar 1 em cada N blocos dentro de cada seção (esparsidade)
SIGMA_SUAV = 4.0      # suavização gaussiana do mapa de variância (blocos)
SIGMA_SUP  = 4.0      # pós-suavização da superfície reconstruída (atenua oscilações cubic)
TH_BG_PCT  = 25       # percentil abaixo → fundo
TH_OBJ_PCT = 65       # percentil acima → objeto candidato
MIN_PEAK_D = 12       # distância mínima entre picos (blocos)


# ─── funções ───────────────────────────────────────────────────────────────────

def variancia_local(img_gray: np.ndarray, block_sz: int) -> np.ndarray:
    """Variância em blocos não-sobrepostos → mapa (Hb × Wb) float32."""
    H, W = img_gray.shape
    Hb, Wb = H // block_sz, W // block_sz
    blocos = img_gray[:Hb*block_sz, :Wb*block_sz].reshape(Hb, block_sz, Wb, block_sz)
    return blocos.var(axis=(1, 3)).astype(np.float32)


def amostrar_secoes(mapa: np.ndarray, n_h: int, m_v: int, passo: int):
    """
    Amostra n_h seções horizontais + m_v seções verticais do mapa,
    retornando pontos (x_norm, y_norm) e valores.
    passo: pegar 1 a cada 'passo' elementos dentro de cada seção.
    """
    Hb, Wb = mapa.shape
    ys_h = np.linspace(0, Hb - 1, n_h, dtype=int)
    xs_v = np.linspace(0, Wb - 1, m_v, dtype=int)

    pts, vals = [], []

    # seções horizontais
    xs_all = np.arange(0, Wb, passo)
    for y in ys_h:
        for x in xs_all:
            pts.append((x / (Wb - 1), y / (Hb - 1)))
            vals.append(float(mapa[y, x]))

    # seções verticais
    ys_all = np.arange(0, Hb, passo)
    for x in xs_v:
        for y in ys_all:
            pts.append((x / (Wb - 1), y / (Hb - 1)))
            vals.append(float(mapa[y, x]))

    return np.array(pts, dtype=np.float64), np.array(vals, dtype=np.float64)


def reconstruir_griddata(pts, vals, shape):
    """Interpolação cúbica (griddata) sobre pontos amostrados → superfície completa."""
    Hb, Wb = shape
    gx = np.linspace(0, 1, Wb)
    gy = np.linspace(0, 1, Hb)
    GX, GY = np.meshgrid(gx, gy)
    sup = griddata(pts, vals, (GX, GY), method="cubic", fill_value=0.0)
    return sup.astype(np.float32)


def normalizar(arr):
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


def detectar_picos(superficie_norm, th_rel=0.60, min_dist=6):
    """
    Componentes conectados do obj_mask → 1 candidato por blob.
    Mais robusto que máximos locais para objetos alongados.
    """
    from scipy.ndimage import binary_closing, binary_opening
    mask = superficie_norm > th_rel
    # morfologia: fecha buracos internos, remove ruído fino
    mask = binary_closing(mask, iterations=2)
    mask = binary_opening(mask, iterations=1)
    labeled, n = label(mask)
    picos = []
    for i in range(1, n + 1):
        ys, xs = np.where(labeled == i)
        area = len(ys)
        if area < 4:          # ignora componentes muito pequenos (ruído)
            continue
        picos.append({
            "cx":    float(xs.mean() / (superficie_norm.shape[1] - 1)),
            "cy":    float(ys.mean() / (superficie_norm.shape[0] - 1)),
            "forca": float(superficie_norm[ys, xs].max()),
            "area_blocos": area,
        })
    return sorted(picos, key=lambda p: -p["forca"])


def largura_por_linha(superficie_norm, threshold):
    """
    W(y): largura do sinal acima do threshold em cada linha.
    Retorna array de shape (Hb,).
    Bordas com sinal ainda alto → objeto cortado.
    """
    mask = superficie_norm > threshold
    W = np.zeros(mask.shape[0], dtype=np.float32)
    for y in range(mask.shape[0]):
        row = mask[y]
        if row.any():
            xs = np.where(row)[0]
            W[y] = xs[-1] - xs[0] + 1
    return W


# ─── pipeline principal ─────────────────────────────────────────────────────────

def analisar_cena(img_path: Path, verbose=True) -> dict:
    t0 = time.time()

    img_bgr = cv2.imread(str(img_path))
    assert img_bgr is not None
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H_orig, W_orig = img_gray.shape

    if verbose:
        print(f"[analise_cena] imagem: {W_orig}×{H_orig}px")

    # 1. variância local → mapa de "agitação"
    t1 = time.time()
    var_map = variancia_local(img_gray, BLOCK_SZ)
    var_suav = gaussian_filter(var_map, sigma=SIGMA_SUAV)
    var_norm = normalizar(var_suav)
    Hb, Wb = var_norm.shape
    if verbose:
        print(f"  mapa variância: {Wb}×{Hb} blocos  ({time.time()-t1:.2f}s)")

    # 2. amostrar seções ortogonais
    t2 = time.time()
    pts, vals = amostrar_secoes(var_norm, N_SECOES, M_SECOES, PASSO_SAMP)
    n_pts = len(pts)
    if verbose:
        print(f"  seções: {N_SECOES}H + {M_SECOES}V  →  {n_pts} pontos amostrados  ({time.time()-t2:.2f}s)")

    # 3. reconstrução thin-plate spline
    t3 = time.time()
    sup_raw = reconstruir_griddata(pts, vals, (Hb, Wb))
    sup_raw = np.clip(sup_raw, 0, None)
    sup_suav = gaussian_filter(sup_raw, sigma=SIGMA_SUP)   # atenua oscilações cubic
    superficie = normalizar(sup_suav)
    if verbose:
        print(f"  griddata cubic reconstrução  ({time.time()-t3:.2f}s)")

    # 4. análise da superfície
    th_bg  = np.percentile(superficie, TH_BG_PCT)
    th_obj = np.percentile(superficie, TH_OBJ_PCT)
    bg_mask  = superficie < th_bg
    obj_mask = superficie > th_obj

    picos = detectar_picos(superficie, th_rel=th_obj, min_dist=MIN_PEAK_D)
    n_est = len(picos)
    complexidade = "simples" if n_est <= 5 else ("media" if n_est <= 12 else "densa")

    # 5. função de largura W(y) — diagnóstico de bordas
    W_func = largura_por_linha(superficie, th_obj)
    borda_top    = bool(superficie[0,  :]  .max() > th_obj)
    borda_bottom = bool(superficie[-1, :]  .max() > th_obj)
    borda_left   = bool(superficie[:,  0]  .max() > th_obj)
    borda_right  = bool(superficie[:, -1]  .max() > th_obj)

    resultado = {
        "superficie":         superficie,
        "var_norm":           var_norm,
        "bg_mask":            bg_mask,
        "obj_mask":           obj_mask,
        "pts_amostrados":     pts,
        "picos":              picos,
        "n_objetos_estimado": n_est,
        "complexidade":       complexidade,
        "W_func":             W_func,
        "bordas_ativas":      {
            "top": borda_top, "bottom": borda_bottom,
            "left": borda_left, "right": borda_right,
        },
        "shape_blocos":       (Hb, Wb),
        "shape_original":     (H_orig, W_orig),
        "tempo_s":            round(time.time() - t0, 2),
    }

    if verbose:
        print(f"\n  ── resultado ──")
        print(f"  n_objetos_estimado : {n_est}")
        print(f"  complexidade       : {complexidade}")
        print(f"  bordas com objeto  : {resultado['bordas_ativas']}")
        print(f"  tempo total        : {resultado['tempo_s']}s")

    return resultado


# ─── visualização ────────────────────────────────────────────────────────────────

def plotar(img_path: Path, res: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = img_path.stem
    Hb, Wb = res["shape_blocos"]

    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_thumb = cv2.resize(img_rgb, (Wb, Hb), interpolation=cv2.INTER_AREA)

    fig = plt.figure(figsize=(20, 14))
    gs  = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)

    # --- linha 1: imagem, variância, pontos amostrados, superfície ---
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img_thumb)
    ax.set_title(f"Imagem ({res['shape_original'][1]}×{res['shape_original'][0]}px)")
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(res["var_norm"], cmap="inferno", vmin=0, vmax=1)
    ax.set_title(f"Variância local (blocos {BLOCK_SZ}px)")
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(res["var_norm"], cmap="gray", alpha=0.4, vmin=0, vmax=1)
    px = res["pts_amostrados"][:, 0] * (Wb - 1)
    py = res["pts_amostrados"][:, 1] * (Hb - 1)
    ax.scatter(px, py, s=1, c="cyan", alpha=0.5)
    ax.set_title(f"Pontos amostrados\n{N_SECOES}H+{M_SECOES}V seções, passo={PASSO_SAMP} (n={len(px)})")
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 3])
    im = ax.imshow(res["superficie"], cmap="plasma", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Superfície griddata cubic\n(object-ness clássica)")
    ax.axis("off")

    # --- linha 2: bg_mask, obj_mask, picos sobre superfície, W(y) ---
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(res["bg_mask"], cmap="Blues")
    ax.set_title(f"bg_mask (percentil < {TH_BG_PCT})")
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(res["obj_mask"], cmap="Reds")
    ax.set_title(f"obj_mask (percentil > {TH_OBJ_PCT})")
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(res["superficie"], cmap="plasma", vmin=0, vmax=1, alpha=0.7)
    for p in res["picos"]:
        cx_px = p["cx"] * (Wb - 1)
        cy_px = p["cy"] * (Hb - 1)
        ax.plot(cx_px, cy_px, "w+", markersize=10, markeredgewidth=2)
        ax.annotate(f"{p['forca']:.2f}", (cx_px, cy_px),
                    color="white", fontsize=6, ha="center", va="bottom")
    n = res["n_objetos_estimado"]
    c = res["complexidade"]
    ax.set_title(f"Picos detectados: {n}  ({c})")
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 3])
    W = res["W_func"]
    ys = np.arange(len(W)) / (len(W) - 1)
    ax.plot(W, ys, color="steelblue", linewidth=1.5)
    ax.invert_yaxis()
    ax.set_xlabel("Largura sinal (blocos)")
    ax.set_ylabel("y normalizado")
    ax.set_title("W(y) — largura do objeto por linha\n(borda viva = objeto cortado)")
    ax.grid(True, alpha=0.3)
    bordas = res["bordas_ativas"]
    cor_top = "red" if bordas["top"] else "green"
    cor_bot = "red" if bordas["bottom"] else "green"
    ax.axhline(0,   color=cor_top, linestyle="--", linewidth=1.5, label=f"topo: {'⚠️ ativo' if bordas['top'] else 'ok'}")
    ax.axhline(1.0, color=cor_bot, linestyle="--", linewidth=1.5, label=f"base: {'⚠️ ativo' if bordas['bottom'] else 'ok'}")
    ax.legend(fontsize=7)

    # --- linha 3: comparação variância vs superfície (scatter) + perfis centrais ---
    ax = fig.add_subplot(gs[2, 0:2])
    sup_flat  = res["superficie"].ravel()
    var_flat  = res["var_norm"].ravel()
    idx = np.random.choice(len(sup_flat), size=min(3000, len(sup_flat)), replace=False)
    ax.scatter(var_flat[idx], sup_flat[idx], s=1, alpha=0.3, c="steelblue")
    ax.set_xlabel("Variância local (real)")
    ax.set_ylabel("Superfície RBF (reconstruída)")
    ax.set_title("Variância real vs. reconstruída\n(fidelidade da interpolação)")
    ax.grid(True, alpha=0.3)
    # linha diagonal
    ax.plot([0,1], [0,1], "r--", linewidth=1, alpha=0.5, label="y=x")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[2, 2:4])
    mid_row = Hb // 2
    mid_col = Wb // 2
    xs_norm = np.linspace(0, 1, Wb)
    ys_norm = np.linspace(0, 1, Hb)
    ax.plot(xs_norm, res["superficie"][mid_row, :], label=f"Seção H central (y={mid_row})", color="orangered")
    ax.plot(xs_norm, res["var_norm"][mid_row, :],   label="Variância real (H)", color="orangered", linestyle="--", alpha=0.5)
    ax.plot(ys_norm, res["superficie"][:, mid_col], label=f"Seção V central (x={mid_col})", color="steelblue")
    ax.plot(ys_norm, res["var_norm"][:, mid_col],   label="Variância real (V)", color="steelblue", linestyle="--", alpha=0.5)
    ax.axhline(np.percentile(res["superficie"], TH_OBJ_PCT), color="gray", linestyle=":", linewidth=1, label=f"th_obj ({TH_OBJ_PCT}p)")
    ax.axhline(np.percentile(res["superficie"], TH_BG_PCT),  color="lightgray", linestyle=":", linewidth=1, label=f"th_bg ({TH_BG_PCT}p)")
    ax.set_xlabel("Posição normalizada")
    ax.set_ylabel("Sinal")
    ax.set_title("Perfis centrais: RBF vs. variância real\n(seção H e V)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"analise_cena — {img_path.name}\n"
        f"n_objetos≈{n}  |  complexidade={c}  |  tempo={res['tempo_s']}s  |  "
        f"{N_SECOES}H+{M_SECOES}V seções, passo={PASSO_SAMP}",
        fontsize=13, fontweight="bold"
    )

    out_path = out_dir / f"analise_cena_{stem}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ✓ salvo: {out_path}")
    return out_path


# ─── main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    out_dir = Path("/home/cuco/projetos/casaiq/poc_dinov2/output")
    print(f"Analisando: {IMG_PATH.name}")
    res = analisar_cena(IMG_PATH, verbose=True)
    plotar(IMG_PATH, res, out_dir)
