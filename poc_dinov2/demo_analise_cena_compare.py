"""
demo_analise_cena_compare.py — griddata cubic vs. Kriging (RBFInterpolator thin-plate spline)
com 100×100 seções ortogonais.

Uso:
    conda run -n casaiq python poc_dinov2/demo_analise_cena_compare.py
"""

import numpy as np
import cv2
from scipy.interpolate import griddata, RBFInterpolator
from scipy.ndimage import label, gaussian_filter, binary_closing, binary_opening
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

# ─── parâmetros ────────────────────────────────────────────────────────────────
IMG_PATH   = Path("/home/cuco/Downloads/IMG_20260610_112623028.jpg")
BLOCK_SZ   = 16
N_SECOES   = 100
M_SECOES   = 100
PASSO_SAMP = 4
SIGMA_SUAV = 4.0
SIGMA_SUP  = 4.0
TH_BG_PCT  = 25
TH_OBJ_PCT = 65
RBF_SMOOTH = 1.0    # smoothing Kriging (0 = interpolação exata)


# ─── utilitários ───────────────────────────────────────────────────────────────

def variancia_local(img_gray, block_sz):
    H, W = img_gray.shape
    Hb, Wb = H // block_sz, W // block_sz
    blocos = img_gray[:Hb*block_sz, :Wb*block_sz].reshape(Hb, block_sz, Wb, block_sz)
    return blocos.var(axis=(1, 3)).astype(np.float32)


def normalizar(arr):
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


def amostrar_secoes(mapa, n_h, m_v, passo):
    Hb, Wb = mapa.shape
    ys_h = np.linspace(0, Hb - 1, n_h, dtype=int)
    xs_v = np.linspace(0, Wb - 1, m_v, dtype=int)
    pts, vals = [], []
    for y in ys_h:
        for x in np.arange(0, Wb, passo):
            pts.append((x / (Wb - 1), y / (Hb - 1)))
            vals.append(float(mapa[y, x]))
    for x in xs_v:
        for y in np.arange(0, Hb, passo):
            pts.append((x / (Wb - 1), y / (Hb - 1)))
            vals.append(float(mapa[y, x]))
    return np.array(pts, dtype=np.float64), np.array(vals, dtype=np.float64)


def grade_query(shape):
    Hb, Wb = shape
    gx = np.linspace(0, 1, Wb)
    gy = np.linspace(0, 1, Hb)
    GX, GY = np.meshgrid(gx, gy)
    return GX, GY


def detectar_objetos(superficie_norm, th_pct=TH_OBJ_PCT):
    th = np.percentile(superficie_norm, th_pct)
    mask = superficie_norm > th
    mask = binary_closing(mask, iterations=2)
    mask = binary_opening(mask, iterations=1)
    labeled, n = label(mask)
    picos = []
    for i in range(1, n + 1):
        ys, xs = np.where(labeled == i)
        if len(ys) < 4:
            continue
        picos.append({
            "cx": float(xs.mean() / (superficie_norm.shape[1] - 1)),
            "cy": float(ys.mean() / (superficie_norm.shape[0] - 1)),
            "forca": float(superficie_norm[ys, xs].max()),
        })
    return sorted(picos, key=lambda p: -p["forca"]), mask


# ─── os dois métodos ────────────────────────────────────────────────────────────

def metodo_griddata(var_norm, pts, vals):
    t = time.time()
    GX, GY = grade_query(var_norm.shape)
    sup = griddata(pts, vals, (GX, GY), method="cubic", fill_value=0.0)
    sup = np.clip(sup, 0, None)
    sup = gaussian_filter(sup.astype(np.float32), sigma=SIGMA_SUP)
    sup = normalizar(sup)
    return sup, time.time() - t


def metodo_kriging(var_norm, pts, vals):
    t = time.time()
    rbf = RBFInterpolator(pts, vals, kernel="thin_plate_spline", smoothing=RBF_SMOOTH)
    GX, GY = grade_query(var_norm.shape)
    query = np.column_stack([GX.ravel(), GY.ravel()])
    sup = rbf(query).reshape(var_norm.shape).astype(np.float32)
    sup = np.clip(sup, 0, None)
    sup = gaussian_filter(sup, sigma=SIGMA_SUP)
    sup = normalizar(sup)
    return sup, time.time() - t


# ─── main ───────────────────────────────────────────────────────────────────────

def main():
    img_bgr  = cv2.imread(str(IMG_PATH))
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H_orig, W_orig = img_gray.shape

    # variância pré-suavizada
    t0 = time.time()
    var_map  = variancia_local(img_gray, BLOCK_SZ)
    var_suav = gaussian_filter(var_map, sigma=SIGMA_SUAV)
    var_norm = normalizar(var_suav)
    Hb, Wb   = var_norm.shape
    t_var    = time.time() - t0

    pts, vals = amostrar_secoes(var_norm, N_SECOES, M_SECOES, PASSO_SAMP)
    n_pts = len(pts)
    print(f"Imagem {W_orig}×{H_orig}  |  mapa {Wb}×{Hb} blocos  |  {n_pts} pontos amostrados")

    # griddata cubic
    print(f"\n[griddata cubic] reconstruindo...", end=" ", flush=True)
    sup_gd, t_gd = metodo_griddata(var_norm, pts, vals)
    picos_gd, mask_gd = detectar_objetos(sup_gd)
    print(f"{t_gd:.2f}s  →  {len(picos_gd)} objetos")

    # Kriging
    print(f"[Kriging RBF]    reconstruindo...", end=" ", flush=True)
    sup_kr, t_kr = metodo_kriging(var_norm, pts, vals)
    picos_kr, mask_kr = detectar_objetos(sup_kr)
    print(f"{t_kr:.2f}s  →  {len(picos_kr)} objetos")

    # ─── plot ──────────────────────────────────────────────────────────────────
    thumb = cv2.resize(img_rgb, (Wb, Hb), interpolation=cv2.INTER_AREA)

    fig, axes = plt.subplots(3, 5, figsize=(24, 14))
    fig.subplots_adjust(hspace=0.35, wspace=0.25)

    def show(ax, data, title, cmap="plasma", vmin=0, vmax=1):
        ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    def show_picos(ax, superficie, picos, title):
        ax.imshow(superficie, cmap="plasma", vmin=0, vmax=1, alpha=0.85)
        for p in picos:
            cx = p["cx"] * (Wb - 1)
            cy = p["cy"] * (Hb - 1)
            ax.plot(cx, cy, "w+", markersize=9, markeredgewidth=2)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    # coluna 0: referência
    show(axes[0, 0], thumb,    "Imagem original")
    show(axes[1, 0], var_norm, "Variância local (σ=4)", cmap="inferno")
    # pontos amostrados
    axes[2, 0].imshow(var_norm, cmap="gray", alpha=0.3, vmin=0, vmax=1)
    px = pts[:, 0] * (Wb - 1)
    py = pts[:, 1] * (Hb - 1)
    axes[2, 0].scatter(px, py, s=0.5, c="cyan", alpha=0.4)
    axes[2, 0].set_title(f"{N_SECOES}H+{M_SECOES}V seções\npasso={PASSO_SAMP}  n={n_pts}", fontsize=9)
    axes[2, 0].axis("off")

    # colunas 1-2: griddata
    show(axes[0, 1], sup_gd,  f"griddata cubic\n({t_gd:.2f}s)")
    show(axes[1, 1], mask_gd, "obj_mask (cubic)", cmap="Reds", vmin=0, vmax=1)
    show_picos(axes[2, 1], sup_gd, picos_gd, f"Picos: {len(picos_gd)}  (real=12)")

    # colunas 3-4: Kriging
    show(axes[0, 2], sup_kr,  f"Kriging thin-plate\n({t_kr:.2f}s)")
    show(axes[1, 2], mask_kr, "obj_mask (Kriging)", cmap="Reds", vmin=0, vmax=1)
    show_picos(axes[2, 2], sup_kr, picos_kr, f"Picos: {len(picos_kr)}  (real=12)")

    # diferença entre superfícies
    diff = sup_gd - sup_kr
    im = axes[0, 3].imshow(diff, cmap="RdBu_r", vmin=-0.4, vmax=0.4)
    axes[0, 3].set_title("Diferença (cubic − Kriging)", fontsize=9)
    axes[0, 3].axis("off")
    plt.colorbar(im, ax=axes[0, 3], fraction=0.046)

    # scatter cubic vs kriging
    flat_gd = sup_gd.ravel()
    flat_kr = sup_kr.ravel()
    idx = np.random.choice(len(flat_gd), size=3000, replace=False)
    axes[1, 3].scatter(flat_gd[idx], flat_kr[idx], s=1, alpha=0.3, c="steelblue")
    axes[1, 3].plot([0, 1], [0, 1], "r--", linewidth=1, alpha=0.6)
    axes[1, 3].set_xlabel("griddata cubic")
    axes[1, 3].set_ylabel("Kriging")
    axes[1, 3].set_title("Correlação superfícies", fontsize=9)
    axes[1, 3].grid(True, alpha=0.3)
    corr = float(np.corrcoef(flat_gd, flat_kr)[0, 1])
    axes[1, 3].text(0.05, 0.92, f"r={corr:.3f}", transform=axes[1, 3].transAxes, fontsize=9)

    # perfil horizontal central
    mid_row = Hb // 2
    xs_norm = np.linspace(0, 1, Wb)
    axes[2, 3].plot(xs_norm, sup_gd[mid_row, :], label="griddata", color="orangered", linewidth=1.5)
    axes[2, 3].plot(xs_norm, sup_kr[mid_row, :], label="Kriging",  color="steelblue", linewidth=1.5)
    axes[2, 3].plot(xs_norm, var_norm[mid_row, :], label="var real", color="gray", linewidth=1, linestyle="--", alpha=0.6)
    axes[2, 3].set_title(f"Perfil H central (y={mid_row})", fontsize=9)
    axes[2, 3].legend(fontsize=7)
    axes[2, 3].grid(True, alpha=0.3)

    # perfil vertical central
    mid_col = Wb // 2
    ys_norm = np.linspace(0, 1, Hb)
    axes[0, 4].plot(sup_gd[:, mid_col], ys_norm, label="griddata", color="orangered", linewidth=1.5)
    axes[0, 4].plot(sup_kr[:, mid_col], ys_norm, label="Kriging",  color="steelblue", linewidth=1.5)
    axes[0, 4].plot(var_norm[:, mid_col], ys_norm, label="var real", color="gray", linewidth=1, linestyle="--", alpha=0.6)
    axes[0, 4].invert_yaxis()
    axes[0, 4].set_title(f"Perfil V central (x={mid_col})", fontsize=9)
    axes[0, 4].legend(fontsize=7)
    axes[0, 4].grid(True, alpha=0.3)

    # W(y) ambos
    def w_func(sup, th_pct):
        th = np.percentile(sup, th_pct)
        mask = sup > th
        W = np.zeros(mask.shape[0])
        for y in range(mask.shape[0]):
            row = mask[y]
            if row.any():
                xs = np.where(row)[0]
                W[y] = xs[-1] - xs[0] + 1
        return W
    W_gd = w_func(sup_gd, TH_OBJ_PCT)
    W_kr = w_func(sup_kr, TH_OBJ_PCT)
    axes[1, 4].plot(W_gd, ys_norm, label="griddata", color="orangered", linewidth=1.5)
    axes[1, 4].plot(W_kr, ys_norm, label="Kriging",  color="steelblue", linewidth=1.5)
    axes[1, 4].invert_yaxis()
    axes[1, 4].set_xlabel("Largura (blocos)")
    axes[1, 4].set_title("W(y) — largura por linha", fontsize=9)
    axes[1, 4].legend(fontsize=7)
    axes[1, 4].grid(True, alpha=0.3)

    # tabela resumo
    axes[2, 4].axis("off")
    tabela = [
        ["",              "griddata cubic",   "Kriging RBF"],
        ["Tempo",         f"{t_gd:.2f}s",     f"{t_kr:.2f}s"],
        ["n_obj estimado",str(len(picos_gd)), str(len(picos_kr))],
        ["n_obj real",    "12",               "12"],
        ["Erro",          str(abs(len(picos_gd)-12)), str(abs(len(picos_kr)-12))],
        ["Correlação",    f"r={corr:.3f}",    "(referência)"],
        ["Pontos RBF",    f"{n_pts}",         f"{n_pts}"],
    ]
    tbl = axes[2, 4].table(cellText=tabela[1:], colLabels=tabela[0],
                            loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.8)
    axes[2, 4].set_title("Resumo comparativo", fontsize=9, pad=12)

    fig.suptitle(
        f"analise_cena: griddata cubic vs. Kriging — {N_SECOES}×{N_SECOES} seções, passo={PASSO_SAMP}\n"
        f"{IMG_PATH.name}  |  variância pré-σ={SIGMA_SUAV}, pós-σ={SIGMA_SUP}",
        fontsize=12, fontweight="bold"
    )

    out = Path("/home/cuco/projetos/casaiq/poc_dinov2/output/analise_cena_compare.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✓ salvo: {out}")


if __name__ == "__main__":
    main()
