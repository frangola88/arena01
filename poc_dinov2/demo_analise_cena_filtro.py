"""
demo_analise_cena_filtro.py — griddata cubic 48×48 vs 100×100, com e sem filtro não-linear.

Filtros não-lineares testados:
  sigmoid   : 1/(1+exp(-k*(x-mu))) — transição abrupta fundo/objeto
  clahe     : equalização adaptativa local (cv2.CLAHE)
  gamma     : x^γ com γ>1 — suprime fundo, realça picos
  tophat    : erosão morfológica subtrai base local → só os picos sobram

Uso:
    conda run -n casaiq python poc_dinov2/demo_analise_cena_filtro.py
"""

import numpy as np
import cv2
from scipy.interpolate import griddata
from scipy.ndimage import label, gaussian_filter, binary_closing, binary_opening, grey_erosion
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

# ─── parâmetros ────────────────────────────────────────────────────────────────
IMG_PATH   = Path("/home/cuco/Downloads/IMG_20260610_112623028.jpg")
BLOCK_SZ   = 16
SIGMA_SUAV = 4.0
SIGMA_SUP  = 4.0
TH_BG_PCT  = 25
TH_OBJ_PCT = 65
PASSO_SAMP = 4

# filtro não-linear
SIGMOID_K  = 12.0    # inclinação da sigmoid (maior → mais abrupto)
SIGMOID_MU = 0.45    # ponto de inflexão (abaixo → mais objetos visíveis)
GAMMA      = 2.5     # expoente gamma (>1 suprime fundo)
CLAHE_CL   = 3.0     # clip limit CLAHE
TOPHAT_SZ  = 15      # tamanho da janela de erosão (blocos)


# ─── utilitários ───────────────────────────────────────────────────────────────

def variancia_local(img_gray, block_sz):
    H, W = img_gray.shape
    Hb, Wb = H // block_sz, W // block_sz
    bl = img_gray[:Hb*block_sz, :Wb*block_sz].reshape(Hb, block_sz, Wb, block_sz)
    return bl.var(axis=(1, 3)).astype(np.float32)


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


def reconstruir(var_norm, pts, vals):
    Hb, Wb = var_norm.shape
    gx = np.linspace(0, 1, Wb)
    gy = np.linspace(0, 1, Hb)
    GX, GY = np.meshgrid(gx, gy)
    sup = griddata(pts, vals, (GX, GY), method="cubic", fill_value=0.0)
    sup = np.clip(sup, 0, None)
    sup = gaussian_filter(sup.astype(np.float32), sigma=SIGMA_SUP)
    return normalizar(sup)


def detectar_objetos(superficie_norm):
    th = np.percentile(superficie_norm, TH_OBJ_PCT)
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
    return sorted(picos, key=lambda p: -p["forca"])


# ─── filtros não-lineares ───────────────────────────────────────────────────────

def filtro_sigmoid(sup):
    """Transição abrupta fundo/objeto. k alto → quase binariza."""
    return normalizar(1.0 / (1.0 + np.exp(-SIGMOID_K * (sup - SIGMOID_MU))))


def filtro_clahe(sup):
    """Equalização adaptativa local — realça estrutura regional."""
    u8 = (sup * 255).clip(0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CL, tileGridSize=(8, 8))
    return normalizar(clahe.apply(u8).astype(np.float32))


def filtro_gamma(sup):
    """Potência γ>1 suprime fundo próximo de 0, realça picos."""
    return normalizar(sup ** GAMMA)


def filtro_tophat(sup):
    """Top-hat: subtrai background local por erosão morfológica."""
    eroded = grey_erosion(sup, size=TOPHAT_SZ)
    return normalizar(np.clip(sup - eroded, 0, None))


FILTROS = {
    "sigmoid": filtro_sigmoid,
    "clahe":   filtro_clahe,
    "gamma":   filtro_gamma,
    "top-hat": filtro_tophat,
}


# ─── pipeline por configuração ──────────────────────────────────────────────────

def pipeline(var_norm, n_secoes, label):
    pts, vals = amostrar_secoes(var_norm, n_secoes, n_secoes, PASSO_SAMP)
    t0 = time.time()
    sup = reconstruir(var_norm, pts, vals)
    t_rec = time.time() - t0
    picos = detectar_objetos(sup)
    return sup, picos, t_rec, len(pts)


# ─── main ───────────────────────────────────────────────────────────────────────

def main():
    img_bgr  = cv2.imread(str(IMG_PATH))
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H_orig, W_orig = img_gray.shape

    var_map  = variancia_local(img_gray, BLOCK_SZ)
    var_suav = gaussian_filter(var_map, sigma=SIGMA_SUAV)
    var_norm = normalizar(var_suav)
    Hb, Wb   = var_norm.shape
    thumb    = cv2.resize(img_rgb, (Wb, Hb), interpolation=cv2.INTER_AREA)

    print(f"Imagem {W_orig}×{H_orig}  |  mapa {Wb}×{Hb}")

    # reconstrução 48 e 100
    print("48×48 ...", end=" ", flush=True)
    sup48, picos48, t48, n48 = pipeline(var_norm, 48, "48")
    print(f"{t48:.2f}s → {len(picos48)} obj")

    print("100×100 ...", end=" ", flush=True)
    sup100, picos100, t100, n100 = pipeline(var_norm, 100, "100")
    print(f"{t100:.2f}s → {len(picos100)} obj")

    # aplicar todos os filtros nas duas superfícies
    resultados = {}
    for nome, fn in FILTROS.items():
        f48  = fn(sup48)
        f100 = fn(sup100)
        p48  = detectar_objetos(f48)
        p100 = detectar_objetos(f100)
        resultados[nome] = {
            "48":  (f48,  p48),
            "100": (f100, p100),
        }
        print(f"  {nome:8s}  48→{len(p48):2d} obj  100→{len(p100):2d} obj")

    # ─── plot ──────────────────────────────────────────────────────────────────
    # linhas: [bruta-48, bruta-100, sigmoid, clahe, gamma, top-hat]
    # colunas: [sup, obj_mask, picos]  ×2  (48 e 100)
    nfiltros = len(FILTROS)
    nrows = 2 + nfiltros     # 2 linhas brutas + 1 por filtro
    ncols = 6                # sup48 | mask48 | picos48 | sup100 | mask100 | picos100

    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 3.5 * nrows))
    fig.subplots_adjust(hspace=0.3, wspace=0.15)

    def show(ax, data, title, cmap="plasma", vmin=0, vmax=1):
        ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=7.5)
        ax.axis("off")

    def show_picos(ax, sup, picos, title):
        th = np.percentile(sup, TH_OBJ_PCT)
        mask = sup > th
        mask = binary_closing(mask, iterations=2)
        mask = binary_opening(mask, iterations=1)
        ax.imshow(sup, cmap="plasma", vmin=0, vmax=1, alpha=0.8)
        ax.imshow(np.ma.masked_where(~mask, np.ones_like(sup)),
                  cmap="cool", alpha=0.25, vmin=0, vmax=1)
        for p in picos:
            cx = p["cx"] * (Wb - 1)
            cy = p["cy"] * (Hb - 1)
            ax.plot(cx, cy, "w+", markersize=8, markeredgewidth=1.8)
        ax.set_title(title, fontsize=7.5)
        ax.axis("off")

    def linha(row, sup48_, sup100_, label_, n48_, n100_, picos48_=None, picos100_=None):
        if picos48_  is None: picos48_  = detectar_objetos(sup48_)
        if picos100_ is None: picos100_ = detectar_objetos(sup100_)
        th48  = np.percentile(sup48_,  TH_OBJ_PCT)
        th100 = np.percentile(sup100_, TH_OBJ_PCT)
        mask48  = binary_closing(binary_opening(sup48_  > th48,  iterations=1), iterations=2)
        mask100 = binary_closing(binary_opening(sup100_ > th100, iterations=1), iterations=2)
        show(axes[row, 0], sup48_,  f"{label_}\n48×48 ({t48:.2f}s  n={n48_})")
        show(axes[row, 1], mask48,  f"obj_mask 48", cmap="Reds")
        show_picos(axes[row, 2], sup48_,  picos48_,  f"picos 48: {len(picos48_)}")
        show(axes[row, 3], sup100_, f"{label_}\n100×100 ({t100:.2f}s  n={n100_})")
        show(axes[row, 4], mask100, f"obj_mask 100", cmap="Reds")
        show_picos(axes[row, 5], sup100_, picos100_, f"picos 100: {len(picos100_)}")

    # cabeçalho de colunas
    for ax, title in zip(axes[0], ["sup 48×48", "obj_mask 48", "picos 48",
                                    "sup 100×100", "obj_mask 100", "picos 100"]):
        ax.set_title(title, fontsize=8, fontweight="bold")

    linha(0, sup48, sup100, "bruta (sem filtro)", n48, n100, picos48, picos100)
    for r, (nome, res) in enumerate(resultados.items(), start=1):
        f48, p48   = res["48"]
        f100, p100 = res["100"]
        linha(r, f48, f100, nome, n48, n100, p48, p100)

    # linha extra: histogramas de comparação (bruta vs melhor filtro)
    # (usada para o usuário ver a diferença de distribuição)
    # → substituímos por linha de thumb + curvas de filtro
    row_last = nrows - 1

    # perfil H central — comparação todos os filtros em 48×48
    ax_p = axes[row_last, 0]
    mid_row = Hb // 2
    xs = np.linspace(0, 1, Wb)
    ax_p.plot(xs, sup48[mid_row, :], label="bruta", linewidth=1.2, color="black")
    colors = ["orangered", "steelblue", "seagreen", "purple"]
    for (nome, _), c in zip(FILTROS.items(), colors):
        ax_p.plot(xs, resultados[nome]["48"][0][mid_row, :],
                  label=nome, linewidth=1, color=c, alpha=0.8)
    ax_p.set_title("Perfil H central — 48×48", fontsize=7.5)
    ax_p.legend(fontsize=6, loc="upper right")
    ax_p.grid(True, alpha=0.3)
    ax_p.axis("on")

    # perfil H central — 100×100
    ax_p2 = axes[row_last, 3]
    ax_p2.plot(xs, sup100[mid_row, :], label="bruta", linewidth=1.2, color="black")
    for (nome, _), c in zip(FILTROS.items(), colors):
        ax_p2.plot(xs, resultados[nome]["100"][0][mid_row, :],
                   label=nome, linewidth=1, color=c, alpha=0.8)
    ax_p2.set_title("Perfil H central — 100×100", fontsize=7.5)
    ax_p2.legend(fontsize=6, loc="upper right")
    ax_p2.grid(True, alpha=0.3)
    ax_p2.axis("on")

    # histograma 48 bruta vs filtros
    ax_h = axes[row_last, 1]
    ax_h.hist(sup48.ravel(), bins=60, color="black", alpha=0.4, label="bruta", density=True)
    for (nome, _), c in zip(FILTROS.items(), colors):
        ax_h.hist(resultados[nome]["48"][0].ravel(), bins=60,
                  color=c, alpha=0.35, label=nome, density=True)
    ax_h.set_title("Histograma 48×48", fontsize=7.5)
    ax_h.legend(fontsize=6)
    ax_h.axis("on")

    # tabela resumo
    ax_t = axes[row_last, 4]
    ax_t.axis("off")
    rows_t = [["filtro", "48 obj", "100 obj"]]
    rows_t.append(["bruta", str(len(picos48)), str(len(picos100))])
    for nome, res in resultados.items():
        rows_t.append([nome, str(len(res["48"][1])), str(len(res["100"][1]))])
    rows_t.append(["──", "──", "──"])
    rows_t.append(["REAL", "12", "12"])
    tbl = ax_t.table(cellText=rows_t[1:], colLabels=rows_t[0],
                     loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.1, 1.8)
    ax_t.set_title("Resumo (real=12)", fontsize=8, pad=10)

    # imagem original
    show(axes[row_last, 2], thumb, "imagem original", cmap=None)
    axes[row_last, 2].imshow(thumb)
    axes[row_last, 2].axis("off")

    show(axes[row_last, 5], thumb, "", cmap=None)
    axes[row_last, 5].imshow(thumb)
    axes[row_last, 5].axis("off")

    fig.suptitle(
        f"griddata cubic — filtros não-lineares | {IMG_PATH.name}\n"
        f"σ_var={SIGMA_SUAV}  σ_sup={SIGMA_SUP}  sigmoid k={SIGMOID_K} μ={SIGMOID_MU}  "
        f"γ={GAMMA}  tophat={TOPHAT_SZ}blocos  CLAHE cl={CLAHE_CL}",
        fontsize=11, fontweight="bold"
    )

    out = Path("/home/cuco/projetos/casaiq/poc_dinov2/output/analise_cena_filtro.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✓ salvo: {out}")


if __name__ == "__main__":
    main()
