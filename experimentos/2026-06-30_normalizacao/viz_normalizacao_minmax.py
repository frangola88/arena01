"""
Demonstração: RGB 3D → escalar único + normalização min-max por limite.

Técnica:
  1. RGB (R, G, B) → escalar L usando luminância perceptual
     L = 0.299·R + 0.587·G + 0.114·B
     (pesos ITU-R BT.601 — aproxima sensibilidade do olho humano)

  2. Normalização min-max ao limite escolhido (31, 127, 255):
     sup = (L - L.min) / (L.max - L.min) * limite
     → o pixel mais escuro da imagem vira 0
     → o pixel mais claro vira `limite`
     → todos os outros são interpolados linearmente entre eles

  Isso é diferente da técnica anterior (255 - gray) / limite que:
    - fixava o branco absoluto como 0 e comprimia a escala
  Aqui aproveitamos TODO o contraste existente na imagem.

Imagem: /home/cuco/Downloads/IMG_20260602_185719031.jpg (bancada c/ alicates)
"""

import sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from PIL import Image
from shapely.geometry import shape

sys.path.insert(0, '/home/cuco/projetos/casaiq')
from core.vetorizador_raster import vetorizar_superficie

IMG_PATH = '/home/cuco/Downloads/IMG_20260602_185719031.jpg'
GRID     = 280
THRESHOLD = 0.5
LIMITES  = [31, 127, 255]
NOMES    = ['L=31', 'L=127', 'L=255']
CORES    = ['#e84393', '#4488ff', '#22cc88']   # magenta, azul, verde


# ── helpers ─────────────────────────────────────────────────────────────────

def rgb_para_luminancia(img_pil: Image.Image, grid: int = GRID) -> np.ndarray:
    """Converte PIL RGB → array float [0, 255] via luminância perceptual."""
    img = img_pil.resize((grid, grid), Image.LANCZOS).convert('RGB')
    rgb = np.array(img, dtype=np.float32)
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def normalizar_minmax(L: np.ndarray, limite: int) -> np.ndarray:
    """
    Normalização min-max ao intervalo [0, limite]:
      sup[i,j] = (L[i,j] - L.min) / (L.max - L.min) * limite
    Retorna array float32 em [0, 1] para o vetorizador.
    """
    L_min, L_max = L.min(), L.max()
    if L_max == L_min:
        return np.zeros_like(L, dtype=np.float32)
    stretched = (L - L_min) / (L_max - L_min) * limite   # [0, limite]
    return (stretched / limite).astype(np.float32)        # normaliza p/ [0,1]


def vetorizar(sup: np.ndarray):
    res   = vetorizar_superficie(sup)
    feats = json.loads(res['geojson'])['features']
    polys = []
    for f in feats:
        try:
            g = shape(f['geometry'])
            if not g.is_empty:
                polys.append(g)
        except Exception:
            pass
    area  = sum(g.area for g in polys)
    pct   = 100.0 * np.sum(sup >= THRESHOLD) / (GRID * GRID)
    return polys, area, pct


def draw_polys(ax, img_arr, polys, color_cycle=None):
    ax.imshow(img_arr)
    cmap = plt.cm.tab10
    for i, geom in enumerate(polys):
        c = color_cycle[i % len(color_cycle)] if color_cycle else cmap(i % 10)
        xs, ys = geom.exterior.xy
        ax.fill(xs, ys, alpha=0.28, color=c)
        ax.plot(xs, ys, color=c, linewidth=1.5)
        ax.plot(geom.centroid.x, geom.centroid.y, '+', color=c, ms=9, mew=2)
    ax.set_xlim(0, GRID); ax.set_ylim(GRID, 0); ax.axis('off')


# ── carga ────────────────────────────────────────────────────────────────────

img_orig = Image.open(IMG_PATH).convert('RGB')
img_arr  = np.array(img_orig.resize((GRID, GRID), Image.LANCZOS))
L        = rgb_para_luminancia(img_orig)

print(f"Luminância L: min={L.min():.1f}  max={L.max():.1f}  "
      f"range={L.max()-L.min():.1f}  mean={L.mean():.1f}")

resultados = []
for lim, nome in zip(LIMITES, NOMES):
    sup = normalizar_minmax(L, lim)
    polys, area, pct = vetorizar(sup)
    resultados.append({'lim': lim, 'nome': nome, 'sup': sup,
                       'polys': polys, 'area': area, 'pct': pct})
    print(f"  [{nome}]  px≥{THRESHOLD}={pct:.1f}%  polys={len(polys)}  área={area:.0f}px²")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 1 — Explicação conceitual: RGB → escalar
# ══════════════════════════════════════════════════════════════════════════════

fig_concept, axes_c = plt.subplots(1, 5, figsize=(24, 5))
fig_concept.suptitle(
    'RGB 3D → Escalar único via Luminância Perceptual\n'
    r'$L = 0.299\,R + 0.587\,G + 0.114\,B$   '
    r'(ITU-R BT.601 — pesos imitam sensibilidade do olho humano)',
    fontsize=13, fontweight='bold', y=1.01)

# Col 0: original
axes_c[0].imshow(img_arr); axes_c[0].axis('off')
axes_c[0].set_title('Original RGB\n(3 canais independentes)', fontsize=10)

# Col 1: canal R
axes_c[1].imshow(img_arr[:, :, 0], cmap='Reds_r', vmin=0, vmax=255)
axes_c[1].axis('off')
axes_c[1].set_title(f'Canal R\nmean={img_arr[:,:,0].mean():.0f}', fontsize=10)

# Col 2: canal G
axes_c[2].imshow(img_arr[:, :, 1], cmap='Greens_r', vmin=0, vmax=255)
axes_c[2].axis('off')
axes_c[2].set_title(f'Canal G\nmean={img_arr[:,:,1].mean():.0f}', fontsize=10)

# Col 3: canal B
axes_c[3].imshow(img_arr[:, :, 2], cmap='Blues_r', vmin=0, vmax=255)
axes_c[3].axis('off')
axes_c[3].set_title(f'Canal B\nmean={img_arr[:,:,2].mean():.0f}', fontsize=10)

# Col 4: luminância L
im4 = axes_c[4].imshow(L, cmap='gray', vmin=0, vmax=255)
axes_c[4].axis('off')
axes_c[4].set_title(
    f'Luminância L (escalar único)\n'
    f'min={L.min():.0f}  max={L.max():.0f}  mean={L.mean():.0f}',
    fontsize=10)
plt.colorbar(im4, ax=axes_c[4], fraction=0.046, pad=0.04)

# anotação de fórmula
fig_concept.text(
    0.5, -0.06,
    'Cada pixel RGB(R,G,B) → um único número L.\n'
    r'Pesos: $w_R=0.299$ (vermelho, menos sensível), '
    r'$w_G=0.587$ (verde, mais sensível), '
    r'$w_B=0.114$ (azul, menos sensível).',
    ha='center', fontsize=10, style='italic',
    bbox=dict(boxstyle='round', facecolor='#fffbe6', alpha=0.8))

plt.tight_layout()
out_concept = '/tmp/rgb_para_escalar.png'
fig_concept.savefig(out_concept, dpi=130, bbox_inches='tight')
plt.close(fig_concept)
print(f'\nFigura 1 salva: {out_concept}')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 2 — Normalização min-max: 3 limites lado a lado
# ══════════════════════════════════════════════════════════════════════════════

fig2, axes2 = plt.subplots(4, 4, figsize=(22, 22))
fig2.suptitle(
    f'Normalização min-max — IMG_20260602 (bancada / alicates)\n'
    f'L: min={L.min():.0f}  max={L.max():.0f}  range={L.max()-L.min():.0f}  '
    f'→  sup[i,j] = (L[i,j] - {L.min():.0f}) / {L.max()-L.min():.0f} × limite',
    fontsize=12, fontweight='bold', y=1.005)

# Linha 0: cabeçalhos + original
axes2[0, 0].imshow(img_arr); axes2[0, 0].axis('off')
axes2[0, 0].set_title('Original', fontsize=11, fontweight='bold')

for col, r in enumerate(resultados, start=1):
    axes2[0, col].imshow(r['sup'], cmap='gray', vmin=0, vmax=1)
    axes2[0, col].axis('off')
    axes2[0, col].set_title(
        f"Luminância norm. [{r['nome']}]\n"
        f"min→0  max→{r['lim']}  ÷{r['lim']}→[0,1]",
        fontsize=9, fontweight='bold')

# Linha 1: histogramas
ax_hist_orig = axes2[1, 0]
ax_hist_orig.hist(L.ravel(), bins=64, color='gray', alpha=0.7)
ax_hist_orig.axvline(L.min(), color='red',  ls='--', lw=1.5, label=f'min={L.min():.0f}')
ax_hist_orig.axvline(L.max(), color='blue', ls='--', lw=1.5, label=f'max={L.max():.0f}')
ax_hist_orig.set_title('Histograma L original', fontsize=9)
ax_hist_orig.set_xlabel('L [0-255]'); ax_hist_orig.legend(fontsize=8)
ax_hist_orig.set_xlim(0, 255)

for col, r in enumerate(resultados, start=1):
    ax = axes2[1, col]
    vals = r['sup'].ravel() * r['lim']
    ax.hist(vals, bins=64, color=CORES[col-1], alpha=0.7)
    ax.axvline(r['lim'] * THRESHOLD, color='black', ls=':', lw=2,
               label=f'thr={r["lim"]*THRESHOLD:.0f}')
    ax.set_title(
        f"Histograma [{r['nome']}]\n"
        f"threshold={r['lim']*THRESHOLD:.0f} → {r['pct']:.1f}% pixels ativos",
        fontsize=9)
    ax.set_xlabel(f'Valor [0-{r["lim"]}]')
    ax.legend(fontsize=8)
    ax.set_xlim(0, r['lim'])

# Linha 2: isocontornos sobre a superficie
axes2[2, 0].imshow(img_arr); axes2[2, 0].axis('off')
axes2[2, 0].set_title('Original (referência)', fontsize=9)

for col, r in enumerate(resultados, start=1):
    ax = axes2[2, col]
    ax.imshow(r['sup'], cmap='gray', vmin=0, vmax=1, alpha=0.65)
    for lvl, ls in [(0.25, ':'), (0.50, '-'), (0.75, '--')]:
        try:
            cs = ax.contour(r['sup'], levels=[lvl],
                            colors=[CORES[col-1]], linewidths=1.8, linestyles=[ls])
            ax.clabel(cs, fmt=f'{lvl}', fontsize=7, colors=[CORES[col-1]])
        except Exception:
            pass
    ax.axis('off')
    ax.set_title(f"Isocontornos [{r['nome']}]\n(linhas 0.25 / 0.50 / 0.75)", fontsize=9)

# Linha 3: polígonos sobre imagem original
axes2[3, 0].imshow(img_arr); axes2[3, 0].axis('off')
axes2[3, 0].set_title('Original', fontsize=9)

for col, r in enumerate(resultados, start=1):
    ax = axes2[3, col]
    draw_polys(ax, img_arr, r['polys'], color_cycle=[CORES[col-1]])
    verd = '✅' if 5 <= r['pct'] <= 40 else ('⚠️' if r['pct'] <= 60 else '❌')
    ax.set_title(
        f"{verd} Polígonos [{r['nome']}]\n"
        f"{len(r['polys'])}p  cob={r['pct']:.1f}%  área={r['area']:.0f}px²",
        fontsize=9, fontweight='bold')

# labels de linha
for row_idx, label in enumerate(['Luminância norm.', 'Histograma', 'Isocontornos', 'Polígonos']):
    axes2[row_idx, 0].set_ylabel(label, fontsize=11, fontweight='bold',
                                  rotation=0, labelpad=70, va='center')

plt.tight_layout()
out2 = '/tmp/normalizacao_minmax_alicates.png'
fig2.savefig(out2, dpi=120, bbox_inches='tight')
plt.close(fig2)
print(f'Figura 2 salva: {out2}')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 3 — Comparação direta: antiga técnica (255-gray)/lim  vs  min-max
# ══════════════════════════════════════════════════════════════════════════════

def old_technique(L: np.ndarray, limite: int) -> np.ndarray:
    """Técnica anterior: (255 - L) clampado a [0, limite], dividido por limite."""
    return (np.clip(255.0 - L, 0, limite) / limite).astype(np.float32)

fig3, axes3 = plt.subplots(3, 4, figsize=(22, 16))
fig3.suptitle(
    'Comparação: Técnica ANTIGA (inversão fixa) vs NOVA (min-max da imagem)\n'
    'Imagem: bancada com alicates (fundo escuro misto)',
    fontsize=12, fontweight='bold')

col_labels = ['Original', 'L=31', 'L=127', 'L=255']
for c, lbl in enumerate(col_labels):
    axes3[0, c].set_title(lbl, fontsize=11, fontweight='bold', pad=6)

row_labels = ['Original', 'ANTIGA\n(255-L)/lim', 'NOVA\n(L-min)/(max-min)·lim']
for r_idx, rl in enumerate(row_labels):
    axes3[r_idx, 0].set_ylabel(rl, fontsize=10, fontweight='bold',
                                rotation=0, labelpad=70, va='center')

# Linha 0: original em todas as colunas
for c in range(4):
    axes3[0, c].imshow(img_arr); axes3[0, c].axis('off')

# Linhas 1 e 2: técnicas
for col, lim in enumerate(LIMITES, start=1):
    sup_old = old_technique(L, lim)
    sup_new = normalizar_minmax(L, lim)

    polys_old, area_old, pct_old = vetorizar(sup_old)
    polys_new, area_new, pct_new = vetorizar(sup_new)

    # Linha 1: antiga
    ax1 = axes3[1, col]
    draw_polys(ax1, img_arr, polys_old)
    v = '✅' if 5 <= pct_old <= 40 else ('⚠️' if pct_old <= 60 else '❌')
    ax1.set_title(f"{v} {len(polys_old)}p  cob={pct_old:.1f}%  área={area_old:.0f}px²",
                  fontsize=9)

    # Linha 2: nova
    ax2 = axes3[2, col]
    draw_polys(ax2, img_arr, polys_new)
    v = '✅' if 5 <= pct_new <= 40 else ('⚠️' if pct_new <= 60 else '❌')
    ax2.set_title(f"{v} {len(polys_new)}p  cob={pct_new:.1f}%  área={area_new:.0f}px²",
                  fontsize=9)

axes3[1, 0].imshow(img_arr); axes3[1, 0].axis('off')
axes3[2, 0].imshow(img_arr); axes3[2, 0].axis('off')

plt.tight_layout()
out3 = '/tmp/comparacao_antiga_vs_nova.png'
fig3.savefig(out3, dpi=120, bbox_inches='tight')
plt.close(fig3)
print(f'Figura 3 salva: {out3}')

import os
for f in [out_concept, out2, out3]:
    print(f"  {f}: {os.path.getsize(f)//1024} KB")
print('\nDone!')
