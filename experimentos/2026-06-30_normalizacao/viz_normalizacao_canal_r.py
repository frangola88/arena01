"""
Técnica: normalização pelo canal R, replicada para G e B.

Passo a passo:
  1. Extraia o canal R da imagem (R_orig, valores 0-255)
  2. Encontre R_min e R_max na imagem toda
  3. Regra de três: R_novo = (R_orig - R_min) / (R_max - R_min) × limite
  4. Copie R_novo para G_novo e B_novo
     → resultado: imagem RGB(R_novo, R_novo, R_novo) = tons de cinza puro
  5. Para o vetorizador: sup = R_novo / limite  ∈ [0, 1]

Imagem: /home/cuco/Downloads/IMG_20260602_185719031.jpg (bancada com alicates)
"""

import sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from shapely.geometry import shape

sys.path.insert(0, '/home/cuco/projetos/casaiq')
from core.vetorizador_raster import vetorizar_superficie

IMG_PATH  = '/home/cuco/Downloads/IMG_20260602_185719031.jpg'
GRID      = 280
THRESHOLD = 0.5
LIMITES   = [31, 127, 255]
CORES_LIM = ['#e84393', '#4488ff', '#22cc88']


def carregar_rgb(path, grid=GRID):
    img = Image.open(path).convert('RGB').resize((grid, grid), Image.LANCZOS)
    return np.array(img, dtype=np.float32)


def normalizar_por_R(rgb: np.ndarray, limite: int):
    """
    Normaliza usando APENAS o canal R como referência.
    Retorna:
      - gray_img: array uint8 (grid, grid, 3) com R=G=B (tons de cinza visível)
      - sup: float32 [0,1] para o vetorizador
    """
    R = rgb[:, :, 0]
    R_min, R_max = R.min(), R.max()
    # regra de três: R_orig → [0, limite]
    R_novo = (R - R_min) / (R_max - R_min) * limite   # float em [0, limite]
    # replica para G e B → tons de cinza
    gray_img = np.stack([R_novo, R_novo, R_novo], axis=-1).clip(0, limite)
    gray_uint8 = (gray_img * (255 / limite)).clip(0, 255).astype(np.uint8)
    # superficie para vetorizador: divide pelo limite → [0, 1]
    sup = (R_novo / limite).astype(np.float32)
    return R_min, R_max, gray_uint8, sup


def vetorizar(sup):
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
    area = sum(g.area for g in polys)
    pct  = 100.0 * np.sum(sup >= THRESHOLD) / (GRID * GRID)
    return polys, area, pct


def draw_polys(ax, bg, polys, color):
    ax.imshow(bg)
    for geom in polys:
        xs, ys = geom.exterior.xy
        ax.fill(xs, ys, alpha=0.30, color=color)
        ax.plot(xs, ys, color=color, linewidth=1.8)
        ax.plot(geom.centroid.x, geom.centroid.y,
                '+', color=color, ms=10, mew=2.5)
    ax.set_xlim(0, GRID); ax.set_ylim(GRID, 0); ax.axis('off')


# ── carga ────────────────────────────────────────────────────────────────────

rgb  = carregar_rgb(IMG_PATH)
img_orig_arr = rgb.clip(0, 255).astype(np.uint8)
R    = rgb[:, :, 0]

print(f"Canal R — min={R.min():.1f}  max={R.max():.1f}  "
      f"mean={R.mean():.1f}  range={R.max()-R.min():.1f}")
print(f"Para comparação:")
print(f"  Canal G — min={rgb[:,:,1].min():.1f}  max={rgb[:,:,1].max():.1f}")
print(f"  Canal B — min={rgb[:,:,2].min():.1f}  max={rgb[:,:,2].max():.1f}")

resultados = []
for lim in LIMITES:
    R_min, R_max, gray_img, sup = normalizar_por_R(rgb, lim)
    polys, area, pct = vetorizar(sup)
    resultados.append({
        'lim': lim, 'R_min': R_min, 'R_max': R_max,
        'gray_img': gray_img, 'sup': sup,
        'polys': polys, 'area': area, 'pct': pct
    })
    print(f"  [L={lim}]  R: {R_min:.0f}→0  {R_max:.0f}→{lim}  "
          f"→  px≥{THRESHOLD}={pct:.1f}%  polys={len(polys)}  área={area:.0f}px²")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA PRINCIPAL — 5 linhas × 4 colunas
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(5, 4, figsize=(22, 27))

fig.suptitle(
    f'Normalização pelo Canal R → replicado para G e B\n'
    f'IMG_20260602 (bancada / alicates)  |  R: min={R.min():.0f} → 0   '
    f'max={R.max():.0f} → limite   (regra de três)\n'
    f'Sup = R_novo / limite  ∈ [0,1]   |   threshold fixo = {THRESHOLD}',
    fontsize=12, fontweight='bold', y=1.005)

# ── rótulos de coluna ──
col_titles = ['Original (RGB)', 'L = 31', 'L = 127', 'L = 255']
for c, t in enumerate(col_titles):
    axes[0, c].set_title(t, fontsize=11, fontweight='bold', pad=8)

# ── rótulos de linha ──
row_labels = [
    'Imagem\noriginal',
    'Tons de cinza\n(R→G→B)',
    'Superfície\n[0,1]',
    'Histograma\n+ threshold',
    'Polígonos\nvetorizados',
]
for r, rl in enumerate(row_labels):
    axes[r, 0].set_ylabel(rl, fontsize=10, fontweight='bold',
                          rotation=0, labelpad=72, va='center')

# ── Linha 0: imagem original em todas as colunas ──
for c in range(4):
    axes[0, c].imshow(img_orig_arr)
    axes[0, c].axis('off')
# anotação sobre o canal R no original
axes[0, 0].text(
    0.5, -0.07,
    f'Canal R original: min={R.min():.0f}  max={R.max():.0f}  '
    f'mean={R.mean():.0f}  range={R.max()-R.min():.0f}',
    transform=axes[0, 0].transAxes, ha='left', fontsize=8,
    color='darkred', style='italic')

# ── Linha 1: tons de cinza (R→G→B) ──
# col 0: canal R isolado (mapa de calor)
im_r = axes[1, 0].imshow(R, cmap='Reds_r', vmin=0, vmax=255)
axes[1, 0].axis('off')
axes[1, 0].set_title(f'Canal R bruto\nmin={R.min():.0f}  max={R.max():.0f}', fontsize=8, pad=3)
plt.colorbar(im_r, ax=axes[1, 0], fraction=0.046, pad=0.04)

for col, r in enumerate(resultados, start=1):
    axes[1, col].imshow(r['gray_img'])
    axes[1, col].axis('off')
    axes[1, col].set_title(
        f'RGB({r["R_min"]:.0f},{r["R_min"]:.0f},{r["R_min"]:.0f})→(0,0,0)\n'
        f'RGB({r["R_max"]:.0f},{r["R_max"]:.0f},{r["R_max"]:.0f})→({r["lim"]},{r["lim"]},{r["lim"]})',
        fontsize=8, pad=3)

# ── Linha 2: superfície [0,1] ──
# col 0: canal R normalizado 0-255 (sem clamp de limite)
sup_ref = ((R - R.min()) / (R.max() - R.min())).astype(np.float32)
im_s0 = axes[2, 0].imshow(sup_ref, cmap='plasma', vmin=0, vmax=1)
axes[2, 0].axis('off')
axes[2, 0].set_title('Sup R normalizado\n(sem limite, L=255)', fontsize=8, pad=3)
plt.colorbar(im_s0, ax=axes[2, 0], fraction=0.046, pad=0.04)

for col, r in enumerate(resultados, start=1):
    im_s = axes[2, col].imshow(r['sup'], cmap='plasma', vmin=0, vmax=1)
    # isocontorno do threshold
    try:
        cs = axes[2, col].contour(r['sup'], levels=[THRESHOLD],
                                  colors=[CORES_LIM[col-1]], linewidths=2)
        axes[2, col].clabel(cs, fmt=f'{THRESHOLD}', fontsize=8,
                            colors=[CORES_LIM[col-1]])
    except Exception:
        pass
    axes[2, col].axis('off')
    axes[2, col].set_title(
        f'Sup / {r["lim"]}  ∈ [0,1]\n'
        f'{r["pct"]:.1f}% pixels ≥ {THRESHOLD}',
        fontsize=8, pad=3)
    plt.colorbar(im_s, ax=axes[2, col], fraction=0.046, pad=0.04)

# ── Linha 3: histogramas ──
# col 0: histograma do R bruto
axes[3, 0].hist(R.ravel(), bins=64, color='tomato', alpha=0.75)
axes[3, 0].axvline(R.min(), color='navy', ls='--', lw=1.5,
                   label=f'min={R.min():.0f}')
axes[3, 0].axvline(R.max(), color='green', ls='--', lw=1.5,
                   label=f'max={R.max():.0f}')
axes[3, 0].set_title('Histograma R bruto', fontsize=8, pad=3)
axes[3, 0].legend(fontsize=7); axes[3, 0].set_xlim(0, 255)

for col, r in enumerate(resultados, start=1):
    vals = r['sup'].ravel() * r['lim']
    ax = axes[3, col]
    ax.hist(vals, bins=64, color=CORES_LIM[col-1], alpha=0.75)
    thr_val = THRESHOLD * r['lim']
    ax.axvline(thr_val, color='black', ls=':', lw=2,
               label=f'thr={thr_val:.0f} ({r["pct"]:.0f}% acima)')
    ax.set_title(
        f'Histograma R_novo [0-{r["lim"]}]\n'
        f'thr={thr_val:.0f} → {r["pct"]:.1f}% ativos',
        fontsize=8, pad=3)
    ax.legend(fontsize=7); ax.set_xlim(0, r['lim'])

# ── Linha 4: polígonos vetorizados ──
axes[4, 0].imshow(img_orig_arr); axes[4, 0].axis('off')
axes[4, 0].set_title('Referência', fontsize=8, pad=3)

for col, r in enumerate(resultados, start=1):
    ax = axes[4, col]
    draw_polys(ax, img_orig_arr, r['polys'], CORES_LIM[col-1])
    verd = ('OK' if 5 <= r['pct'] <= 40 else
            ('ALTO' if r['pct'] <= 60 else 'SATURADO'))
    ax.set_title(
        f'[L={r["lim"]}]  {len(r["polys"])}p  '
        f'cob={r["pct"]:.1f}% [{verd}]  area={r["area"]:.0f}px²',
        fontsize=9, fontweight='bold',
        color='darkgreen' if verd == 'OK' else 'darkorange' if verd == 'ALTO' else 'red',
        pad=4)

plt.tight_layout(rect=[0, 0, 1, 1.0])
out = '/tmp/normalizacao_canal_r_alicates.png'
fig.savefig(out, dpi=120, bbox_inches='tight')
plt.close(fig)

import os
print(f'\nFigura salva: {out}  ({os.path.getsize(out)//1024} KB)')
print('\n=== RESUMO ===')
for r in resultados:
    status = ('OK' if 5 <= r['pct'] <= 40 else
              ('ALTO' if r['pct'] <= 60 else 'SATURADO'))
    print(f"  L={r['lim']:3d}: R {r['R_min']:.0f}→0  {r['R_max']:.0f}→{r['lim']}  "
          f"| {len(r['polys'])}p  cob={r['pct']:.1f}%  [{status}]")
