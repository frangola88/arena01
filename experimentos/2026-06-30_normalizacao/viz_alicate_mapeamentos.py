"""Compara mapeamentos cinza invertido [0-127], [0-63], [0-31] no vetorizador.

Inversao: fundo branco (gray=255) → 0, objeto escuro → alto.
sup = clip(255 - gray, 0, max_val) / max_val
"""
import sys, time, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
from shapely.geometry import shape

sys.path.insert(0, '/home/cuco/projetos/casaiq')
from core.vetorizador_raster import vetorizar_superficie

IMG_PATH = '/home/cuco/Downloads/IMG_20260602_185719031.jpg'
GRID = 280
THRESHOLD = 0.5
LEVELS = [0.25, 0.50, 0.75]   # isocontornos extras

img_orig = Image.open(IMG_PATH).convert('RGB')
img_g    = img_orig.resize((GRID, GRID), Image.LANCZOS)
rgb      = np.array(img_g, dtype=np.float32)
gray255  = 0.299*rgb[:,:,0] + 0.587*rgb[:,:,1] + 0.114*rgb[:,:,2]
inv255   = 255.0 - gray255     # inverso: escuro→alto

MAPS = [('0-127', 127), ('0-63', 63), ('0-31', 31)]

# ── Figura principal: 3 linhas × 5 colunas ──────────────────────────────────
fig, axes = plt.subplots(3, 5, figsize=(26, 17))
fig.suptitle(
    'IMG_20260602 — Mapeamento INVERTIDO  (objeto escuro → valor alto)\n'
    'Linhas: clip a [0-127] / [0-63] / [0-31]   |   threshold fixo = 0.5',
    fontsize=13, fontweight='bold', y=0.99)

COL_TITLES = ['Original', 'Sup mapeada', 'Isocontornos (0.25/0.50/0.75)',
              f'Polígonos (thr={THRESHOLD})', 'Comparação de áreas']
for col, t in enumerate(COL_TITLES):
    axes[0, col].set_title(t, fontsize=10, fontweight='bold', pad=8)

# Col 4: gráfico de barras comparando cobertura — montar depois
cob_data   = {}
area_data  = {}

for row, (label, max_val) in enumerate(MAPS):
    sup = (np.clip(inv255, 0, max_val) / max_val).astype(np.float32)

    t0 = time.time()
    res = vetorizar_superficie(sup)
    dt  = time.time() - t0
    feats  = json.loads(res['geojson'])['features']
    n_poly = len(feats)
    n_acima = int(np.sum(sup >= THRESHOLD))
    pct     = 100 * n_acima / (GRID * GRID)

    # área total dos polígonos (em pixels²)
    area_total = sum(shape(f['geometry']).area for f in feats
                     if not shape(f['geometry']).is_empty)

    cob_data[label]  = pct
    area_data[label] = area_total
    print(f'[{label}]  px≥{THRESHOLD}={n_acima:,} ({pct:.1f}%)  '
          f'polys={n_poly}  area={area_total:.0f}px²  {dt*1000:.0f}ms')

    # ── Col 0: original ──
    axes[row, 0].imshow(img_orig)
    axes[row, 0].set_ylabel(f'clip\n[{label}]', fontsize=12, fontweight='bold',
                            rotation=0, labelpad=62, va='center')
    axes[row, 0].axis('off')

    # ── Col 1: superficie mapeada ──
    im1 = axes[row, 1].imshow(sup, cmap='inferno', vmin=0, vmax=1)
    axes[row, 1].set_title(
        f'(255-gray) clip [0,{max_val}] / {max_val}\n'
        f'mean={sup.mean():.3f}  std={sup.std():.3f}', fontsize=8, pad=3)
    plt.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)
    axes[row, 1].axis('off')

    # ── Col 2: isocontornos 0.25 / 0.50 / 0.75 ──
    axes[row, 2].imshow(sup, cmap='gray', vmin=0, vmax=1, alpha=0.6)
    colors_iso = ['#00ff88', '#ff4400', '#4488ff']
    for lvl, col_iso in zip(LEVELS, colors_iso):
        try:
            cs = axes[row, 2].contour(sup, levels=[lvl], colors=[col_iso], linewidths=1.8)
            axes[row, 2].clabel(cs, fmt=f'{lvl:.2f}', fontsize=7, colors=col_iso)
        except Exception:
            pass
    axes[row, 2].axis('off')

    # ── Col 3: polígonos sobre imagem ──
    axes[row, 3].imshow(img_g)
    cmap_c = plt.cm.Set1
    for i, feat in enumerate(feats):
        try:
            geom = shape(feat['geometry'])
            if geom.is_empty:
                continue
            color = cmap_c(i % 9)
            xs, ys = geom.exterior.xy
            axes[row, 3].fill(xs, ys, alpha=0.30, color=color)
            axes[row, 3].plot(xs, ys, color=color, linewidth=1.2)
            # centroide
            cx, cy = geom.centroid.x, geom.centroid.y
            axes[row, 3].plot(cx, cy, '+', color=color, ms=8, mew=2)
        except Exception:
            pass
    axes[row, 3].set_title(
        f'{n_poly} polígono(s)   área={area_total:.0f}px²\n'
        f'cobertura={pct:.1f}%   {dt*1000:.0f}ms',
        fontsize=9, color='darkgreen' if n_poly > 0 else 'red', pad=3)
    axes[row, 3].set_xlim(0, GRID)
    axes[row, 3].set_ylim(GRID, 0)
    axes[row, 3].axis('off')

# ── Col 4: barras comparativas (preenchemos após o loop) ──────────────────
for row, (label, _) in enumerate(MAPS):
    ax = axes[row, 4]
    ax.axis('off')

# Substituímos os 3 axes[*, 4] por um único subplot compartilhado
for ax in axes[:, 4]:
    ax.set_visible(False)

# Cria eixo novo ocupando toda a 5ª coluna
left   = axes[0, 4].get_position().x0
bottom = axes[2, 4].get_position().y0
width  = axes[0, 4].get_position().width
height = axes[0, 4].get_position().y1 - bottom
ax_bar = fig.add_axes([left, bottom, width, height])

labels  = [m[0] for m in MAPS]
covs    = [cob_data[l] for l in labels]
areas   = [area_data[l] for l in labels]
x       = np.arange(len(labels))
w       = 0.35
bars1   = ax_bar.bar(x - w/2, covs,  w, label='Cobertura (%)', color='steelblue')
ax2b    = ax_bar.twinx()
bars2   = ax2b.bar(x + w/2, areas, w, label='Área poly (px²)', color='tomato', alpha=0.85)

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(labels, fontsize=11, fontweight='bold')
ax_bar.set_ylabel('Cobertura (%)', color='steelblue', fontsize=10)
ax_bar.tick_params(axis='y', labelcolor='steelblue')
ax2b.set_ylabel('Área polígonos (px²)', color='tomato', fontsize=10)
ax2b.tick_params(axis='y', labelcolor='tomato')
ax_bar.set_title('Comparação entre mapeamentos', fontsize=10, fontweight='bold', pad=8)

lines1, labs1 = ax_bar.get_legend_handles_labels()
lines2, labs2 = ax2b.get_legend_handles_labels()
ax_bar.legend(lines1+lines2, labs1+labs2, loc='upper right', fontsize=9)

for bar, v in zip(bars1, covs):
    ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{v:.1f}%', ha='center', va='bottom', fontsize=9, color='steelblue')
for bar, v in zip(bars2, areas):
    ax2b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
              f'{v:.0f}', ha='center', va='bottom', fontsize=8, color='tomato')

plt.tight_layout(rect=[0, 0, 1, 0.96])
out = '/tmp/img_downloads_mapeamentos.png'
plt.savefig(out, dpi=130, bbox_inches='tight')
plt.close()
print(f'\nSalvo: {out}')
