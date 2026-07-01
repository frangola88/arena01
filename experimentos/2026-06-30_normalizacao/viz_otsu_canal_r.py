"""
Normalização canal R + threshold de Otsu adaptativo.

Otsu: encontra o threshold T que maximiza a variância ENTRE os dois grupos
(pixels escuros vs claros). Matematicamente:
  σ²_B(T) = w0(T)·w1(T)·[μ0(T) - μ1(T)]²
  w0, w1 = peso de cada grupo  |  μ0, μ1 = média de cada grupo
  T* = argmax σ²_B(T)

Pipeline:
  1. R_novo = (R - R_min) / (R_max - R_min) × limite  → [0, limite]
  2. Otsu calcula T* sobre o histograma de R_novo
  3. sup_otsu[i,j] = 1 se R_novo[i,j] >= T*  else  0  (máscara binária)
  4. Vetorizar sup_otsu

Limites testados: 31, 127, 255
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
LIMITES   = [31, 127, 255]
CORES_LIM = ['#e84393', '#4488ff', '#22cc88']


def otsu_threshold(valores: np.ndarray, n_bins: int = 256) -> tuple[float, float]:
    """
    Calcula o threshold de Otsu sobre um array 1D de valores inteiros.
    Retorna (threshold_absoluto, variancia_maxima).
    """
    hist, bin_edges = np.histogram(valores.ravel(), bins=n_bins)
    hist = hist.astype(np.float64)
    total = hist.sum()
    bins  = (bin_edges[:-1] + bin_edges[1:]) / 2.0   # centros dos bins

    # acumulados
    w0_cum  = np.cumsum(hist) / total          # peso acumulado grupo 0
    mu0_cum = np.cumsum(hist * bins) / (np.cumsum(hist) + 1e-12)

    w1      = 1.0 - w0_cum
    mu_total = (hist * bins).sum() / total
    mu0     = mu0_cum
    mu1     = np.where(w1 > 1e-6,
                       (mu_total - w0_cum * mu0) / w1,
                       0.0)

    sigma2_b = w0_cum * w1 * (mu0 - mu1) ** 2
    idx_max  = np.argmax(sigma2_b)
    return float(bins[idx_max]), float(sigma2_b[idx_max])


def vetorizar_binario(mascara: np.ndarray):
    """Vetoriza uma máscara binária [0/1] float32."""
    res   = vetorizar_superficie(mascara)
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
    pct  = 100.0 * mascara.mean()
    return polys, area, pct


def draw_polys(ax, bg, polys, color):
    ax.imshow(bg)
    for geom in polys:
        xs, ys = geom.exterior.xy
        ax.fill(xs, ys, alpha=0.32, color=color)
        ax.plot(xs, ys, color=color, linewidth=2.0)
        ax.plot(geom.centroid.x, geom.centroid.y,
                '+', color=color, ms=11, mew=2.5)
    ax.set_xlim(0, GRID); ax.set_ylim(GRID, 0); ax.axis('off')


# ── carga ─────────────────────────────────────────────────────────────────────
img = Image.open(IMG_PATH).convert('RGB').resize((GRID, GRID), Image.LANCZOS)
rgb = np.array(img, dtype=np.float32)
img_arr = rgb.clip(0, 255).astype(np.uint8)
R = rgb[:, :, 0]
R_min, R_max = R.min(), R.max()
print(f"Canal R: min={R_min:.0f}  max={R_max:.0f}  range={R_max-R_min:.0f}")

resultados = []
for lim in LIMITES:
    # 1. normalização min-max ao limite
    R_novo = (R - R_min) / (R_max - R_min) * lim        # [0, lim] float

    # 2. Otsu sobre R_novo
    T_otsu, var_max = otsu_threshold(R_novo, n_bins=lim + 1)
    T_rel = T_otsu / lim                                  # threshold relativo [0,1]

    # 3. máscara binária
    mascara = (R_novo >= T_otsu).astype(np.float32)       # 0 ou 1

    # 4. superfície contínua para referência visual ([0,1])
    sup_cont = (R_novo / lim).astype(np.float32)

    # 5. vetorização
    polys, area, pct = vetorizar_binario(mascara)

    resultados.append({
        'lim': lim, 'R_novo': R_novo, 'sup_cont': sup_cont,
        'T_otsu': T_otsu, 'T_rel': T_rel, 'var_max': var_max,
        'mascara': mascara, 'polys': polys, 'area': area, 'pct': pct,
    })
    print(f"  [L={lim:3d}]  T*={T_otsu:.1f}/{lim} ({T_rel*100:.1f}%)  "
          f"σ²_B={var_max:.1f}  |  {len(polys)}p  cob={pct:.1f}%  área={area:.0f}px²")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA — 5 linhas × 4 colunas
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(5, 4, figsize=(22, 27))

fig.suptitle(
    'Threshold de Otsu sobre Canal R normalizado min-max\n'
    f'R: {R_min:.0f}→0  {R_max:.0f}→limite  (regra de três)  |  '
    'T* = argmax variância entre grupos',
    fontsize=12, fontweight='bold', y=1.005)

col_titles = ['Original', 'L = 31', 'L = 127', 'L = 255']
for c, t in enumerate(col_titles):
    axes[0, c].set_title(t, fontsize=11, fontweight='bold', pad=8)

row_labels = [
    'Original',
    'Superfície\nR norm [0,1]',
    'Histograma\n+ Otsu T*',
    'Máscara\nbinária Otsu',
    'Polígonos\nvetorizados',
]
for r, rl in enumerate(row_labels):
    axes[r, 0].set_ylabel(rl, fontsize=10, fontweight='bold',
                          rotation=0, labelpad=72, va='center')

# ── L0: original ──
for c in range(4):
    axes[0, c].imshow(img_arr); axes[0, c].axis('off')

# ── L1: superfície contínua ──
axes[1, 0].imshow(img_arr); axes[1, 0].axis('off')
for col, r in enumerate(resultados, 1):
    im = axes[1, col].imshow(r['sup_cont'], cmap='plasma', vmin=0, vmax=1)
    try:
        cs = axes[1, col].contour(r['sup_cont'], levels=[r['T_rel']],
                                  colors=[CORES_LIM[col-1]], linewidths=2.5)
        axes[1, col].clabel(cs, fmt=f"T*={r['T_rel']:.2f}", fontsize=8,
                            colors=[CORES_LIM[col-1]])
    except Exception:
        pass
    axes[1, col].axis('off')
    axes[1, col].set_title(
        f"Sup [0,1]  |  T*={r['T_rel']:.2f}  ({r['T_otsu']:.1f}/{r['lim']})",
        fontsize=9, pad=3)
    plt.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)

# ── L2: histogramas com T* ──
axes[2, 0].hist(R.ravel(), bins=64, color='gray', alpha=0.7)
axes[2, 0].set_title('Canal R bruto', fontsize=9, pad=3)
axes[2, 0].set_xlim(0, 255)

for col, r in enumerate(resultados, 1):
    ax = axes[2, col]
    vals = r['R_novo'].ravel()
    ax.hist(vals, bins=r['lim'] + 1, color=CORES_LIM[col-1], alpha=0.7)
    ax.axvline(r['T_otsu'], color='black', lw=2.5, ls='--',
               label=f"T*={r['T_otsu']:.1f}")

    # preenche as duas regiões (fundo / objeto)
    ylim = ax.get_ylim()
    ax.axvspan(0,          r['T_otsu'], alpha=0.10, color='navy',  label='fundo')
    ax.axvspan(r['T_otsu'], r['lim'],   alpha=0.10, color='green', label='objeto')

    ax.set_title(
        f"[L={r['lim']}]  T*={r['T_otsu']:.1f}  ({r['T_rel']*100:.0f}%)\n"
        f"σ²_B={r['var_max']:.0f}  →  {r['pct']:.1f}% pixels = objeto",
        fontsize=8, pad=3)
    ax.legend(fontsize=7)
    ax.set_xlim(0, r['lim'])

# ── L3: máscara binária ──
axes[3, 0].imshow(img_arr); axes[3, 0].axis('off')
for col, r in enumerate(resultados, 1):
    axes[3, col].imshow(r['mascara'], cmap='gray', vmin=0, vmax=1)
    axes[3, col].axis('off')
    axes[3, col].set_title(
        f"Binária Otsu  |  branco={r['pct']:.1f}%  preto={100-r['pct']:.1f}%",
        fontsize=9, pad=3)

# ── L4: polígonos ──
axes[4, 0].imshow(img_arr); axes[4, 0].axis('off')
for col, r in enumerate(resultados, 1):
    ax = axes[4, col]
    draw_polys(ax, img_arr, r['polys'], CORES_LIM[col-1])
    status = ('OK' if 5 <= r['pct'] <= 40 else
              ('ALTO' if r['pct'] <= 60 else 'SATURADO'))
    ax.set_title(
        f"[L={r['lim']}]  {len(r['polys'])}p  cob={r['pct']:.1f}% [{status}]  "
        f"área={r['area']:.0f}px²",
        fontsize=9, fontweight='bold',
        color='darkgreen' if status == 'OK' else
              'darkorange' if status == 'ALTO' else 'red',
        pad=4)

plt.tight_layout(rect=[0, 0, 1, 1.0])
out = '/tmp/otsu_canal_r_alicates.png'
fig.savefig(out, dpi=120, bbox_inches='tight')
plt.close(fig)

import os
dest = '/home/cuco/projetos/casaiq/otsu_canal_r_alicates.png'
import shutil; shutil.copy(out, dest)
print(f'\nSalvo: {dest}  ({os.path.getsize(dest)//1024} KB)')
