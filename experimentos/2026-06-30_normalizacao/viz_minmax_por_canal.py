"""
Normalização min-max POR CANAL independente, depois conversão para cinza.

Para cada limite L em [31, 127, 255]:
  R' = (R - R.min) / (R.max - R.min) * L
  G' = (G - G.min) / (G.max - G.min) * L
  B' = (B - B.min) / (B.max - B.min) * L
  gray = 0.299·R' + 0.587·G' + 0.114·B'   → [0, L]
  sup  = gray / L                            → [0, 1] para o vetorizador

Isso expande o contraste de cada canal ao máximo possível de forma independente,
eliminando cast de cor e aproveitando todo o range disponível em R, G e B.
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
CORES     = ['#e84393', '#4488ff', '#22cc88']


# ── helpers ─────────────────────────────────────────────────────────────────

def normalizar_canal(canal: np.ndarray, limite: int) -> np.ndarray:
    """Regra de três: menor→0, maior→limite, resto interpolado linearmente."""
    lo, hi = float(canal.min()), float(canal.max())
    if hi == lo:
        return np.zeros_like(canal, dtype=np.float32)
    return ((canal.astype(np.float32) - lo) / (hi - lo) * limite)


def minmax_por_canal(img_pil: Image.Image, limite: int, grid: int = GRID) -> dict:
    """
    Aplica normalização min-max em R, G, B separadamente,
    depois combina em luminância → [0,1].
    Retorna dicionário com os arrays intermediários e a superfície final.
    """
    img  = img_pil.resize((grid, grid), Image.LANCZOS).convert('RGB')
    rgb  = np.array(img, dtype=np.float32)
    R, G, B = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]

    R2 = normalizar_canal(R, limite)
    G2 = normalizar_canal(G, limite)
    B2 = normalizar_canal(B, limite)

    gray = 0.299 * R2 + 0.587 * G2 + 0.114 * B2   # [0, limite]
    sup  = (gray / limite).astype(np.float32)       # [0, 1]

    return {
        'R_orig': R, 'G_orig': G, 'B_orig': B,
        'R2': R2, 'G2': G2, 'B2': B2,
        'gray': gray, 'sup': sup,
        'R_stats': (R.min(), R.max()),
        'G_stats': (G.min(), G.max()),
        'B_stats': (B.min(), B.max()),
    }


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
    area = sum(g.area for g in polys)
    pct  = 100.0 * np.sum(sup >= THRESHOLD) / (GRID * GRID)
    return polys, area, pct


def draw_polys(ax, img_arr, polys, color):
    ax.imshow(img_arr)
    for geom in polys:
        xs, ys = geom.exterior.xy
        ax.fill(xs, ys, alpha=0.28, color=color)
        ax.plot(xs, ys, color=color, linewidth=1.8)
        ax.plot(geom.centroid.x, geom.centroid.y, '+', color=color, ms=9, mew=2)
    ax.set_xlim(0, GRID); ax.set_ylim(GRID, 0); ax.axis('off')


# ── carga e processamento ────────────────────────────────────────────────────

img_orig = Image.open(IMG_PATH).convert('RGB')
img_arr  = np.array(img_orig.resize((GRID, GRID), Image.LANCZOS))
R0 = img_arr[:,:,0].astype(np.float32)
G0 = img_arr[:,:,1].astype(np.float32)
B0 = img_arr[:,:,2].astype(np.float32)

print("=== Estatísticas dos canais ORIGINAIS ===")
print(f"  R: min={R0.min():.0f}  max={R0.max():.0f}  range={R0.max()-R0.min():.0f}")
print(f"  G: min={G0.min():.0f}  max={G0.max():.0f}  range={G0.max()-G0.min():.0f}")
print(f"  B: min={B0.min():.0f}  max={B0.max():.0f}  range={B0.max()-B0.min():.0f}")

resultados = []
for lim, cor in zip(LIMITES, CORES):
    d = minmax_por_canal(img_orig, lim)
    polys, area, pct = vetorizar(d['sup'])
    d.update({'lim': lim, 'cor': cor, 'polys': polys, 'area': area, 'pct': pct})
    resultados.append(d)
    print(f"\n  [L={lim}]")
    print(f"    R: [{d['R_stats'][0]:.0f}→0, {d['R_stats'][1]:.0f}→{lim}]")
    print(f"    G: [{d['G_stats'][0]:.0f}→0, {d['G_stats'][1]:.0f}→{lim}]")
    print(f"    B: [{d['B_stats'][0]:.0f}→0, {d['B_stats'][1]:.0f}→{lim}]")
    print(f"    gray: min={d['gray'].min():.1f}  max={d['gray'].max():.1f}")
    print(f"    sup:  px≥{THRESHOLD}={pct:.1f}%  polys={len(polys)}  área={area:.0f}px²")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA PRINCIPAL — 6 linhas × 4 colunas
# ══════════════════════════════════════════════════════════════════════════════
#
# Linha 0: Original | R norm (L=31) | R norm (L=127) | R norm (L=255)
# Linha 1: —        | G norm        | G norm          | G norm
# Linha 2: —        | B norm        | B norm          | B norm
# Linha 3: —        | RGB remont.   | RGB remont.     | RGB remont.
# Linha 4: —        | Superfície[0,1] | ...            | ...
# Linha 5: —        | Polígonos     | Polígonos        | Polígonos

print("\nGerando figura...")

ROWS, COLS = 6, 4
fig, axes = plt.subplots(ROWS, COLS, figsize=(22, 30))
fig.suptitle(
    'Normalização min-max POR CANAL (R, G, B independentes)\n'
    r'$R\prime = (R - R_{min}) / (R_{max} - R_{min}) \times L$  '
    r'  $G\prime = (G - G_{min}) / (G_{max} - G_{min}) \times L$  '
    r'  $B\prime = (B - B_{min}) / (B_{max} - B_{min}) \times L$',
    fontsize=13, fontweight='bold', y=1.002)

col_titles = ['Original', 'L = 31', 'L = 127', 'L = 255']
for c, t in enumerate(col_titles):
    axes[0, c].set_title(t, fontsize=12, fontweight='bold', pad=8)

row_labels = [
    f"Canal R\n(orig {R0.min():.0f}–{R0.max():.0f})",
    f"Canal G\n(orig {G0.min():.0f}–{G0.max():.0f})",
    f"Canal B\n(orig {B0.min():.0f}–{B0.max():.0f})",
    "RGB recombinado\n(R'+G'+B' norm.)",
    "Superfície cinza\n[0,1] → vetorizador",
    "Polígonos\nsobre imagem",
]
for r, rl in enumerate(row_labels):
    axes[r, 0].set_ylabel(rl, fontsize=10, fontweight='bold',
                           rotation=0, labelpad=80, va='center')

# Col 0: canais originais
axes[0, 0].imshow(R0, cmap='Reds',   vmin=0, vmax=255); axes[0, 0].axis('off')
axes[1, 0].imshow(G0, cmap='Greens', vmin=0, vmax=255); axes[1, 0].axis('off')
axes[2, 0].imshow(B0, cmap='Blues',  vmin=0, vmax=255); axes[2, 0].axis('off')
axes[3, 0].imshow(img_arr);                              axes[3, 0].axis('off')
# Superfície original (luminância simples sem normalização)
L_orig = (0.299*R0 + 0.587*G0 + 0.114*B0) / 255.0
im_l = axes[4, 0].imshow(L_orig, cmap='gray', vmin=0, vmax=1)
axes[4, 0].axis('off')
axes[4, 0].set_title(f'Luminância sem norm.\nmean={L_orig.mean():.2f}', fontsize=8, pad=3)
plt.colorbar(im_l, ax=axes[4, 0], fraction=0.046, pad=0.04)
axes[5, 0].imshow(img_arr); axes[5, 0].axis('off')

# Colunas 1-3: cada limite
for col, d in enumerate(resultados, start=1):
    lim = d['lim']
    cor = d['cor']

    # L0: canal R normalizado
    im = axes[0, col].imshow(d['R2'], cmap='Reds', vmin=0, vmax=lim)
    axes[0, col].axis('off')
    axes[0, col].set_title(
        f"R: {d['R_stats'][0]:.0f}→0  {d['R_stats'][1]:.0f}→{lim}\n"
        f"mean={d['R2'].mean():.1f}  std={d['R2'].std():.1f}", fontsize=8, pad=3)
    plt.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)

    # L1: canal G normalizado
    im = axes[1, col].imshow(d['G2'], cmap='Greens', vmin=0, vmax=lim)
    axes[1, col].axis('off')
    axes[1, col].set_title(
        f"G: {d['G_stats'][0]:.0f}→0  {d['G_stats'][1]:.0f}→{lim}\n"
        f"mean={d['G2'].mean():.1f}  std={d['G2'].std():.1f}", fontsize=8, pad=3)
    plt.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)

    # L2: canal B normalizado
    im = axes[2, col].imshow(d['B2'], cmap='Blues', vmin=0, vmax=lim)
    axes[2, col].axis('off')
    axes[2, col].set_title(
        f"B: {d['B_stats'][0]:.0f}→0  {d['B_stats'][1]:.0f}→{lim}\n"
        f"mean={d['B2'].mean():.1f}  std={d['B2'].std():.1f}", fontsize=8, pad=3)
    plt.colorbar(im, ax=axes[2, col], fraction=0.046, pad=0.04)

    # L3: RGB recombinado
    rgb_norm = np.stack([
        np.clip(d['R2'] / lim, 0, 1),
        np.clip(d['G2'] / lim, 0, 1),
        np.clip(d['B2'] / lim, 0, 1),
    ], axis=2)
    axes[3, col].imshow(rgb_norm)
    axes[3, col].axis('off')
    axes[3, col].set_title(f"RGB(R',G',B') recombinado [L={lim}]", fontsize=8, pad=3)

    # L4: superfície cinza
    im = axes[4, col].imshow(d['sup'], cmap='gray', vmin=0, vmax=1)
    axes[4, col].axis('off')
    # isocontorno no threshold
    try:
        cs = axes[4, col].contour(d['sup'], levels=[THRESHOLD],
                                   colors=[cor], linewidths=2)
        axes[4, col].clabel(cs, fmt=f'{THRESHOLD}', fontsize=8, colors=[cor])
    except Exception:
        pass
    axes[4, col].set_title(
        f"gray/L → [0,1]  (thr={THRESHOLD})\n"
        f"mean={d['sup'].mean():.3f}  {d['pct']:.1f}% ativos",
        fontsize=8, pad=3)
    plt.colorbar(im, ax=axes[4, col], fraction=0.046, pad=0.04)

    # L5: polígonos
    draw_polys(axes[5, col], img_arr, d['polys'], cor)
    verd = 'OK' if 5 <= d['pct'] <= 40 else ('COB ALTA' if d['pct'] > 40 else 'COB BAIXA')
    axes[5, col].set_title(
        f"[{verd}] {len(d['polys'])}p  cob={d['pct']:.1f}%  área={d['area']:.0f}px²",
        fontsize=9, fontweight='bold',
        color='darkgreen' if verd == 'OK' else 'red')

plt.tight_layout(rect=[0, 0, 1, 0.999])
out = '/tmp/minmax_por_canal_alicates.png'
fig.savefig(out, dpi=110, bbox_inches='tight')
plt.close(fig)

import os
print(f"\nSalvo: {out}  ({os.path.getsize(out)//1024} KB)")
print("Done!")
