"""
Compara vetorização via canal alpha (u2net) vs cinza [0-31].

Para cada imagem:
  Col 0: Original
  Col 1: Alpha mask (u2net)
  Col 2: Superficie alpha [0,1]
  Col 3: Polígonos alpha
  Col 4: Polígonos cinza [0-31] (referência)
"""
import sys, time, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from shapely.geometry import shape
import onnxruntime as ort

sys.path.insert(0, '/home/cuco/projetos/casaiq')
from core.vetorizador_raster import vetorizar_superficie

ONNX_PATH  = '/home/cuco/.u2net/u2net.onnx'
GRID       = 280
THRESHOLD  = 0.5

IMAGENS = [
    ('Catálogo\n(fundo branco)',
     '/home/cuco/dataset_imgs/1a016cd1da7638554b687ec2d9c9438392020afee437b774159a58ff1b9185d0.jpg'),
    ('Foto real\n(bancada)',
     '/home/cuco/Downloads/IMG_20260602_185719031.jpg'),
]

# ── U2Net inference ──────────────────────────────────────────────────────────
sess = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
IN_NAME  = sess.get_inputs()[0].name
OUT_NAME = sess.get_outputs()[0].name

def u2net_alpha(img_pil: Image.Image) -> np.ndarray:
    """Retorna máscara alpha [0,1] no tamanho original da imagem."""
    inp = img_pil.resize((320, 320), Image.LANCZOS).convert('RGB')
    x = np.array(inp, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    x = x.transpose(2, 0, 1)[None]          # (1,3,320,320)
    pred = sess.run([OUT_NAME], {IN_NAME: x})[0][0, 0]  # (320,320)
    # normaliza para [0,1]
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    # redimensiona para tamanho original
    alpha = Image.fromarray((pred * 255).astype(np.uint8)).resize(
        img_pil.size, Image.LANCZOS)
    return np.array(alpha, dtype=np.float32) / 255.0

def gray_inv31(img_pil: Image.Image) -> np.ndarray:
    """Superficie cinza invertido [0-31], redimensionada para GRID."""
    img = img_pil.resize((GRID, GRID), Image.LANCZOS)
    rgb = np.array(img, dtype=np.float32)
    gray = 0.299*rgb[:,:,0] + 0.587*rgb[:,:,1] + 0.114*rgb[:,:,2]
    return (np.clip(255.0 - gray, 0, 31) / 31).astype(np.float32)

def vetorizar(sup: np.ndarray):
    t0 = time.time()
    res = vetorizar_superficie(sup)
    dt  = time.time() - t0
    feats = json.loads(res['geojson'])['features']
    return feats, dt

def draw_polys(ax, feats, img, grid):
    ax.imshow(img.resize((grid, grid), Image.LANCZOS))
    cmap = plt.cm.tab10
    for i, feat in enumerate(feats):
        try:
            geom = shape(feat['geometry'])
            if geom.is_empty: continue
            color = cmap(i % 10)
            xs, ys = geom.exterior.xy
            ax.fill(xs, ys, alpha=0.30, color=color)
            ax.plot(xs, ys, color=color, linewidth=1.2)
            cx, cy = geom.centroid.x, geom.centroid.y
            ax.plot(cx, cy, '+', color=color, ms=8, mew=2)
        except Exception:
            pass
    ax.set_xlim(0, grid); ax.set_ylim(grid, 0); ax.axis('off')

# ── Figura: 2 linhas (imagens) × 5 colunas ──────────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(26, 12))
fig.suptitle(
    'Alpha U2Net vs Cinza [0-31]  —  Canal alpha = entrada robusta p/ vetorização',
    fontsize=13, fontweight='bold', y=0.99)

for col, t in enumerate(['Original', 'Alpha U2Net (320→orig)', 'Sup alpha [0,1]',
                          'Polígonos — ALPHA', 'Polígonos — cinza [0-31]']):
    axes[0, col].set_title(t, fontsize=10, fontweight='bold', pad=8)

for row, (label, path) in enumerate(IMAGENS):
    print(f'\n=== {label.strip()} ===')
    img = Image.open(path).convert('RGB')

    # ── Alpha U2Net ──
    print('  rodando u2net...', end=' ', flush=True)
    t0 = time.time()
    alpha_full = u2net_alpha(img)          # shape = img.size (W,H) ordem PIL
    print(f'{(time.time()-t0)*1000:.0f}ms')

    # redimensiona alpha para GRID×GRID
    alpha_grid = np.array(
        Image.fromarray((alpha_full * 255).astype(np.uint8)).resize(
            (GRID, GRID), Image.LANCZOS),
        dtype=np.float32) / 255.0

    feats_alpha, dt_alpha = vetorizar(alpha_grid)
    area_alpha = sum(shape(f['geometry']).area for f in feats_alpha
                     if not shape(f['geometry']).is_empty)
    pct_alpha  = 100 * np.sum(alpha_grid >= THRESHOLD) / (GRID*GRID)
    print(f'  alpha: px≥thr={pct_alpha:.1f}%  polys={len(feats_alpha)}  '
          f'area={area_alpha:.0f}px²  {dt_alpha*1000:.0f}ms')

    # ── Cinza [0-31] ──
    sup31 = gray_inv31(img)
    feats31, dt31 = vetorizar(sup31)
    area31 = sum(shape(f['geometry']).area for f in feats31
                 if not shape(f['geometry']).is_empty)
    pct31  = 100 * np.sum(sup31 >= THRESHOLD) / (GRID*GRID)
    print(f'  [0-31]: px≥thr={pct31:.1f}%  polys={len(feats31)}  '
          f'area={area31:.0f}px²  {dt31*1000:.0f}ms')

    # Col 0 — original
    axes[row, 0].imshow(img)
    axes[row, 0].set_ylabel(label, fontsize=11, fontweight='bold',
                            rotation=0, labelpad=70, va='center')
    axes[row, 0].axis('off')

    # Col 1 — alpha mask (tamanho original)
    axes[row, 1].imshow(alpha_full, cmap='RdYlGn', vmin=0, vmax=1)
    axes[row, 1].set_title(f'mean={alpha_full.mean():.3f}  '
                           f'std={alpha_full.std():.3f}', fontsize=8, pad=3)
    axes[row, 1].axis('off')
    plt.colorbar(axes[row,1].images[0], ax=axes[row,1], fraction=0.046, pad=0.04)

    # Col 2 — superficie alpha no grid
    im2 = axes[row, 2].imshow(alpha_grid, cmap='plasma', vmin=0, vmax=1)
    try:
        cs = axes[row, 2].contour(alpha_grid, levels=[THRESHOLD],
                                  colors=['cyan'], linewidths=2)
        axes[row, 2].clabel(cs, fmt=f'{THRESHOLD}', fontsize=8, colors='cyan')
    except Exception:
        pass
    axes[row, 2].set_title(f'{pct_alpha:.1f}% ≥ {THRESHOLD}', fontsize=9, pad=3)
    plt.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)
    axes[row, 2].axis('off')

    # Col 3 — polígonos alpha
    draw_polys(axes[row, 3], feats_alpha, img, GRID)
    axes[row, 3].set_title(
        f'{len(feats_alpha)} polígono(s)  área={area_alpha:.0f}px²\n'
        f'{dt_alpha*1000:.0f}ms',
        fontsize=9, color='darkgreen' if feats_alpha else 'red', pad=3)

    # Col 4 — polígonos cinza [0-31]
    draw_polys(axes[row, 4], feats31, img, GRID)
    axes[row, 4].set_title(
        f'{len(feats31)} polígono(s)  área={area31:.0f}px²\n'
        f'{dt31*1000:.0f}ms  (cobertura={pct31:.1f}%)',
        fontsize=9, color='darkblue', pad=3)

plt.tight_layout(rect=[0, 0, 1, 0.97])
out = '/tmp/alpha_vs_cinza.png'
plt.savefig(out, dpi=130, bbox_inches='tight')
plt.close()
print(f'\nSalvo: {out}')
