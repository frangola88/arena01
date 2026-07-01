import sys, json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from shapely.geometry import shape
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, '/home/cuco/projetos/casaiq')
from core.vetorizador_raster import vetorizar_superficie

IMG_PATH = "/home/cuco/Downloads/IMG_20260602_185719031.jpg"
LIM = 127

img = np.array(Image.open(IMG_PATH).convert("RGB")).astype(np.float64)
pc_median = np.median(img, axis=2)
pc_raiz = np.sqrt(pc_median)
tmin, tmax = pc_raiz.min(), pc_raiz.max()
mediana127 = np.round((pc_raiz - tmin) / (tmax - tmin) * LIM).astype(np.int32)

hist, edges = np.histogram(mediana127.ravel(), bins=LIM + 1, range=(0, LIM))
hist_smooth = gaussian_filter1d(hist.astype(float), sigma=2)

peaks, _ = find_peaks(hist_smooth, prominence=hist_smooth.max() * 0.12, distance=20)
print(f"picos: {peaks.tolist()}")

# ── FWHM de cada pico: largura onde o histograma cai à metade da altura ────
def fwhm_bounds(hist_s, peak):
    half = hist_s[peak] / 2.0
    lo = peak
    while lo > 0 and hist_s[lo] > half:
        lo -= 1
    hi = peak
    while hi < len(hist_s) - 1 and hist_s[hi] > half:
        hi += 1
    return lo, hi

bandas = []
for p in peaks:
    lo, hi = fwhm_bounds(hist_smooth, p)
    bandas.append((p, lo, hi))
    print(f"pico={p}  faixa FWHM=[{lo},{hi}]")

# ── Mapa de classes: 0=transição, 1=pico esquerdo travado, 2=pico direito travado
classes = np.zeros_like(mediana127, dtype=np.uint8)
nomes = ["transição (indefinido)"]
for i, (p, lo, hi) in enumerate(bandas):
    classes[(mediana127 >= lo) & (mediana127 <= hi)] = i + 1
    nomes.append(f"pico {p} (faixa [{lo},{hi}])")

cores = ["#888888", "#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
cmap_classes = ListedColormap(cores[:len(bandas) + 1])

# ── Contornos por classe travada (usando a técnica já estabelecida) ────────
resultados = []
for i, (p, lo, hi) in enumerate(bandas):
    mask = ((mediana127 >= lo) & (mediana127 <= hi)).astype(np.float32)
    res = vetorizar_superficie(mask, threshold=0.5)
    feats = json.loads(res['geojson'])['features']
    print(f"pico={p}  polígonos={len(feats)}  cobertura={res['stats']['coverage_pct']}%")
    resultados.append((p, lo, hi, feats))

# ── Figura ───────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(2, 2)

ax_hist = fig.add_subplot(gs[0, 0])
ax_hist.bar(range(LIM + 1), hist, color='lightgray', edgecolor='gray', width=1.0)
ax_hist.plot(hist_smooth, color='black', linewidth=1.5)
for i, (p, lo, hi) in enumerate(bandas):
    ax_hist.axvline(p, color=cores[i + 1], linestyle='--', linewidth=1.5)
    ax_hist.axvspan(lo, hi, color=cores[i + 1], alpha=0.25, label=f'pico {p} (FWHM)')
ax_hist.set_title("Histograma + faixas FWHM travadas em cada pico")
ax_hist.set_xlabel("nível")
ax_hist.legend()

ax_orig = fig.add_subplot(gs[0, 1])
ax_orig.imshow(mediana127, cmap="gray", vmin=0, vmax=LIM)
ax_orig.set_title("Mediana(raiz) [0,127] — original")
ax_orig.axis("off")

ax_classes = fig.add_subplot(gs[1, 0])
ax_classes.imshow(classes, cmap=cmap_classes, vmin=0, vmax=len(bandas))
ax_classes.set_title("Mapa de classes: cinza=transição, cores=picos travados")
ax_classes.axis("off")

ax_cont = fig.add_subplot(gs[1, 1])
ax_cont.imshow(mediana127, cmap="gray", vmin=0, vmax=LIM)
for i, (p, lo, hi, feats) in enumerate(resultados):
    color = cores[i + 1]
    for feat in feats:
        geom = shape(feat['geometry'])
        if geom.is_empty:
            continue
        xs, ys = geom.exterior.xy
        ax_cont.plot(xs, ys, color=color, linewidth=1.5)
n_total = sum(len(f) for _, _, _, f in resultados)
ax_cont.set_title(f"Contornos das faixas travadas ({n_total} polígono(s) no total)")
ax_cont.axis("off")

plt.tight_layout()
out_path = "/home/cuco/projetos/casaiq/picos_chapados_fwhm.png"
plt.savefig(out_path, dpi=150)
print("salvo em", out_path)
