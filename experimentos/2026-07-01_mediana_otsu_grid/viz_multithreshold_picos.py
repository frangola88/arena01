import sys, json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
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

# ── 1. Histograma suavizado + picos locais ──────────────────────────────────
hist, edges = np.histogram(mediana127.ravel(), bins=LIM + 1, range=(0, LIM))
hist_smooth = gaussian_filter1d(hist.astype(float), sigma=2)

peaks, _ = find_peaks(hist_smooth, prominence=hist_smooth.max() * 0.12, distance=20)
print(f"picos encontrados nos níveis: {peaks.tolist()}")

# ── 2. Vales entre picos consecutivos (mínimo local entre cada par) ────────
fronteiras = [0]
for i in range(len(peaks) - 1):
    a, b = peaks[i], peaks[i + 1]
    vale = a + np.argmin(hist_smooth[a:b + 1])
    fronteiras.append(vale)
fronteiras.append(LIM)
print(f"fronteiras das bandas: {fronteiras}")

# ── 3. Para cada banda [lo, hi), extrai contorno via vetorizar_superficie ──
cores_bandas = plt.cm.tab10
resultados_bandas = []
for i in range(len(fronteiras) - 1):
    lo, hi = fronteiras[i], fronteiras[i + 1]
    mask = ((mediana127 >= lo) & (mediana127 < hi)).astype(np.float32)
    res = vetorizar_superficie(mask, threshold=0.5)
    feats = json.loads(res['geojson'])['features']
    pico = peaks[i] if i < len(peaks) else None
    print(f"banda [{lo},{hi})  pico~{pico}  polígonos={len(feats)}  "
          f"cobertura={res['stats']['coverage_pct']}%")
    resultados_bandas.append((lo, hi, feats))

# ── Figura ───────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(2, len(resultados_bandas) + 1)

ax_hist = fig.add_subplot(gs[0, :])
ax_hist.bar(range(LIM + 1), hist, color='lightgray', edgecolor='gray', width=1.0)
ax_hist.plot(hist_smooth, color='black', linewidth=1.5, label='histograma suavizado')
for p in peaks:
    ax_hist.axvline(p, color='green', linestyle='--', linewidth=1.5)
    ax_hist.text(p, hist_smooth[p], f'pico={p}', color='green', fontsize=9,
                 ha='center', va='bottom')
for f in fronteiras[1:-1]:
    ax_hist.axvline(f, color='red', linewidth=2)
    ax_hist.text(f, ax_hist.get_ylim()[1]*0.9, f'vale={f}', color='red', fontsize=9,
                 ha='center', rotation=90)
ax_hist.set_title("Histograma mediana(raiz) [0,127] — picos (verde) e vales/fronteiras (vermelho)")
ax_hist.set_xlabel("nível")
ax_hist.legend()

ax_full = fig.add_subplot(gs[1, 0])
ax_full.imshow(mediana127, cmap="gray", vmin=0, vmax=LIM)
for i, (lo, hi, feats) in enumerate(resultados_bandas):
    color = cores_bandas(i % 10)
    for feat in feats:
        geom = shape(feat['geometry'])
        if geom.is_empty:
            continue
        xs, ys = geom.exterior.xy
        ax_full.plot(xs, ys, color=color, linewidth=1.3)
ax_full.set_title("Todas as bandas sobrepostas")
ax_full.axis("off")

for i, (lo, hi, feats) in enumerate(resultados_bandas):
    ax = fig.add_subplot(gs[1, i + 1])
    ax.imshow(mediana127, cmap="gray", vmin=0, vmax=LIM)
    color = cores_bandas(i % 10)
    for feat in feats:
        geom = shape(feat['geometry'])
        if geom.is_empty:
            continue
        xs, ys = geom.exterior.xy
        ax.fill(xs, ys, alpha=0.35, color=color)
        ax.plot(xs, ys, color=color, linewidth=1.3)
    ax.set_title(f"banda [{lo},{hi})\n{len(feats)} polígono(s)")
    ax.axis("off")

plt.tight_layout()
out_path = "/home/cuco/projetos/casaiq/multithreshold_picos_bandas.png"
plt.savefig(out_path, dpi=150)
print("salvo em", out_path)
