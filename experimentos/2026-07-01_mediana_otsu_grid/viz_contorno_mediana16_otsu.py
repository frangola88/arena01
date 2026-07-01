import sys, json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from shapely.geometry import shape
from skimage.filters import threshold_otsu

sys.path.insert(0, '/home/cuco/projetos/casaiq')
from core.vetorizador_raster import vetorizar_superficie

IMG_PATH = "/home/cuco/Downloads/IMG_20260602_185719031.jpg"
LIM = 15

img = np.array(Image.open(IMG_PATH).convert("RGB")).astype(np.float64)
pc_median = np.median(img, axis=2)
pmin, pmax = pc_median.min(), pc_median.max()
mediana16 = np.round((pc_median - pmin) / (pmax - pmin) * LIM).astype(np.int32)
superficie = (mediana16 / LIM).astype(np.float32)

# Otsu direto sobre os níveis discretos [0,15] (16 bins naturais)
otsu_nivel = threshold_otsu(mediana16.astype(np.uint8))
otsu_thr_norm = otsu_nivel / LIM

print(f"Otsu (nível [0,{LIM}]) = {otsu_nivel}  ->  threshold normalizado = {otsu_thr_norm:.4f}")

res = vetorizar_superficie(superficie, threshold=otsu_thr_norm)
feats = json.loads(res['geojson'])['features']
print(f"thr_otsu={otsu_thr_norm:.4f}  polígonos={len(feats)}  "
      f"cobertura={res['stats']['coverage_pct']}%  latência={res['performance']['latency_ms']}ms")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

axes[0].hist(mediana16.ravel(), bins=range(LIM + 2), color='gray', edgecolor='black')
axes[0].axvline(otsu_nivel + 0.5, color='red', linewidth=2, label=f'Otsu = {otsu_nivel}')
axes[0].set_title(f"Histograma mediana [0,{LIM}] + limiar Otsu")
axes[0].set_xlabel("nível")
axes[0].legend()

axes[1].imshow(mediana16, cmap="gray", vmin=0, vmax=LIM)
cmap = plt.cm.tab10
for i, feat in enumerate(feats):
    geom = shape(feat['geometry'])
    if geom.is_empty:
        continue
    color = cmap(i % 10)
    xs, ys = geom.exterior.xy
    axes[1].plot(xs, ys, color=color, linewidth=1.5)
    for interior in geom.interiors:
        ixs, iys = interior.xy
        axes[1].plot(ixs, iys, color=color, linewidth=1.0, linestyle='--')
axes[1].set_title(f"Contornos — Otsu thr={otsu_thr_norm:.3f} (nível {otsu_nivel})\n{len(feats)} polígono(s)")
axes[1].axis("off")

plt.tight_layout()
out_path = "/home/cuco/projetos/casaiq/mediana16_contornos_otsu.png"
plt.savefig(out_path, dpi=150)
print("salvo em", out_path)
