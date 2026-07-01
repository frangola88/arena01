import sys, json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from shapely.geometry import shape

sys.path.insert(0, '/home/cuco/projetos/casaiq')
from core.vetorizador_raster import vetorizar_superficie
from core.config import ISOCONTOUR_THRESHOLD

IMG_PATH = "/home/cuco/Downloads/IMG_20260602_185719031.jpg"
LIM = 15  # 0-15 → 16 níveis

img = np.array(Image.open(IMG_PATH).convert("RGB")).astype(np.float64)
pc_median = np.median(img, axis=2)

pmin, pmax = pc_median.min(), pc_median.max()
mediana16 = np.round((pc_median - pmin) / (pmax - pmin) * LIM).astype(np.int32)

superficie = (mediana16 / LIM).astype(np.float32)

resultado = vetorizar_superficie(superficie, threshold=ISOCONTOUR_THRESHOLD)
feats = json.loads(resultado['geojson'])['features']
print(f"threshold={ISOCONTOUR_THRESHOLD}  polígonos={len(feats)}  "
      f"cobertura={resultado['stats']['coverage_pct']}%  "
      f"latência={resultado['performance']['latency_ms']}ms")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

axes[0].imshow(mediana16, cmap="gray", vmin=0, vmax=LIM)
axes[0].set_title(f"Mediana min-max → [0,{LIM}]")
axes[0].axis("off")

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
axes[1].set_title(f"Contornos (isocontour) — thr={ISOCONTOUR_THRESHOLD}\n{len(feats)} polígono(s)")
axes[1].axis("off")

plt.tight_layout()
out_path = "/home/cuco/projetos/casaiq/mediana16_contornos.png"
plt.savefig(out_path, dpi=150)
print("min original:", pmin, "max original:", pmax)
print("salvo em", out_path)
