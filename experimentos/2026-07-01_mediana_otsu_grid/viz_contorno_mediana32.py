import sys, json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from shapely.geometry import shape

sys.path.insert(0, '/home/cuco/projetos/casaiq')
from core.vetorizador_raster import vetorizar_superficie
from core.config import ISOCONTOUR_THRESHOLD

IMG_PATH = "/home/cuco/Downloads/IMG_20260602_185719031.jpg"

img = np.array(Image.open(IMG_PATH).convert("RGB")).astype(np.float64)
pc_median = np.median(img, axis=2)

pmin, pmax = pc_median.min(), pc_median.max()
mediana32 = np.round((pc_median - pmin) / (pmax - pmin) * 31).astype(np.int32)

# superficie normalizada [0,1] exigida por vetorizar_superficie
superficie = (mediana32 / 31.0).astype(np.float32)

resultado = vetorizar_superficie(superficie, threshold=ISOCONTOUR_THRESHOLD)
feats = json.loads(resultado['geojson'])['features']
print(f"threshold={ISOCONTOUR_THRESHOLD}  polígonos={len(feats)}  "
      f"cobertura={resultado['stats']['coverage_pct']}%  "
      f"latência={resultado['performance']['latency_ms']}ms")

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.imshow(mediana32, cmap="gray", vmin=0, vmax=31)

cmap = plt.cm.tab10
for i, feat in enumerate(feats):
    geom = shape(feat['geometry'])
    if geom.is_empty:
        continue
    color = cmap(i % 10)
    xs, ys = geom.exterior.xy
    ax.plot(xs, ys, color=color, linewidth=1.5)
    for interior in geom.interiors:
        ixs, iys = interior.xy
        ax.plot(ixs, iys, color=color, linewidth=1.0, linestyle='--')

ax.set_title(f"Contornos (isocontour) sobre mediana [0,31] — thr={ISOCONTOUR_THRESHOLD}\n"
             f"{len(feats)} polígono(s)")
ax.axis("off")

plt.tight_layout()
out_path = "/home/cuco/projetos/casaiq/mediana32_contornos.png"
plt.savefig(out_path, dpi=150)
print("salvo em", out_path)
