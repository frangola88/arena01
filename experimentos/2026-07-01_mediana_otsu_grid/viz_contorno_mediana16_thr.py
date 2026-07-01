import sys, json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from shapely.geometry import shape

sys.path.insert(0, '/home/cuco/projetos/casaiq')
from core.vetorizador_raster import vetorizar_superficie

IMG_PATH = "/home/cuco/Downloads/IMG_20260602_185719031.jpg"
LIM = 15

img = np.array(Image.open(IMG_PATH).convert("RGB")).astype(np.float64)
pc_median = np.median(img, axis=2)
pmin, pmax = pc_median.min(), pc_median.max()
mediana16 = np.round((pc_median - pmin) / (pmax - pmin) * LIM).astype(np.int32)
superficie = (mediana16 / LIM).astype(np.float32)

thresholds = [0.4, 0.6]
resultados = []
for thr in thresholds:
    res = vetorizar_superficie(superficie, threshold=thr)
    feats = json.loads(res['geojson'])['features']
    print(f"thr={thr}  polígonos={len(feats)}  cobertura={res['stats']['coverage_pct']}%  "
          f"latência={res['performance']['latency_ms']}ms")
    resultados.append((thr, feats))

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
cmap = plt.cm.tab10

for ax, (thr, feats) in zip(axes, resultados):
    ax.imshow(mediana16, cmap="gray", vmin=0, vmax=LIM)
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
    ax.set_title(f"thr={thr} — {len(feats)} polígono(s)")
    ax.axis("off")

plt.tight_layout()
out_path = "/home/cuco/projetos/casaiq/mediana16_contornos_thr04_06.png"
plt.savefig(out_path, dpi=150)
print("salvo em", out_path)
