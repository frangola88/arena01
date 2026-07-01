import sys, json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from shapely.geometry import shape
from skimage.filters import threshold_otsu

sys.path.insert(0, '/home/cuco/projetos/casaiq')
from core.vetorizador_raster import vetorizar_superficie

IMG_PATH = "/home/cuco/Downloads/IMG_20260602_185719031.jpg"
LIM = 127

img = np.array(Image.open(IMG_PATH).convert("RGB")).astype(np.float64)

# 1. pseudocanal mediana
pc_median = np.median(img, axis=2)

# 2. raiz quadrada (pixel a pixel)
pc_quad = np.sqrt(pc_median)

# 3. min-max -> [0, 127], arredondado a inteiro
qmin, qmax = pc_quad.min(), pc_quad.max()
mediana_raiz127 = np.round((pc_quad - qmin) / (qmax - qmin) * LIM).astype(np.int32)
superficie = (mediana_raiz127 / LIM).astype(np.float32)

# 4. Otsu adaptativo + contornos
otsu_nivel = threshold_otsu(mediana_raiz127.astype(np.uint16))
otsu_thr_norm = otsu_nivel / LIM

res = vetorizar_superficie(superficie, threshold=otsu_thr_norm)
feats = json.loads(res['geojson'])['features']
print(f"mediana original: min={pc_median.min():.1f} max={pc_median.max():.1f}")
print(f"mediana raiz: min={qmin:.1f} max={qmax:.1f}")
print(f"otsu_nivel={otsu_nivel}  thr_norm={otsu_thr_norm:.4f}  "
      f"polígonos={len(feats)}  cobertura={res['stats']['coverage_pct']}%  "
      f"latência={res['performance']['latency_ms']}ms")

fig, axes = plt.subplots(1, 3, figsize=(20, 7))

axes[0].imshow(mediana_raiz127, cmap="gray", vmin=0, vmax=LIM)
axes[0].set_title(f"Mediana raiz min-max → [0,{LIM}]")
axes[0].axis("off")

axes[1].hist(mediana_raiz127.ravel(), bins=128, color='gray', edgecolor='black')
axes[1].axvline(otsu_nivel, color='red', linewidth=2, label=f'Otsu = {otsu_nivel}')
axes[1].set_title("Histograma mediana raiz [0,127] + limiar Otsu")
axes[1].set_xlabel("nível")
axes[1].legend()

axes[2].imshow(mediana_raiz127, cmap="gray", vmin=0, vmax=LIM)
cmap_polys = plt.cm.tab10
for i, feat in enumerate(feats):
    geom = shape(feat['geometry'])
    if geom.is_empty:
        continue
    color = cmap_polys(i % 10)
    xs, ys = geom.exterior.xy
    axes[2].plot(xs, ys, color=color, linewidth=1.5)
    for interior in geom.interiors:
        ixs, iys = interior.xy
        axes[2].plot(ixs, iys, color=color, linewidth=1.0, linestyle='--')
axes[2].set_title(f"Contornos — Otsu thr={otsu_thr_norm:.3f} (nível {otsu_nivel})\n{len(feats)} polígono(s)  cob={res['stats']['coverage_pct']}%")
axes[2].axis("off")

plt.tight_layout()
out_path = "/home/cuco/projetos/casaiq/mediana_raiz127_otsu.png"
plt.savefig(out_path, dpi=150)
print("salvo em", out_path)
