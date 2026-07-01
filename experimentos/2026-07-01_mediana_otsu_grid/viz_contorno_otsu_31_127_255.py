import sys, json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from shapely.geometry import shape
from skimage.filters import threshold_otsu

sys.path.insert(0, '/home/cuco/projetos/casaiq')
from core.vetorizador_raster import vetorizar_superficie

IMG_PATH = "/home/cuco/Downloads/IMG_20260602_185719031.jpg"
LIMITES = [31, 127, 255]

img = np.array(Image.open(IMG_PATH).convert("RGB")).astype(np.float64)
pc_median = np.median(img, axis=2)
pmin, pmax = pc_median.min(), pc_median.max()

fig, axes = plt.subplots(2, 3, figsize=(20, 11))
cmap_polys = plt.cm.tab10

for col, LIM in enumerate(LIMITES):
    medianaN = np.round((pc_median - pmin) / (pmax - pmin) * LIM).astype(np.int32)
    superficie = (medianaN / LIM).astype(np.float32)

    otsu_nivel = threshold_otsu(medianaN.astype(np.uint16))
    otsu_thr_norm = otsu_nivel / LIM

    res = vetorizar_superficie(superficie, threshold=otsu_thr_norm)
    feats = json.loads(res['geojson'])['features']
    print(f"LIM={LIM}  otsu_nivel={otsu_nivel}  thr_norm={otsu_thr_norm:.4f}  "
          f"polígonos={len(feats)}  cobertura={res['stats']['coverage_pct']}%  "
          f"latência={res['performance']['latency_ms']}ms")

    # Histograma
    axes[0, col].hist(medianaN.ravel(), bins=min(LIM + 1, 256), color='gray', edgecolor='black')
    axes[0, col].axvline(otsu_nivel, color='red', linewidth=2, label=f'Otsu = {otsu_nivel}')
    axes[0, col].set_title(f"Histograma mediana [0,{LIM}] + limiar Otsu")
    axes[0, col].set_xlabel("nível")
    axes[0, col].legend()

    # Contornos
    axes[1, col].imshow(medianaN, cmap="gray", vmin=0, vmax=LIM)
    for i, feat in enumerate(feats):
        geom = shape(feat['geometry'])
        if geom.is_empty:
            continue
        color = cmap_polys(i % 10)
        xs, ys = geom.exterior.xy
        axes[1, col].plot(xs, ys, color=color, linewidth=1.5)
        for interior in geom.interiors:
            ixs, iys = interior.xy
            axes[1, col].plot(ixs, iys, color=color, linewidth=1.0, linestyle='--')
    axes[1, col].set_title(f"[0,{LIM}] — Otsu thr={otsu_thr_norm:.3f} (nível {otsu_nivel})\n{len(feats)} polígono(s)  cob={res['stats']['coverage_pct']}%")
    axes[1, col].axis("off")

plt.tight_layout()
out_path = "/home/cuco/projetos/casaiq/contornos_otsu_31_127_255.png"
plt.savefig(out_path, dpi=150)
print("salvo em", out_path)
