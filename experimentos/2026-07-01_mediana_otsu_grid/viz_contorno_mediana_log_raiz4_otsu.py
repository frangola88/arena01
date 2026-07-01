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
pc_median = np.median(img, axis=2)

transformacoes = [
    ("log", np.log(pc_median)),
    ("raiz quarta", pc_median ** 0.25),
]

fig, axes = plt.subplots(3, 2, figsize=(14, 16))
cmap_polys = plt.cm.tab10

for col, (nome, pc_t) in enumerate(transformacoes):
    tmin, tmax = pc_t.min(), pc_t.max()
    mediana127 = np.round((pc_t - tmin) / (tmax - tmin) * LIM).astype(np.int32)
    superficie = (mediana127 / LIM).astype(np.float32)

    otsu_nivel = threshold_otsu(mediana127.astype(np.uint16))
    otsu_thr_norm = otsu_nivel / LIM

    res = vetorizar_superficie(superficie, threshold=otsu_thr_norm)
    feats = json.loads(res['geojson'])['features']
    print(f"{nome}: min={tmin:.4f} max={tmax:.4f}  otsu_nivel={otsu_nivel}  "
          f"thr_norm={otsu_thr_norm:.4f}  polígonos={len(feats)}  "
          f"cobertura={res['stats']['coverage_pct']}%  latência={res['performance']['latency_ms']}ms")

    axes[0, col].imshow(mediana127, cmap="gray", vmin=0, vmax=LIM)
    axes[0, col].set_title(f"Mediana ({nome}) min-max → [0,{LIM}]")
    axes[0, col].axis("off")

    axes[1, col].hist(mediana127.ravel(), bins=128, color='gray', edgecolor='black')
    axes[1, col].axvline(otsu_nivel, color='red', linewidth=2, label=f'Otsu = {otsu_nivel}')
    axes[1, col].set_title(f"Histograma ({nome}) + limiar Otsu")
    axes[1, col].set_xlabel("nível")
    axes[1, col].legend()

    axes[2, col].imshow(mediana127, cmap="gray", vmin=0, vmax=LIM)
    for i, feat in enumerate(feats):
        geom = shape(feat['geometry'])
        if geom.is_empty:
            continue
        color = cmap_polys(i % 10)
        xs, ys = geom.exterior.xy
        axes[2, col].plot(xs, ys, color=color, linewidth=1.5)
        for interior in geom.interiors:
            ixs, iys = interior.xy
            axes[2, col].plot(ixs, iys, color=color, linewidth=1.0, linestyle='--')
    axes[2, col].set_title(f"Contornos ({nome}) — thr={otsu_thr_norm:.3f} (nível {otsu_nivel})\n"
                            f"{len(feats)} polígono(s)  cob={res['stats']['coverage_pct']}%")
    axes[2, col].axis("off")

plt.tight_layout()
out_path = "/home/cuco/projetos/casaiq/mediana_log_raiz4_otsu.png"
plt.savefig(out_path, dpi=150)
print("salvo em", out_path)
