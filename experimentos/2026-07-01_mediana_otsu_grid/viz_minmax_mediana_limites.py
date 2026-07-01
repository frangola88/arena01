import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

IMG_PATH = "/home/cuco/Downloads/IMG_20260602_185719031.jpg"

img = np.array(Image.open(IMG_PATH).convert("RGB")).astype(np.float64)
pc_median = np.median(img, axis=2)

pmin = pc_median.min()
pmax = pc_median.max()

limites = [31, 127, 255]
resultados = []
for lim in limites:
    norm = (pc_median - pmin) / (pmax - pmin) * lim
    norm_int = np.round(norm).astype(np.int32)
    resultados.append(norm_int)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, res, lim in zip(axes, resultados, limites):
    ax.imshow(res, cmap="gray", vmin=0, vmax=lim)
    ax.set_title(f"Mediana min-max → [0, {lim}]")
    ax.axis("off")

plt.tight_layout()
out_path = "/home/cuco/projetos/casaiq/mediana_minmax_31_127_255.png"
plt.savefig(out_path, dpi=150)
print("min original:", pmin, "max original:", pmax)
print("salvo em", out_path)
