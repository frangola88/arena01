import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

IMG_PATH = "/home/cuco/Downloads/IMG_20260602_185719031.jpg"

img = np.array(Image.open(IMG_PATH).convert("RGB")).astype(np.float64)

pc_max = img.max(axis=2)
pc_median = np.median(img, axis=2)
pc_mean = img.mean(axis=2)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
titles = ["Pseudocanal MÁXIMO", "Pseudocanal MEDIANA", "Pseudocanal MÉDIA"]
canais = [pc_max, pc_median, pc_mean]

for ax, canal, titulo in zip(axes, canais, titles):
    ax.imshow(canal, cmap="gray", vmin=0, vmax=255)
    ax.set_title(titulo)
    ax.axis("off")

plt.tight_layout()
out_path = "/home/cuco/projetos/casaiq/pseudocanais_max_mediana_media.png"
plt.savefig(out_path, dpi=150)
print("salvo em", out_path)
