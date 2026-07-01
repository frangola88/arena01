"""
Visualização grid: Pipeline Alpha U2Net → Morph Close → Vetorização
Gera dois arquivos:
  /tmp/grid_alpha_11_ferramentas.png  — grid 4×3 com painel 2×2 interno
  /tmp/grid_alpha_comparacao.png      — 11 linhas × 3 colunas comparação
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
from PIL import Image
from shapely.geometry import mapping

sys.path.insert(0, "/home/cuco/projetos/casaiq")
from core.alpha_pipeline import processar_imagem

ITEMS = [
    {"cat": "alicate",    "path": "/home/cuco/dataset_imgs/1a016cd1da7638554b687ec2d9c9438392020afee437b774159a58ff1b9185d0.jpg"},
    {"cat": "martelo",    "path": "/home/cuco/dataset_imgs/16cf9ba75091dc7c02fc5970e1076cbebb9d47b4a42516ea1da4da62c7124091.jpg"},
    {"cat": "chave",      "path": "/home/cuco/dataset_imgs/1a7954245a251ba9470eb743620e51d794ec32b1f617cf866b1ee7c31b735ae9.jpg"},
    {"cat": "faca",       "path": "/home/cuco/dataset_imgs/818a0ad8f349d47b6cce3ad89cb90ce37b6b762c1c66a9b655ea2899a6377f8a.jpg"},
    {"cat": "tesoura",    "path": "/home/cuco/dataset_imgs/c6e024394e4933ceacd5c00139862573816f891d56c5f16a03e53b4090f441b4.jpg"},
    {"cat": "furadeira",  "path": "/home/cuco/dataset_imgs/d517e12be1b185278a05ab1c1ce43fcad983df2b0f46669d43fe6c8409f17a9b.jpg"},
    {"cat": "serra",      "path": "/home/cuco/dataset_imgs/a8cc0b7e46939a5a9714bd8db620d5f9fe7eaee01192fd8e347b0a597f351b20.jpg"},
    {"cat": "broca",      "path": "/home/cuco/dataset_imgs/7bc97eaa54adad3d0d0f93eab77268ce5109bd29293b1545bc4bfb52d5cb88d1.jpg"},
    {"cat": "nivel",      "path": "/home/cuco/dataset_imgs/ff8321c12f60b8e42c4cf3aac0542823da7763b84a6508fd5e071875296c329c.jpg"},
    {"cat": "trena",      "path": "/home/cuco/dataset_imgs/b7330467acc32c0b62c1d03b88ab97eac4591bb4be98ba9a60dd2a5ccf41129a.jpg"},
    {"cat": "foto_real",  "path": "/home/cuco/Downloads/IMG_20260602_185719031.jpg"},
]

GRID = 280
TAB10 = plt.cm.tab10.colors


def load_thumb(path, size=GRID):
    img = Image.open(path).convert("RGB")
    img.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), (240, 240, 240))
    ox = (size - img.width) // 2
    oy = (size - img.height) // 2
    canvas.paste(img, (ox, oy))
    return np.array(canvas)


def draw_polys(ax, thumb, polys, alpha_fill=0.3):
    ax.imshow(thumb)
    for i, p in enumerate(polys):
        geom = p["geometry"]
        color = TAB10[i % 10]
        gtype = geom.geom_type
        if gtype == "Polygon":
            parts = [geom]
        else:
            parts = list(geom.geoms)
        for part in parts:
            xs, ys = part.exterior.xy
            ax.fill(xs, ys, alpha=alpha_fill, color=color)
            ax.plot(xs, ys, color=color, linewidth=1.2)
    ax.axis("off")


def gray_sup(thumb_arr):
    gray = np.mean(thumb_arr, axis=2).astype(np.float32)
    sup = np.clip(255 - gray, 0, 31) / 31.0
    return sup


print("=== Processando imagens ===")
results = []
for item in ITEMS:
    print(f"  {item['cat']} ...", flush=True)
    r = processar_imagem(item["path"], grid=GRID)
    r["cat"] = item["cat"]
    r["path"] = item["path"]
    r["thumb"] = load_thumb(item["path"])
    results.append(r)
    print(f"    n_poly={r['n_poligonos']} cob={r['cobertura_pct']:.1f}% t={r['tempo_ms']['total']:.0f}ms")

# ─── Figura 1: grid 4×3, painel 2×2 interno ──────────────────────────────────
print("\n=== Gerando grid_alpha_11_ferramentas.png ===")
NCOLS = 3
NROWS = 4
fig, axes = plt.subplots(NROWS * 2, NCOLS * 2, figsize=(24, 32), dpi=110)
fig.suptitle(
    "Pipeline Alpha U2Net → Morph Close → Vetorização — 11 ferramentas",
    fontsize=16, fontweight="bold", y=0.995
)

for idx, r in enumerate(results):
    row_block = idx // NCOLS
    col_block = idx % NCOLS
    r0 = row_block * 2
    c0 = col_block * 2

    ax_tl = axes[r0,     c0]
    ax_tr = axes[r0,     c0 + 1]
    ax_bl = axes[r0 + 1, c0]
    ax_br = axes[r0 + 1, c0 + 1]

    thumb = r["thumb"]
    alpha = r["alpha_grid"]
    morph = r["sup_morph"]
    polys = r["poligonos"]

    # TL: original
    ax_tl.imshow(thumb)
    ax_tl.set_title("original", fontsize=7)
    ax_tl.axis("off")

    # TR: alpha U2Net (plasma)
    ax_tr.imshow(alpha, cmap="plasma", vmin=0, vmax=1)
    ax_tr.set_title("alpha U2Net", fontsize=7)
    ax_tr.axis("off")

    # BL: morph close
    ax_bl.imshow(morph, cmap="plasma", vmin=0, vmax=1)
    ax_bl.set_title("morph close", fontsize=7)
    ax_bl.axis("off")

    # BR: polígonos sobre original
    draw_polys(ax_br, thumb, polys)
    ax_br.set_title("polígonos", fontsize=7)

    title = (
        f"{r['cat'].upper()} — {r['n_poligonos']} poly | "
        f"área={r['area_total']:.0f}px² | cob={r['cobertura_pct']:.1f}%"
    )
    # Label block via annotation in TL
    ax_tl.set_title(title, fontsize=8, fontweight="bold")

# Esconder células extras
n = len(results)
total_cells = NROWS * NCOLS
for idx in range(n, total_cells):
    row_block = idx // NCOLS
    col_block = idx % NCOLS
    r0 = row_block * 2
    c0 = col_block * 2
    for dr in range(2):
        for dc in range(2):
            axes[r0 + dr, c0 + dc].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.995])
out1 = "/tmp/grid_alpha_11_ferramentas.png"
fig.savefig(out1, dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"Salvo: {out1}")

# ─── Figura 2: comparação 11 linhas × 3 colunas ──────────────────────────────
print("\n=== Gerando grid_alpha_comparacao.png ===")
fig2, axes2 = plt.subplots(len(results), 3, figsize=(18, 44), dpi=100)
fig2.suptitle(
    "Comparação: Original | Polígono U2Net | Cinza [0-31]",
    fontsize=14, fontweight="bold", y=1.001
)

col_labels = ["Original", "Polígono U2Net", "Cinza [0-31]"]
for col, lbl in enumerate(col_labels):
    axes2[0, col].set_title(lbl, fontsize=11, fontweight="bold", pad=8)

for i, r in enumerate(results):
    thumb = r["thumb"]
    polys = r["poligonos"]
    sup_gray = gray_sup(thumb)

    row_title = (
        f"{r['cat']} | {r['n_poligonos']} poly | "
        f"área={r['area_total']:.0f}px² | cob={r['cobertura_pct']:.1f}%"
    )

    # Col 0: original
    axes2[i, 0].imshow(thumb)
    axes2[i, 0].set_ylabel(row_title, fontsize=8, rotation=0, labelpad=120, va="center")
    axes2[i, 0].axis("off")

    # Col 1: polígono U2Net
    draw_polys(axes2[i, 1], thumb, polys)

    # Col 2: cinza [0-31]
    axes2[i, 2].imshow(sup_gray, cmap="gray", vmin=0, vmax=1)
    axes2[i, 2].axis("off")

plt.tight_layout()
out2 = "/tmp/grid_alpha_comparacao.png"
fig2.savefig(out2, dpi=100, bbox_inches="tight")
plt.close(fig2)
print(f"Salvo: {out2}")

print("\n=== DONE ===")
