import sys, json, time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from shapely.geometry import shape
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import onnxruntime as ort

sys.path.insert(0, '/home/cuco/projetos/casaiq')
from core.vetorizador_raster import vetorizar_superficie

IMG_PATH = "/home/cuco/Downloads/IMG_20260602_185719031.jpg"
LIM = 127
GRID = 280
ONNX_PATH = '/home/cuco/.u2net/u2net.onnx'

img_pil = Image.open(IMG_PATH).convert("RGB")

# ── (a) downsample ANTES de vetorizar ───────────────────────────────────────
img_grid = np.array(img_pil.resize((GRID, GRID), Image.LANCZOS), dtype=np.float64)
pc_median = np.median(img_grid, axis=2)
pc_raiz = np.sqrt(pc_median)
tmin, tmax = pc_raiz.min(), pc_raiz.max()
mediana127 = np.round((pc_raiz - tmin) / (tmax - tmin) * LIM).astype(np.int32)

hist, edges = np.histogram(mediana127.ravel(), bins=LIM + 1, range=(0, LIM))
hist_smooth = gaussian_filter1d(hist.astype(float), sigma=2)
peaks, _ = find_peaks(hist_smooth, prominence=hist_smooth.max() * 0.12, distance=20)
print(f"picos (grid {GRID}x{GRID}): {peaks.tolist()}")

def fwhm_bounds(hist_s, peak):
    half = hist_s[peak] / 2.0
    lo = peak
    while lo > 0 and hist_s[lo] > half:
        lo -= 1
    hi = peak
    while hi < len(hist_s) - 1 and hist_s[hi] > half:
        hi += 1
    return lo, hi

bandas = []
for p in peaks:
    lo, hi = fwhm_bounds(hist_smooth, p)
    bandas.append((p, lo, hi))
    print(f"pico={p}  faixa FWHM=[{lo},{hi}]")

# assume 2 picos: [0]=esquerdo(escuro/objeto), [1]=direito(claro/fundo)
p_obj, lo_obj, hi_obj = bandas[0]
p_bg, lo_bg, hi_bg = bandas[1]

classes = np.zeros_like(mediana127, dtype=np.uint8)  # 0 = transição
classes[(mediana127 >= lo_obj) & (mediana127 <= hi_obj)] = 1  # objeto travado
classes[(mediana127 >= lo_bg) & (mediana127 <= hi_bg)] = 2    # fundo travado

pct_transicao = 100 * np.mean(classes == 0)
print(f"transição (indefinido): {pct_transicao:.1f}% dos pixels do grid")

# ── (b) U2Net só para resolver a faixa cinza (transição) ───────────────────
sess = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
IN_NAME, OUT_NAME = sess.get_inputs()[0].name, sess.get_outputs()[0].name

def u2net_alpha(pil_img):
    inp = pil_img.resize((320, 320), Image.LANCZOS).convert('RGB')
    x = np.array(inp, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = ((x - mean) / std).transpose(2, 0, 1)[None]
    pred = sess.run([OUT_NAME], {IN_NAME: x})[0][0, 0]
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    return np.array(
        Image.fromarray((pred * 255).astype(np.uint8)).resize(pil_img.size, Image.LANCZOS),
        dtype=np.float32) / 255.0

print("rodando U2Net...", end=' ', flush=True)
t0 = time.time()
alpha_full = u2net_alpha(img_pil)
alpha_grid = np.array(
    Image.fromarray((alpha_full * 255).astype(np.uint8)).resize((GRID, GRID), Image.LANCZOS),
    dtype=np.float32) / 255.0
print(f"{(time.time()-t0)*1000:.0f}ms")

# combinação: travados ficam travados; transição resolvida por U2Net (>=0.5 -> objeto)
fg_final = np.zeros_like(mediana127, dtype=np.float32)
fg_final[classes == 1] = 1.0
fg_final[classes == 2] = 0.0
mask_transicao = classes == 0
fg_final[mask_transicao] = (alpha_grid[mask_transicao] >= 0.5).astype(np.float32)

# ── Vetorização (grid reduzido → deve reduzir MUITO o ruído) ───────────────
res_obj = vetorizar_superficie((classes == 1).astype(np.float32), threshold=0.5)
res_bg  = vetorizar_superficie((classes == 2).astype(np.float32), threshold=0.5)
res_final = vetorizar_superficie(fg_final, threshold=0.5)

feats_obj = json.loads(res_obj['geojson'])['features']
feats_bg = json.loads(res_bg['geojson'])['features']
feats_final = json.loads(res_final['geojson'])['features']

print(f"[grid] objeto travado: {len(feats_obj)} polígonos, cobertura={res_obj['stats']['coverage_pct']}%")
print(f"[grid] fundo travado:  {len(feats_bg)} polígonos, cobertura={res_bg['stats']['coverage_pct']}%")
print(f"[grid] combinado+U2Net: {len(feats_final)} polígonos, cobertura={res_final['stats']['coverage_pct']}%")

# ── Figura ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(20, 13))

axes[0, 0].imshow(mediana127, cmap="gray", vmin=0, vmax=LIM)
axes[0, 0].set_title(f"Mediana(raiz) [0,{LIM}] — grid {GRID}x{GRID}")
axes[0, 0].axis("off")

cmap_classes = ListedColormap(["#888888", "#1f77b4", "#d62728"])
axes[0, 1].imshow(classes, cmap=cmap_classes, vmin=0, vmax=2)
axes[0, 1].set_title(f"Classes travadas (FWHM) — {pct_transicao:.1f}% transição")
axes[0, 1].axis("off")

axes[0, 2].imshow(alpha_grid, cmap="viridis", vmin=0, vmax=1)
axes[0, 2].set_title("Alpha U2Net (grid)")
axes[0, 2].axis("off")

def draw(ax, feats, color):
    ax.imshow(mediana127, cmap="gray", vmin=0, vmax=LIM)
    for feat in feats:
        geom = shape(feat['geometry'])
        if geom.is_empty:
            continue
        xs, ys = geom.exterior.xy
        ax.plot(xs, ys, color=color, linewidth=1.5)
        for interior in geom.interiors:
            ixs, iys = interior.xy
            ax.plot(ixs, iys, color=color, linewidth=1.0, linestyle='--')
    ax.axis("off")

draw(axes[1, 0], feats_obj, "#1f77b4")
axes[1, 0].set_title(f"Contorno objeto travado (grid)\n{len(feats_obj)} polígono(s)")

draw(axes[1, 1], feats_bg, "#d62728")
axes[1, 1].set_title(f"Contorno fundo travado (grid)\n{len(feats_bg)} polígono(s)")

draw(axes[1, 2], feats_final, "#2ca02c")
axes[1, 2].set_title(f"Combinado (travados + U2Net na transição)\n{len(feats_final)} polígono(s)  cob={res_final['stats']['coverage_pct']}%")

plt.tight_layout()
out_path = "/home/cuco/projetos/casaiq/grid_picos_u2net_transicao.png"
plt.savefig(out_path, dpi=150)
print("salvo em", out_path)
