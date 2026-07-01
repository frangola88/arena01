import sys, json, time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from shapely.geometry import shape
import onnxruntime as ort

sys.path.insert(0, '/home/cuco/projetos/casaiq')
from core.vetorizador_raster import vetorizar_superficie

IMG_PATH = "/home/cuco/Downloads/IMG_20260602_185719031.jpg"
GRID = 280
ONNX_PATH = '/home/cuco/.u2net/u2net.onnx'
THRESHOLD = 0.5

img_pil = Image.open(IMG_PATH).convert("RGB")

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

res = vetorizar_superficie(alpha_grid, threshold=THRESHOLD)
feats = json.loads(res['geojson'])['features']
print(f"threshold={THRESHOLD}  polígonos={len(feats)}  "
      f"cobertura={res['stats']['coverage_pct']}%  latência={res['performance']['latency_ms']}ms")

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.imshow(alpha_grid, cmap="viridis", vmin=0, vmax=1)
cmap_polys = plt.cm.tab10
for i, feat in enumerate(feats):
    geom = shape(feat['geometry'])
    if geom.is_empty:
        continue
    color = cmap_polys(i % 10)
    xs, ys = geom.exterior.xy
    ax.plot(xs, ys, color=color, linewidth=2)
    for interior in geom.interiors:
        ixs, iys = interior.xy
        ax.plot(ixs, iys, color=color, linewidth=1.2, linestyle='--')
ax.set_title(f"Polígonos direto sobre Alpha U2Net (grid) — thr={THRESHOLD}\n{len(feats)} polígono(s)  cob={res['stats']['coverage_pct']}%")
ax.axis("off")

plt.tight_layout()
out_path = "/home/cuco/projetos/casaiq/poligonos_u2net_grid.png"
plt.savefig(out_path, dpi=150)
print("salvo em", out_path)
