"""Diagnóstico: Gaussiana vs Cosseno para o caso Philips cluster."""
import sys, numpy as np
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from core.gazetteer import GazetteerMatcher

g = GazetteerMatcher(
    emb_path=ROOT/"embeddings.npy",
    faiss_path=ROOT/"data/gazetteer/gazetteer.faiss",
    map_path=ROOT/"data/gazetteer/index_mapping.jsonl",
)

FOTO  = "/home/cuco/Downloads/IMG_20260324_153910599.jpg"
BBOX  = {"x1": 0.50, "y1": 0.00, "x2": 0.98, "y2": 0.20}
CORES = [
    {"hex": "#C0392B", "nome": "vermelho escuro",  "area_pct": 45, "parte": "cabo"},
    {"hex": "#C0C0C0", "nome": "cinza metalico",   "area_pct": 40, "parte": "haste"},
    {"hex": "#F5CBA7", "nome": "bege translucido", "area_pct": 15, "parte": "cabo"},
]

def area(b): return (b["x2"]-b["x1"])*(b["y2"]-b["y1"])*100
def hex_to_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i+2], 16) for i in (0, 2, 4)], dtype=float)

# ── sem cor ────────────────────────────────────────────────────────────────────
r0 = g.refinar_bbox(FOTO, BBOX, nome="Philips")
print(f"Sem cor    : {area(r0):.1f}%  (Claude=9.6%)")

# ── cosseno atual ─────────────────────────────────────────────────────────────
r_cos = g.refinar_bbox(FOTO, BBOX, nome="Philips", cores=CORES)
print(f"Cosseno N=3: {area(r_cos):.1f}%")

# ── diagnóstico: Gaussiana manual ─────────────────────────────────────────────
S = 64; p = 14; tile = 896
img = Image.open(FOTO).convert("RGB")
img_r = img.resize((tile, tile), Image.LANCZOS)
arr = np.array(img_r, dtype=float)
patches_rgb = arr.reshape(S, p, S, p, 3).mean(axis=(1, 3))   # (64, 64, 3)

sigma = 45.0
w_gauss = np.zeros((S, S))
for c in CORES:
    dist = np.linalg.norm(patches_rgb - hex_to_rgb(c["hex"]), axis=2)
    w_gauss += np.exp(-dist**2 / (2*sigma**2)) * c["area_pct"]
w_gauss = (w_gauss / w_gauss.max())

# ── diagnóstico: Cosseno manual ───────────────────────────────────────────────
K = len(CORES)
sims = np.zeros((S, S, K))
prop = np.zeros(K)
for k, c in enumerate(CORES):
    dist = np.linalg.norm(patches_rgb - hex_to_rgb(c["hex"]), axis=2)
    sims[:,:,k] = np.exp(-dist**2 / (2*sigma**2))
    prop[k] = c["area_pct"]
prop /= prop.sum()
sims_unit = sims / (np.linalg.norm(sims, axis=2, keepdims=True) + 1e-9)
prop_unit = prop / (np.linalg.norm(prop) + 1e-9)
w_cos = (sims_unit * prop_unit).sum(axis=2)
w_cos = w_cos / (w_cos.max() + 1e-9)

# ── blend 50/50 ──────────────────────────────────────────────────────────────
w_blend = 0.5 * w_gauss + 0.5 * w_cos
w_blend = w_blend / w_blend.max()

# ── compara discriminação por região ─────────────────────────────────────────
# Região Philips cluster: colunas 30-60, linhas 0-20 (topo direito)
# Região laranja: colunas 10-40, linhas 40-55 (baixo esquerdo)
def region_score(w, r0, r1, c0, c1):
    return float(w[r0:r1, c0:c1].mean())

ph = (0, 20, 30, 60)   # Philips: rows 0-20, cols 30-60
la = (40, 55, 10, 40)  # Laranja: rows 40-55, cols 10-40

print(f"\n{'':20s}  {'Philips':>10}  {'Laranja':>10}  {'ratio':>8}")
for nome, w in [("Gaussiana", w_gauss), ("Cosseno",   w_cos), ("Blend 50/50", w_blend)]:
    sp = region_score(w, *ph)
    sl = region_score(w, *la)
    ratio = sp/(sl+1e-9)
    print(f"  {nome:18s}  {sp:>10.3f}  {sl:>10.3f}  {ratio:>7.2f}x")

print("\nRatio > 1 = Philips recebe mais peso que o laranja (correto)")
print("Ratio < 1 = laranja recebe mais peso (bug — DINOv2 vai para baixo)")
