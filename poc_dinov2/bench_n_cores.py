"""Benchmark: quantas cores no vetor produz melhor resultado?

Testa 1..5 cores em objetos selecionados das 3 fotos de demo.
Métrica: delta de área vs Claude (menor absoluto = melhor ajuste).
"""
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
    text_idx_path=ROOT/"data/gazetteer/text_index.json",
)

def area(b): return (b["x2"]-b["x1"])*(b["y2"]-b["y1"])*100

# ── casos de teste ────────────────────────────────────────────────────────────
# Cada caso tem até 5 cores ordenadas por abundância.
# Testamos subsets [c1], [c1,c2], [c1,c2,c3], etc.
CASOS = [
    {
        "label": "Alicate Bico Curvo (amarelo/preto)",
        "foto":  "/home/cuco/Downloads/IMG_20260602_185719031.jpg",
        "bbox":  {"x1":0.06,"y1":0.18,"x2":0.30,"y2":0.62},
        "gt":    {"x1":0.10,"y1":0.20,"x2":0.28,"y2":0.58},  # bbox "ideal" manual
        "cores": [
            {"hex":"#E8C84A","nome":"amarelo",    "area_pct":45,"parte":"cabo"},
            {"hex":"#1A1A1A","nome":"preto",      "area_pct":35,"parte":"cabo"},
            {"hex":"#909090","nome":"prata",      "area_pct":15,"parte":"cabeca"},
            {"hex":"#D4A017","nome":"amarelo esc","area_pct": 3,"parte":"borda"},
            {"hex":"#FFFFFF","nome":"branco",     "area_pct": 2,"parte":"fundo"},
        ],
    },
    {
        "label": "Alicate Corte Diagonal (amarelo/preto)",
        "foto":  "/home/cuco/Downloads/IMG_20260602_185719031.jpg",
        "bbox":  {"x1":0.28,"y1":0.14,"x2":0.58,"y2":0.65},
        "gt":    {"x1":0.30,"y1":0.16,"x2":0.55,"y2":0.62},
        "cores": [
            {"hex":"#D4A017","nome":"amarelo esc","area_pct":50,"parte":"cabo"},
            {"hex":"#1A1A1A","nome":"preto",      "area_pct":35,"parte":"cabeca"},
            {"hex":"#808080","nome":"cinza",      "area_pct":10,"parte":"mola"},
            {"hex":"#C8C8C8","nome":"prata claro","area_pct": 3,"parte":"rebite"},
            {"hex":"#FFFFFF","nome":"branco",     "area_pct": 2,"parte":"fundo"},
        ],
    },
    {
        "label": "CRAFTSMAN Azul (azul único)",
        "foto":  "/home/cuco/Downloads/duaschaves.png",
        "bbox":  {"x1":0.04,"y1":0.04,"x2":0.52,"y2":0.78},
        "gt":    {"x1":0.04,"y1":0.04,"x2":0.48,"y2":0.76},
        "cores": [
            {"hex":"#4A90D9","nome":"azul",       "area_pct":55,"parte":"cabo"},
            {"hex":"#CC2200","nome":"vermelho",   "area_pct":20,"parte":"anel"},
            {"hex":"#C0C0C0","nome":"prata",      "area_pct":18,"parte":"haste"},
            {"hex":"#8B0000","nome":"vermelho esc","area_pct": 4,"parte":"logo"},
            {"hex":"#F0F0F0","nome":"branco refl","area_pct": 3,"parte":"reflexo"},
        ],
    },
    {
        "label": "Desencapador LAOA (verde único)",
        "foto":  "/home/cuco/Downloads/IMG_20260602_185719031.jpg",
        "bbox":  {"x1":0.54,"y1":0.04,"x2":0.96,"y2":0.62},
        "gt":    {"x1":0.56,"y1":0.06,"x2":0.94,"y2":0.58},
        "cores": [
            {"hex":"#2E7D32","nome":"verde",      "area_pct":50,"parte":"cabo"},
            {"hex":"#1A1A1A","nome":"preto",      "area_pct":25,"parte":"cabo"},
            {"hex":"#C0C0C0","nome":"prata",      "area_pct":20,"parte":"corpo"},
            {"hex":"#8B7355","nome":"bege fundo", "area_pct": 3,"parte":"fundo"},
            {"hex":"#FFD700","nome":"dourado",    "area_pct": 2,"parte":"parafuso"},
        ],
    },
    {
        "label": "Chave Preta/Vermelha (preto dom.)",
        "foto":  "/home/cuco/Downloads/duaschaves.png",
        "bbox":  {"x1":0.38,"y1":0.04,"x2":0.88,"y2":0.82},
        "gt":    {"x1":0.40,"y1":0.06,"x2":0.86,"y2":0.80},
        "cores": [
            {"hex":"#1A1A1A","nome":"preto",      "area_pct":60,"parte":"cabo"},
            {"hex":"#CC2200","nome":"vermelho",   "area_pct":25,"parte":"topo"},
            {"hex":"#C0C0C0","nome":"prata",      "area_pct":10,"parte":"haste"},
            {"hex":"#8B0000","nome":"verm. esc",  "area_pct": 3,"parte":"base"},
            {"hex":"#F5DEB3","nome":"bege regua", "area_pct": 2,"parte":"fundo"},
        ],
    },
]

def iou(a, b):
    """IoU entre dois bboxes normalizados."""
    ix1=max(a["x1"],b["x1"]); iy1=max(a["y1"],b["y1"])
    ix2=min(a["x2"],b["x2"]); iy2=min(a["y2"],b["y2"])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    ua=area(a)/100 + area(b)/100 - inter
    return inter/(ua+1e-9)

# ── benchmark ────────────────────────────────────────────────────────────────
print(f"\n{'Caso':<40} {'N':>2}  {'Área%':>6}  {'ΔvsClaud':>9}  {'IoU vs GT':>10}")
print("─"*75)

# Acumula IoU por N para sumário final
iou_by_n = {n: [] for n in range(6)}   # 0 = sem cor

for caso in CASOS:
    foto = str(caso["foto"])
    bbox = caso["bbox"]
    gt   = caso["gt"]
    ac   = area(bbox)

    # N=0: sem cor
    r0 = g.refinar_bbox(foto, bbox, nome=caso["label"])
    a0 = area(r0)
    i0 = iou(r0, gt)
    iou_by_n[0].append(i0)
    print(f"  {caso['label'][:38]:<38} {'0':>2}  {a0:>6.1f}%  {a0-ac:>+8.1f}%  {i0:>10.3f}")

    for n in range(1, 6):
        subset = caso["cores"][:n]
        r = g.refinar_bbox(foto, bbox, nome=caso["label"], cores=subset)
        ar = area(r)
        iu = iou(r, gt)
        iou_by_n[n].append(iu)
        marker = " ◄" if iu == max(iou(
            g.refinar_bbox(foto, bbox, nome=caso["label"], cores=caso["cores"][:k]),
            gt) for k in range(1,6)) else ""
        print(f"  {caso['label'][:38]:<38} {n:>2}  {ar:>6.1f}%  {ar-ac:>+8.1f}%  {iu:>10.3f}{marker}")
    print()

# ── sumário ──────────────────────────────────────────────────────────────────
print("─"*75)
print(f"\n{'N cores':<10} {'IoU médio':>12}  {'vs N=0':>8}")
iou0_mean = np.mean(iou_by_n[0])
print(f"  {'sem cor':<9} {iou0_mean:>12.3f}")
for n in range(1, 6):
    mean_n = np.mean(iou_by_n[n])
    delta  = mean_n - iou0_mean
    star   = " ★ MELHOR" if mean_n == max(np.mean(iou_by_n[k]) for k in range(1,6)) else ""
    print(f"  {n:<9} {mean_n:>12.3f}  {delta:>+7.3f}{star}")
