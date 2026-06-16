"""Demo comparativo: sem cor vs com cor (cores_dominantes do Claude).
Mostra side-by-side 3 versões: Claude original / DINOv2 sem cor / DINOv2 com cor.
"""
import sys, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from core.gazetteer import GazetteerMatcher

FOTO = Path("/home/cuco/Downloads/IMG_20260324_153910599.jpg")
EMB   = ROOT / "embeddings.npy"
FAISS = ROOT / "data/gazetteer/gazetteer.faiss"
MAP   = ROOT / "data/gazetteer/index_mapping.jsonl"
TEXT  = ROOT / "data/gazetteer/text_index.json"

print("[demo] iniciando gazetteer...", flush=True)
g = GazetteerMatcher(emb_path=EMB, faiss_path=FAISS, map_path=MAP, text_idx_path=TEXT)

img = Image.open(FOTO).convert("RGB")
W, H = img.size

# Objetos com cores_dominantes como o Claude passaria
OBJETOS = [
    {
        "nome":  "Chave de Fenda Philips cluster",
        "bbox":  {"x1": 0.50, "y1": 0.00, "x2": 0.98, "y2": 0.20},
        "cor_box": "#FF4444",
        "cores": [
            {"hex": "#C0392B", "nome": "vermelho escuro", "area_pct": 45, "parte": "cabo"},
            {"hex": "#C0C0C0", "nome": "prata",           "area_pct": 40, "parte": "haste"},
            {"hex": "#F5CBA7", "nome": "bege translucido","area_pct": 15, "parte": "cabo"},
        ],
    },
    {
        "nome":  "Chave de Fenda Cabo Laranja",
        "bbox":  {"x1": 0.22, "y1": 0.12, "x2": 0.80, "y2": 0.30},
        "cor_box": "#FF9900",
        "cores": [
            {"hex": "#E06428", "nome": "laranja",      "area_pct": 55, "parte": "cabo"},
            {"hex": "#C0C0C0", "nome": "prata",        "area_pct": 30, "parte": "haste"},
            {"hex": "#8B4513", "nome": "marrom madeira","area_pct": 15, "parte": "cabo"},
        ],
    },
    {
        "nome":  "Chave de Fenda Amarela Worker",
        "bbox":  {"x1": 0.28, "y1": 0.68, "x2": 0.75, "y2": 0.98},
        "cor_box": "#FFEE00",
        "cores": [
            {"hex": "#F4D03F", "nome": "amarelo",  "area_pct": 40, "parte": "cabo"},
            {"hex": "#1A1A1A", "nome": "preto",    "area_pct": 35, "parte": "cabo"},
            {"hex": "#C0C0C0", "nome": "prata",    "area_pct": 25, "parte": "haste"},
        ],
    },
    {
        "nome":  "Pote Plastico",
        "bbox":  {"x1": 0.00, "y1": 0.50, "x2": 0.30, "y2": 0.98},
        "cor_box": "#CC88FF",
        "cores": [
            {"hex": "#F0F0F0", "nome": "branco translucido", "area_pct": 90, "parte": "corpo"},
            {"hex": "#D0D0D0", "nome": "cinza claro",        "area_pct": 10, "parte": "borda"},
        ],
    },
]

resultados = []
for obj in OBJETOS:
    t0 = time.time()
    r_sem = g.refinar_bbox(str(FOTO), obj["bbox"], nome=obj["nome"])
    t1 = time.time()
    r_com = g.refinar_bbox(str(FOTO), obj["bbox"], nome=obj["nome"], cores=obj["cores"])
    t2 = time.time()

    def area(b): return (b["x2"]-b["x1"])*(b["y2"]-b["y1"])*100
    ac = area(obj["bbox"])
    as_ = area(r_sem)
    aco = area(r_com)

    print(f"\n  {obj['nome']}")
    print(f"    Claude   : área={ac:.1f}%")
    print(f"    Sem cor  : área={as_:.1f}%  ({as_-ac:+.1f}%)  {t1-t0:.1f}s")
    print(f"    Com cor  : área={aco:.1f}%  ({aco-ac:+.1f}%)  {t2-t1:.1f}s")

    resultados.append({**obj, "r_sem": r_sem, "r_com": r_com})

# ── visualização 3 colunas ────────────────────────────────────────────────────
PAD   = 12
LBL_H = 38
ncols = 3
out = Image.new("RGB", (W*ncols + PAD*(ncols-1), H + LBL_H), (18, 18, 18))

frames = [
    (img.copy(), "Claude — semântica",           "#888888"),
    (img.copy(), "DINOv2 — sem cor",             "#4499FF"),
    (img.copy(), "DINOv2 — COM cor (novo)",      "#44FF88"),
]
draws = [ImageDraw.Draw(f[0]) for f in frames]

for obj, rd in zip(OBJETOS, resultados):
    cor   = obj["cor_box"]
    bboxes = [obj["bbox"], rd["r_sem"], rd["r_com"]]
    for i, (bb, lw) in enumerate(zip(bboxes, [5, 4, 4])):
        x1=int(bb["x1"]*W); y1=int(bb["y1"]*H)
        x2=int(bb["x2"]*W); y2=int(bb["y2"]*H)
        for t in range(lw):
            draws[i].rectangle([x1-t, y1-t, x2+t, y2+t], outline=cor)
        if i == 0:
            draws[i].text((x1+4, y1+4), obj["nome"][:20], fill=cor)

canvas = Image.new("RGB", (W*ncols + PAD*(ncols-1), H + LBL_H), (18, 18, 18))
cd = ImageDraw.Draw(canvas)
for i, (frame, label, lcor) in enumerate(frames):
    ox = i * (W + PAD)
    canvas.paste(frame, (ox, LBL_H))
    cd.rectangle([ox, 0, ox+W-1, LBL_H-1], fill=(30, 30, 30))
    cd.text((ox+8, 11), label, fill=lcor)

out_path = ROOT / "poc_dinov2/output/demo_cores_result.jpg"
canvas.save(out_path, "JPEG", quality=92)
print(f"\n[demo] salvo → {out_path}")
