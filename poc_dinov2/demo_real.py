"""Demo com foto REAL de ferramentas — chaves de fenda na bancada.

Simula o que Claude identificaria na foto e roda GazetteerMatcher 36k + 896px.
"""
import sys, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

FOTO = Path("/home/cuco/Downloads/IMG_20260324_153910599.jpg")

# O que Claude identificaria nesta foto (bboxes semânticas imprecisas)
CLAUDE_OUTPUT = [
    {
        "nome":  "Chave de Fenda Philips",
        "brand": "CRAFTSMAN",
        "bbox":  {"x1": 0.50, "y1": 0.00, "x2": 0.98, "y2": 0.20},
        "cor":   "#FF4444",
    },
    {
        "nome":  "Chave de Fenda com Cabo Laranja",
        "brand": "",
        "bbox":  {"x1": 0.22, "y1": 0.12, "x2": 0.80, "y2": 0.30},
        "cor":   "#FF9900",
    },
    {
        "nome":  "Regua Transparente 30cm",
        "brand": "Morning Glory",
        "bbox":  {"x1": 0.04, "y1": 0.27, "x2": 0.98, "y2": 0.42},
        "cor":   "#AADDFF",
    },
    {
        "nome":  "Chave de Fenda Amarela Bimateria",
        "brand": "WORKER",
        "bbox":  {"x1": 0.28, "y1": 0.68, "x2": 0.75, "y2": 0.98},
        "cor":   "#FFEE00",
    },
    {
        "nome":  "Pote Plastico",
        "brand": "",
        "bbox":  {"x1": 0.00, "y1": 0.50, "x2": 0.30, "y2": 0.98},
        "cor":   "#CC88FF",
    },
]

# ── init gazetteer ────────────────────────────────────────────────────────────
from core.gazetteer import GazetteerMatcher
EMB   = ROOT / "embeddings.npy"
FAISS = ROOT / "data/gazetteer/gazetteer.faiss"
MAP   = ROOT / "data/gazetteer/index_mapping.jsonl"
TEXT  = ROOT / "data/gazetteer/text_index.json"

print(f"[demo] Inicializando GazetteerMatcher ({np.load(EMB).shape[0]:,} embeddings)...")
t0 = time.time()
g = GazetteerMatcher(emb_path=EMB, faiss_path=FAISS, map_path=MAP, text_idx_path=TEXT)
print(f"[demo] Carregado em {time.time()-t0:.1f}s\n")

img = Image.open(FOTO).convert("RGB")
W, H = img.size
print(f"[demo] Foto: {FOTO.name}  {W}×{H}\n")

# ── refinamento ───────────────────────────────────────────────────────────────
resultados = []
for obj in CLAUDE_OUTPUT:
    t1 = time.time()
    refined = g.refinar_bbox(str(FOTO), obj["bbox"],
                             nome=obj["nome"], brand=obj["brand"])
    elapsed = time.time() - t1

    bc = obj["bbox"]
    br = refined
    area_c = (bc["x2"]-bc["x1"]) * (bc["y2"]-bc["y1"])
    area_r = (br["x2"]-br["x1"]) * (br["y2"]-br["y1"])
    reducao = (1 - area_r/area_c) * 100 if area_c > 0 else 0

    print(f"  {obj['nome']}")
    print(f"    Claude  : área={area_c*100:.1f}%  ({bc['x1']:.2f},{bc['y1']:.2f})→({bc['x2']:.2f},{bc['y2']:.2f})")
    print(f"    Refinada: área={area_r*100:.1f}%  ({br['x1']:.2f},{br['y1']:.2f})→({br['x2']:.2f},{br['y2']:.2f})")
    print(f"    Redução : {reducao:.0f}%  ({elapsed:.1f}s)\n")

    resultados.append({**obj, "refined": refined, "reducao": reducao})

# ── visualização lado a lado ───────────────────────────────────────────────────
PAD = 16
LABEL_H = 36
out = Image.new("RGB", (W*2 + PAD, H + LABEL_H), (20, 20, 20))
out.paste(img, (0, LABEL_H))
out.paste(img.copy(), (W + PAD, LABEL_H))
draw = ImageDraw.Draw(out)

# cabeçalhos
draw.rectangle([0, 0, W-1, LABEL_H-1], fill=(40, 40, 40))
draw.text((8, 10), "Claude — bbox semântica (imprecisa)", fill=(220, 220, 220))
draw.rectangle([W+PAD, 0, W*2+PAD-1, LABEL_H-1], fill=(20, 50, 20))
draw.text((W+PAD+8, 10), "DINOv2 refinada  ·  896px  ·  36k gazetteer", fill=(150, 255, 150))

for obj in resultados:
    cor = obj["cor"]
    bc, br = obj["bbox"], obj["refined"]
    lw = 3

    # Esquerda — Claude
    x1c = int(bc["x1"]*W);     y1c = int(bc["y1"]*H) + LABEL_H
    x2c = int(bc["x2"]*W) - 1; y2c = int(bc["y2"]*H) + LABEL_H - 1
    for t in range(lw):
        draw.rectangle([x1c-t, y1c-t, x2c+t, y2c+t], outline=cor)
    draw.text((x1c+4, y1c+3), obj["nome"][:20], fill=cor)

    # Direita — refinada
    ox = W + PAD
    x1r = int(br["x1"]*W) + ox;     y1r = int(br["y1"]*H) + LABEL_H
    x2r = int(br["x2"]*W) + ox - 1; y2r = int(br["y2"]*H) + LABEL_H - 1
    for t in range(lw):
        draw.rectangle([x1r-t, y1r-t, x2r+t, y2r+t], outline=cor)
    tag = f"{obj['nome'][:16]} -{obj['reducao']:.0f}%"
    draw.text((x1r+4, y1r+3), tag, fill=cor)

out_path = ROOT / "poc_dinov2/output/demo_real_result.jpg"
out.save(out_path, "JPEG", quality=93)
print(f"[demo] Salvo → {out_path}")
print("="*52)
reducoes = [r["reducao"] for r in resultados]
print(f"Redução média de área: {np.mean(reducoes):.0f}%")
print(f"Tempo total encode:    {sum(time.time()-t0 for _ in [0]):.1f}s")
