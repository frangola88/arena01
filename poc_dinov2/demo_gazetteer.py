"""Demo end-to-end — GazetteerMatcher com gazetteer 36k + encoder 896px.

Simula o output do Claude (identificações + bboxes semânticas imprecisas)
e roda o refinamento DINOv2 em cima, comparando visualmente.

Uso: conda run -n casaiq python poc_dinov2/demo_gazetteer.py
"""
import sys, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent.parent

# ── foto e identificações simuladas do Claude ───────────────────────────────
FOTO = ROOT / "storage/fotos_originais/20260604_113939_507010.jpg"

# O que Claude diria ao ver esta foto (x1,y1,x2,y2 normalizados 0-1)
CLAUDE_OUTPUT = [
    {
        "nome":  "Chave de Boca Combinada",
        "brand": "WORKER",
        "bbox":  {"x1": 0.27, "y1": 0.10, "x2": 0.57, "y2": 0.40},
        "cor":   "#FF4444",
    },
    {
        "nome":  "Alicate Bico Longo Isolado",
        "brand": "WORKER",
        "bbox":  {"x1": 0.52, "y1": 0.15, "x2": 0.94, "y2": 0.60},
        "cor":   "#44BB44",
    },
    {
        "nome":  "Furadeira Parafusadeira",
        "brand": "MAKITA",
        "bbox":  {"x1": 0.67, "y1": 0.60, "x2": 0.98, "y2": 0.96},
        "cor":   "#4488FF",
    },
]

# ── init gazetteer ───────────────────────────────────────────────────────────
sys.path.insert(0, str(ROOT))
from core.gazetteer import GazetteerMatcher

EMB   = ROOT / "embeddings.npy"
FAISS = ROOT / "data/gazetteer/gazetteer.faiss"
MAP   = ROOT / "data/gazetteer/index_mapping.jsonl"

print(f"[demo] Inicializando GazetteerMatcher (36.681 embeddings)...")
t0 = time.time()
g = GazetteerMatcher(emb_path=EMB, faiss_path=FAISS, map_path=MAP,
                     text_idx_path=ROOT / "data/gazetteer/text_index.json")
print(f"[demo] Carregado em {time.time()-t0:.1f}s")

# ── refinamento ─────────────────────────────────────────────────────────────
img = Image.open(FOTO).convert("RGB")
W, H = img.size
print(f"[demo] Foto: {FOTO.name}  {W}×{H}")
print()

resultados = []
for obj in CLAUDE_OUTPUT:
    t1 = time.time()
    refined = g.refinar_bbox(
        str(FOTO),
        obj["bbox"],
        nome=obj["nome"],
        brand=obj["brand"],
    )
    elapsed = time.time() - t1

    bc = obj["bbox"]
    br = refined

    area_c = (bc["x2"]-bc["x1"]) * (bc["y2"]-bc["y1"])
    area_r = (br["x2"]-br["x1"]) * (br["y2"]-br["y1"])
    reducao = (1 - area_r/area_c) * 100 if area_c > 0 else 0

    print(f"  {obj['nome']} [{obj['brand']}]")
    print(f"    Claude  : ({bc['x1']:.2f},{bc['y1']:.2f}) → ({bc['x2']:.2f},{bc['y2']:.2f})  área={area_c*100:.1f}%")
    print(f"    Refinada: ({br['x1']:.2f},{br['y1']:.2f}) → ({br['x2']:.2f},{br['y2']:.2f})  área={area_r*100:.1f}%")
    print(f"    Redução : {reducao:.0f}%  tempo={elapsed:.1f}s")
    print()

    resultados.append({**obj, "refined": refined, "reducao": reducao})

# ── visualização ─────────────────────────────────────────────────────────────
# Lado a lado: esquerda = bboxes Claude, direita = bboxes refinadas
out = Image.new("RGB", (W*2 + 20, H + 60), (30, 30, 30))
out.paste(img, (0, 30))
out.paste(img, (W + 20, 30))

draw = ImageDraw.Draw(out)

# labels
draw.rectangle([0, 0, W, 28], fill=(50, 50, 50))
draw.text((10, 6), "Claude (bbox semântica)", fill="white")
draw.rectangle([W+20, 0, W*2+20, 28], fill=(50, 50, 50))
draw.text((W+30, 6), "DINOv2 refinada (896px · 36k gazetteer)", fill="white")

for obj in resultados:
    cor = obj["cor"]
    bc  = obj["bbox"]
    br  = obj["refined"]

    # Claude — esquerda (linha tracejada grossa)
    x1c = int(bc["x1"]*W); y1c = int(bc["y1"]*H)+30
    x2c = int(bc["x2"]*W); y2c = int(bc["y2"]*H)+30
    for t in range(3):
        draw.rectangle([x1c-t, y1c-t, x2c+t, y2c+t], outline=cor)
    draw.text((x1c+4, y1c+4), obj["nome"][:22], fill=cor)

    # Refinada — direita (linha sólida)
    ox = W + 20
    x1r = int(br["x1"]*W)+ox; y1r = int(br["y1"]*H)+30
    x2r = int(br["x2"]*W)+ox; y2r = int(br["y2"]*H)+30
    for t in range(3):
        draw.rectangle([x1r-t, y1r-t, x2r+t, y2r+t], outline=cor)
    label = f"{obj['nome'][:18]} -{obj['reducao']:.0f}%"
    draw.text((x1r+4, y1r+4), label, fill=cor)

out_path = ROOT / "poc_dinov2/output/demo_result.jpg"
out.save(out_path, "JPEG", quality=92)
print(f"[demo] Salvo: {out_path}")

# ── sumário ─────────────────────────────────────────────────────────────────
print("="*50)
reducoes = [r["reducao"] for r in resultados]
print(f"Redução média de área: {np.mean(reducoes):.0f}%")
print(f"Gazetteer: {g._idx.ntotal} embeddings")
print(f"Encoder:   896px → 64×64 patches")
