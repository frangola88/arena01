"""Demo completo: DINOv2 × cor × centroide — 4 colunas de comparação."""
import sys, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from core.gazetteer import GazetteerMatcher

FOTO = Path("/home/cuco/Downloads/IMG_20260324_153910599.jpg")
g = GazetteerMatcher(
    emb_path=ROOT/"embeddings.npy",
    faiss_path=ROOT/"data/gazetteer/gazetteer.faiss",
    map_path=ROOT/"data/gazetteer/index_mapping.jsonl",
    text_idx_path=ROOT/"data/gazetteer/text_index.json",
)
img = Image.open(FOTO).convert("RGB")
W, H = img.size
print(f"Foto: {W}×{H}\n")

OBJETOS = [
    {
        "nome":      "Chave de Fenda Philips cluster",
        "bbox":      {"x1":0.50,"y1":0.00,"x2":0.98,"y2":0.20},
        "centroide": {"cx":0.74,"cy":0.10},
        "cor_box":   "#FF4444",
        "cores": [
            {"hex":"#C0392B","area_pct":45},
            {"hex":"#C0C0C0","area_pct":40},
            {"hex":"#F5CBA7","area_pct":15},
        ],
    },
    {
        "nome":      "Chave de Fenda Cabo Laranja",
        "bbox":      {"x1":0.22,"y1":0.12,"x2":0.80,"y2":0.30},
        "centroide": {"cx":0.38,"cy":0.20},
        "cor_box":   "#FF9900",
        "cores": [
            {"hex":"#E06428","area_pct":50},
            {"hex":"#C0C0C0","area_pct":35},
            {"hex":"#8B4513","area_pct":15},
        ],
    },
    {
        "nome":      "Chave de Fenda Amarela Worker",
        "bbox":      {"x1":0.28,"y1":0.68,"x2":0.75,"y2":0.98},
        "centroide": {"cx":0.50,"cy":0.83},
        "cor_box":   "#FFEE00",
        "cores": [
            {"hex":"#F4D03F","area_pct":40},
            {"hex":"#1A1A1A","area_pct":35},
            {"hex":"#C0C0C0","area_pct":25},
        ],
    },
    {
        "nome":      "Pote Plastico",
        "bbox":      {"x1":0.00,"y1":0.50,"x2":0.30,"y2":0.98},
        "centroide": {"cx":0.12,"cy":0.74},
        "cor_box":   "#CC88FF",
        "cores": [
            {"hex":"#F0F0F0","area_pct":90},
            {"hex":"#D0D0D0","area_pct":10},
        ],
    },
]

MODOS = [
    ("claude",  "Claude",             "#888888"),
    ("sem",     "DINOv2 s/ cor",      "#4499FF"),
    ("cor",     "DINOv2 + cor",       "#FF9944"),
    ("full",    "DINOv2 + cor + ⊕",   "#44FF88"),
]

resultados = []
for obj in OBJETOS:
    r_sem  = g.refinar_bbox(str(FOTO), obj["bbox"])
    r_cor  = g.refinar_bbox(str(FOTO), obj["bbox"], cores=obj["cores"])
    r_full = g.refinar_bbox(str(FOTO), obj["bbox"],
                            cores=obj["cores"], centroide=obj["centroide"])

    def area(b): return (b["x2"]-b["x1"])*(b["y2"]-b["y1"])*100
    ac, as_, aco, afu = area(obj["bbox"]), area(r_sem), area(r_cor), area(r_full)
    print(f"  {obj['nome']}")
    print(f"    Claude  {ac:.1f}%  |  S/cor {as_:.1f}%  |  +cor {aco:.1f}%  |  +cor+⊕ {afu:.1f}%\n")
    resultados.append({**obj, "r_sem":r_sem, "r_cor":r_cor, "r_full":r_full})

# ── crops 4 colunas por objeto ────────────────────────────────────────────────
OUT = ROOT/"poc_dinov2/output"
PAD, LH = 8, 30

for obj, rd in zip(OBJETOS, resultados):
    bbs = {
        "claude": tuple(obj["bbox"][k] for k in ("x1","y1","x2","y2")),
        "sem":    tuple(rd["r_sem"][k]  for k in ("x1","y1","x2","y2")),
        "cor":    tuple(rd["r_cor"][k]  for k in ("x1","y1","x2","y2")),
        "full":   tuple(rd["r_full"][k] for k in ("x1","y1","x2","y2")),
    }
    ux1=max(0.0,min(b[0] for b in bbs.values())-0.03)
    uy1=max(0.0,min(b[1] for b in bbs.values())-0.03)
    ux2=min(1.0,max(b[2] for b in bbs.values())+0.03)
    uy2=min(1.0,max(b[3] for b in bbs.values())+0.03)

    frames = []
    for key, lbl, lcor in MODOS:
        crop = img.crop((int(ux1*W),int(uy1*H),int(ux2*W),int(uy2*H))).copy()
        cw,ch = crop.width,crop.height
        d = ImageDraw.Draw(crop)

        def box(coords, cor, lw=7):
            x1n,y1n,x2n,y2n = coords
            x1=int((x1n-ux1)/(ux2-ux1)*cw); y1=int((y1n-uy1)/(uy2-uy1)*ch)
            x2=int((x2n-ux1)/(ux2-ux1)*cw); y2=int((y2n-uy1)/(uy2-uy1)*ch)
            for t in range(lw): d.rectangle([x1-t,y1-t,x2+t,y2+t],outline=cor)

        # referência Claude em cinza
        box(bbs["claude"], "#444444", lw=3)
        # bbox do modo atual
        box(bbs[key], obj["cor_box"], lw=7)

        # centroide (⊕ apenas no modo full)
        if key == "full":
            ccx = int((obj["centroide"]["cx"] - ux1)/(ux2-ux1)*cw)
            ccy = int((obj["centroide"]["cy"] - uy1)/(uy2-uy1)*ch)
            r = 8
            d.ellipse([ccx-r,ccy-r,ccx+r,ccy+r], outline="#FFFFFF", width=3)
            d.ellipse([ccx-3,ccy-3,ccx+3,ccy+3], fill="#FFFFFF")

        sc = min(480/cw, 380/ch, 1.0)
        crop = crop.resize((int(cw*sc),int(ch*sc)), Image.LANCZOS)
        a = (bbs[key][2]-bbs[key][0])*(bbs[key][3]-bbs[key][1])*100
        frames.append((crop, f"{lbl}  {a:.1f}%", lcor))

    fw,fh = frames[0][0].width, frames[0][0].height
    canvas = Image.new("RGB",(fw*4+PAD*3,fh+LH),(18,18,18))
    cd = ImageDraw.Draw(canvas)
    for i,(frame,lbl,lc) in enumerate(frames):
        ox=i*(fw+PAD)
        canvas.paste(frame,(ox,LH))
        cd.rectangle([ox,0,ox+fw-1,LH-1],fill=(30,30,30))
        cd.text((ox+5,9),lbl,fill=lc)

    nome_safe = obj["nome"].replace(" ","_")
    canvas.save(OUT/f"full_{nome_safe}.jpg","JPEG",quality=93)
    print(f"  → full_{nome_safe}.jpg")
