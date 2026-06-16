"""Demo — 3 alicates, dois com cabos idênticos (amarelo/preto)."""
import sys, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from core.gazetteer import GazetteerMatcher

FOTO = Path("/home/cuco/Downloads/IMG_20260602_185719031.jpg")
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
        "nome":    "Alicate Bico Curvo",
        "bbox":    {"x1": 0.06, "y1": 0.18, "x2": 0.30, "y2": 0.62},
        "cor_box": "#FFEE00",
        "cores":   [
            {"hex": "#E8C84A", "nome": "amarelo",  "area_pct": 45, "parte": "cabo"},
            {"hex": "#1A1A1A", "nome": "preto",    "area_pct": 35, "parte": "cabo"},
            {"hex": "#909090", "nome": "prata",    "area_pct": 20, "parte": "cabeca"},
        ],
    },
    {
        "nome":    "Alicate Corte Diagonal",
        "bbox":    {"x1": 0.28, "y1": 0.14, "x2": 0.58, "y2": 0.65},
        "cor_box": "#FF9900",
        "cores":   [
            {"hex": "#D4A017", "nome": "amarelo escuro", "area_pct": 50, "parte": "cabo"},
            {"hex": "#1A1A1A", "nome": "preto",          "area_pct": 35, "parte": "cabeca"},
            {"hex": "#808080", "nome": "cinza",          "area_pct": 15, "parte": "mola"},
        ],
    },
    {
        "nome":    "Alicate Desencapador LAOA",
        "bbox":    {"x1": 0.54, "y1": 0.04, "x2": 0.96, "y2": 0.62},
        "cor_box": "#44BB44",
        "cores":   [
            {"hex": "#2E7D32", "nome": "verde escuro", "area_pct": 50, "parte": "cabo"},
            {"hex": "#1A1A1A", "nome": "preto",        "area_pct": 25, "parte": "cabo"},
            {"hex": "#C0C0C0", "nome": "prata",        "area_pct": 25, "parte": "corpo"},
        ],
    },
]

resultados = []
for obj in OBJETOS:
    r_sem = g.refinar_bbox(str(FOTO), obj["bbox"], nome=obj["nome"])
    r_com = g.refinar_bbox(str(FOTO), obj["bbox"], nome=obj["nome"], cores=obj["cores"])

    def area(b): return (b["x2"]-b["x1"])*(b["y2"]-b["y1"])*100
    ac, as_, aco = area(obj["bbox"]), area(r_sem), area(r_com)

    print(f"  {obj['nome']}")
    print(f"    Claude   {ac:.1f}%")
    print(f"    Sem cor  {as_:.1f}%  ({as_-ac:+.1f}%)")
    print(f"    Com cor  {aco:.1f}%  ({aco-ac:+.1f}%)\n")
    resultados.append({**obj, "r_sem": r_sem, "r_com": r_com})

# ── crops 3 colunas por objeto ────────────────────────────────────────────────
OUT = ROOT / "poc_dinov2/output"
for obj, rd in zip(OBJETOS, resultados):
    bbs = {"claude": tuple(obj["bbox"][k] for k in ("x1","y1","x2","y2")),
           "sem":    tuple(rd["r_sem"][k]  for k in ("x1","y1","x2","y2")),
           "com":    tuple(rd["r_com"][k]  for k in ("x1","y1","x2","y2"))}

    ux1=max(0.0,min(b[0] for b in bbs.values())-0.04)
    uy1=max(0.0,min(b[1] for b in bbs.values())-0.04)
    ux2=min(1.0,max(b[2] for b in bbs.values())+0.04)
    uy2=min(1.0,max(b[3] for b in bbs.values())+0.04)

    PAD, LH = 8, 30
    frames = []
    for key, lcor, lbl in [
        ("claude","#888888","Claude"),
        ("sem",   "#4499FF","DINOv2 s/ cor"),
        ("com",   "#44FF88","DINOv2 c/ cor ★"),
    ]:
        crop = img.crop((int(ux1*W),int(uy1*H),int(ux2*W),int(uy2*H))).copy()
        cw,ch = crop.width, crop.height
        d = ImageDraw.Draw(crop)

        def box(coords, cor, lw=7):
            x1n,y1n,x2n,y2n = coords
            x1=int((x1n-ux1)/(ux2-ux1)*cw); y1=int((y1n-uy1)/(uy2-uy1)*ch)
            x2=int((x2n-ux1)/(ux2-ux1)*cw); y2=int((y2n-uy1)/(uy2-uy1)*ch)
            for t in range(lw): d.rectangle([x1-t,y1-t,x2+t,y2+t],outline=cor)

        box(bbs["claude"], "#555555", lw=4)
        box(bbs[key], obj["cor_box"], lw=7)

        sc = min(560/cw, 420/ch, 1.0)
        crop = crop.resize((int(cw*sc),int(ch*sc)), Image.LANCZOS)
        a = ((bbs[key][2]-bbs[key][0])*(bbs[key][3]-bbs[key][1]))*100
        frames.append((crop, f"{lbl}  {a:.1f}%", lcor))

    fw,fh = frames[0][0].width, frames[0][0].height
    canvas = Image.new("RGB",(fw*3+PAD*2,fh+LH),(18,18,18))
    cd = ImageDraw.Draw(canvas)
    for i,(frame,lbl,lc) in enumerate(frames):
        ox=i*(fw+PAD)
        canvas.paste(frame,(ox,LH))
        cd.rectangle([ox,0,ox+fw-1,LH-1],fill=(30,30,30))
        cd.text((ox+6,9),lbl,fill=lc)

    nome_safe = obj["nome"].replace(" ","_")
    canvas.save(OUT/f"ali_{nome_safe}.jpg","JPEG",quality=93)
    print(f"  → ali_{nome_safe}.jpg")
