"""Demo completo dos 3 alicates: 4 colunas (Claude, s/cor, +cor, +cor+⊕)."""
import sys
from pathlib import Path
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
        "nome":      "Alicate Bico Curvo",
        "bbox":      {"x1":0.06,"y1":0.18,"x2":0.30,"y2":0.62},
        "centroide": {"cx":0.18,"cy":0.40},
        "cor_box":   "#FFFF00",
        "cores": [
            {"hex":"#E8C84A","area_pct":45},
            {"hex":"#1A1A1A","area_pct":35},
            {"hex":"#909090","area_pct":20},
        ],
    },
    {
        "nome":      "Alicate Corte Diagonal",
        "bbox":      {"x1":0.28,"y1":0.14,"x2":0.58,"y2":0.65},
        "centroide": {"cx":0.43,"cy":0.39},
        "cor_box":   "#FF9900",
        "cores": [
            {"hex":"#D4A017","area_pct":50},
            {"hex":"#1A1A1A","area_pct":35},
            {"hex":"#808080","area_pct":15},
        ],
    },
    {
        "nome":      "Alicate Desencapador LAOA",
        "bbox":      {"x1":0.54,"y1":0.04,"x2":0.96,"y2":0.62},
        "centroide": {"cx":0.75,"cy":0.33},
        "cor_box":   "#00FF00",
        "cores": [
            {"hex":"#2E7D32","area_pct":50},
            {"hex":"#1A1A1A","area_pct":25},
            {"hex":"#C0C0C0","area_pct":25},
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

        box(bbs["claude"], "#444444", lw=3)
        box(bbs[key], obj["cor_box"], lw=7)

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
    canvas.save(OUT/f"ali_{nome_safe}.jpg","JPEG",quality=93)
    print(f"  → ali_{nome_safe}.jpg")
