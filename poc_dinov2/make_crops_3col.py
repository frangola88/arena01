"""Gera crops side-by-side 3 colunas por objeto: Claude / sem cor / com cor."""
from PIL import Image, ImageDraw
from pathlib import Path

FOTO = Path("/home/cuco/Downloads/IMG_20260324_153910599.jpg")
OUT  = Path(__file__).parent / "output"
img  = Image.open(FOTO).convert("RGB")
W, H = img.size

OBJETOS = [
    {"nome": "Philips_cluster",
     "claude": (0.50,0.00,0.98,0.20), "cor_box": "#FF4444",
     "sem":    (0.52,0.00,1.00,0.25),
     "com":    (0.50,0.00,0.98,0.20)},   # resultados do demo
    {"nome": "Cabo_Laranja",
     "claude": (0.22,0.12,0.80,0.30), "cor_box": "#FF9900",
     "sem":    (0.17,0.08,0.42,0.34),
     "com":    (0.17,0.08,0.44,0.34)},
    {"nome": "Chave_Amarela",
     "claude": (0.28,0.68,0.75,0.98), "cor_box": "#FFEE00",
     "sem":    (0.23,0.62,0.80,0.89),
     "com":    (0.28,0.64,0.76,0.92)},
    {"nome": "Pote_Plastico",
     "claude": (0.00,0.50,0.30,0.98), "cor_box": "#CC88FF",
     "sem":    (0.09,0.50,0.34,0.77),
     "com":    (0.09,0.50,0.34,0.77)},
]

CORES_COL = {"claude": "#888888", "sem": "#4499FF", "com": "#44FF88"}
LABELS    = {"claude": "Claude",  "sem": "DINOv2 s/ cor", "com": "DINOv2 c/ cor ★"}

for obj in OBJETOS:
    bboxes_all = {k: obj[k] for k in ("claude", "sem", "com")}

    # union region
    allcoords = list(bboxes_all.values())
    ux1 = max(0.0, min(b[0] for b in allcoords) - 0.04)
    uy1 = max(0.0, min(b[1] for b in allcoords) - 0.04)
    ux2 = min(1.0, max(b[2] for b in allcoords) + 0.04)
    uy2 = min(1.0, max(b[3] for b in allcoords) + 0.04)

    PAD, LH = 8, 28
    frames = []
    for key in ("claude", "sem", "com"):
        crop = img.crop((int(ux1*W), int(uy1*H), int(ux2*W), int(uy2*H))).copy()
        cw, ch = crop.width, crop.height
        draw = ImageDraw.Draw(crop)

        def box(x1n,y1n,x2n,y2n, cor, lw=7):
            x1=int((x1n-ux1)/(ux2-ux1)*cw); y1=int((y1n-uy1)/(uy2-uy1)*ch)
            x2=int((x2n-ux1)/(ux2-ux1)*cw); y2=int((y2n-uy1)/(uy2-uy1)*ch)
            for t in range(lw): draw.rectangle([x1-t,y1-t,x2+t,y2+t],outline=cor)

        # sempre mostrar Claude em cinza + a bbox específica colorida
        box(*bboxes_all["claude"], "#666666", lw=4)
        if key != "claude":
            box(*bboxes_all[key], obj["cor_box"], lw=7)
        else:
            box(*bboxes_all["claude"], obj["cor_box"], lw=7)

        scale = min(600/cw, 460/ch, 1.0)
        crop = crop.resize((int(cw*scale), int(ch*scale)), Image.LANCZOS)
        frames.append((crop, key))

    fw, fh = frames[0][0].width, frames[0][0].height
    canvas = Image.new("RGB", (fw*3 + PAD*2, fh + LH), (20,20,20))
    cd = ImageDraw.Draw(canvas)

    for i, (frame, key) in enumerate(frames):
        ox = i * (fw + PAD)
        canvas.paste(frame, (ox, LH))
        cd.rectangle([ox, 0, ox+fw-1, LH-1], fill=(30,30,30))
        lbl = LABELS[key]
        ac = (bboxes_all[key][2]-bboxes_all[key][0])*(bboxes_all[key][3]-bboxes_all[key][1])*100
        cd.text((ox+6, 8), f"{lbl}  {ac:.1f}%", fill=CORES_COL[key])

    canvas.save(OUT / f"cmp_{obj['nome']}.jpg", "JPEG", quality=93)
    print(f"  {obj['nome']}: claude={((bboxes_all['claude'][2]-bboxes_all['claude'][0])*(bboxes_all['claude'][3]-bboxes_all['claude'][1])*100):.1f}% → sem={((bboxes_all['sem'][2]-bboxes_all['sem'][0])*(bboxes_all['sem'][3]-bboxes_all['sem'][1])*100):.1f}% → com={((bboxes_all['com'][2]-bboxes_all['com'][0])*(bboxes_all['com'][3]-bboxes_all['com'][1])*100):.1f}%")
