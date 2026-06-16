from PIL import Image, ImageDraw
from pathlib import Path

FOTO = Path("/home/cuco/Downloads/IMG_20260324_153910599.jpg")
OUT  = Path(__file__).parent / "output"
img  = Image.open(FOTO).convert("RGB")
W, H = img.size

resultados = [
    {"nome": "Chave_Philips_topo",  "cor": "#FF4444",
     "claude":  (0.50, 0.00, 0.98, 0.20),
     "refined": (0.52, 0.00, 1.00, 0.25)},
    {"nome": "Cabo_Laranja",        "cor": "#FF9900",
     "claude":  (0.22, 0.12, 0.80, 0.30),
     "refined": (0.17, 0.08, 0.42, 0.34)},
    {"nome": "Regua_30cm",          "cor": "#AADDFF",
     "claude":  (0.04, 0.27, 0.98, 0.42),
     "refined": (0.36, 0.31, 0.97, 0.47)},
    {"nome": "Chave_Amarela",       "cor": "#FFEE00",
     "claude":  (0.28, 0.68, 0.75, 0.98),
     "refined": (0.23, 0.62, 0.80, 0.89)},
    {"nome": "Pote_Plastico",       "cor": "#CC88FF",
     "claude":  (0.00, 0.50, 0.30, 0.98),
     "refined": (0.09, 0.50, 0.34, 0.77)},
]

for r in resultados:
    cx1,cy1,cx2,cy2 = r["claude"]
    rx1,ry1,rx2,ry2 = r["refined"]

    # crop = union das duas bboxes + margem
    ux1 = max(0.0, min(cx1,rx1) - 0.04)
    uy1 = max(0.0, min(cy1,ry1) - 0.04)
    ux2 = min(1.0, max(cx2,rx2) + 0.04)
    uy2 = min(1.0, max(cy2,ry2) + 0.04)

    crop = img.crop((int(ux1*W), int(uy1*H), int(ux2*W), int(uy2*H))).copy()
    cw, ch = crop.width, crop.height
    draw = ImageDraw.Draw(crop)

    def box(x1n, y1n, x2n, y2n, cor, lw=8):
        x1 = int((x1n - ux1) / (ux2 - ux1) * cw)
        y1 = int((y1n - uy1) / (uy2 - uy1) * ch)
        x2 = int((x2n - ux1) / (ux2 - ux1) * cw)
        y2 = int((y2n - uy1) / (uy2 - uy1) * ch)
        for t in range(lw):
            draw.rectangle([x1-t, y1-t, x2+t, y2+t], outline=cor)

    box(cx1, cy1, cx2, cy2, r["cor"])           # Claude
    box(rx1, ry1, rx2, ry2, "#00FF44", lw=5)   # Refinada

    scale = min(900/cw, 680/ch, 1.0)
    crop = crop.resize((int(cw*scale), int(ch*scale)), Image.LANCZOS)
    p = OUT / f"crop_{r['nome']}.jpg"
    crop.save(p, "JPEG", quality=93)

    ac = (cx2-cx1)*(cy2-cy1)*100
    ar = (rx2-rx1)*(ry2-ry1)*100
    print(f"{r['nome']:25s}  Claude={ac:.1f}%  Refined={ar:.1f}%  ({ar-ac:+.1f}%)")

print("done")
