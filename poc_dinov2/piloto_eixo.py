"""
Piloto: crop orientado pelo eixo_principal (ponta_a → ponta_b).

Para ferramentas alongadas:
  1. Calcula ângulo do eixo ponta_a→ponta_b
  2. Rotaciona a imagem para alinhar a ferramenta ao eixo horizontal
  3. Corta um retângulo centrado no eixo (comprimento + 20% padding, largura_pct + 40% padding)

Para objetos compactos (sem eixo_principal):
  Crop da bbox com margens escurecidas (12% brilho).

Uso:
  conda run -n casaiq python poc_dinov2/piloto_eixo.py [caminho_foto]
"""
import sys, json, math, unicodedata, tempfile, os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

FOTO_PADRAO = ROOT / "storage/fotos_originais/20260602_191715_158037.jpg"
FOTO = Path(sys.argv[1]) if len(sys.argv) > 1 else FOTO_PADRAO
CACHE = Path(__file__).parent / "output" / "piloto_analise_cache.json"
OUT_DIR = Path(__file__).parent / "output" / "piloto_eixo"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PALETA = [
    (0,210,0),(255,100,0),(0,120,255),(180,0,255),
    (0,210,210),(255,200,0),(255,0,120),(0,160,80),
    (0,210,0),(255,100,0),(0,120,255),(180,0,255),
]

# ── EXIF ──────────────────────────────────────────────────────────────────────
_img_raw = Image.open(FOTO)
_img_corr = ImageOps.exif_transpose(_img_raw).convert("RGB")
_tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
FOTO_PROC = Path(_tmp.name)
_img_corr.save(FOTO_PROC, 'JPEG', quality=95)
_tmp.close()

# ── análise Claude ─────────────────────────────────────────────────────────────
if CACHE.exists():
    print(f"[cache] {CACHE.name}")
    analise = json.loads(CACHE.read_text())
else:
    print("[1] Chamando Claude (schema com eixo_principal)...")
    from core.visao_global import analisar_foto_completa
    analise = analisar_foto_completa(str(FOTO_PROC))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(analise, ensure_ascii=False, indent=2))
    print(f"    Salvo em {CACHE.name}")

objetos = [o for o in analise.get("objetos", []) if o.get("bbox_normalizada")]
com_eixo = [o for o in objetos if o.get("eixo_principal")]
print(f"    {len(objetos)} objetos  |  {len(com_eixo)} com eixo_principal\n")

# Imprime o que o Claude devolveu
for i, o in enumerate(objetos, 1):
    eixo = o.get("eixo_principal")
    bb = o["bbox_normalizada"]
    if eixo:
        pa, pb = eixo["ponta_a"], eixo["ponta_b"]
        larg = eixo.get("largura_estimada_pct", "?")
        print(f"  [{i:2d}] {o['nome']:<32} eixo: ({pa['x']:.2f},{pa['y']:.2f})→({pb['x']:.2f},{pb['y']:.2f}) w={larg}%")
    else:
        print(f"  [{i:2d}] {o['nome']:<32} bbox: ({bb['x1']:.2f},{bb['y1']:.2f})→({bb['x2']:.2f},{bb['y2']:.2f})")
print()

# ── imagem ────────────────────────────────────────────────────────────────────
img_rgb = np.array(_img_corr)
img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
H, W = img_bgr.shape[:2]
os.unlink(FOTO_PROC)


def ascii_nome(nome, n=26):
    s = unicodedata.normalize("NFKD", nome).encode("ascii", "ignore").decode("ascii")
    return s[:n] + "..." if len(s) > n else s


def _adicionar_badge(crop, numero, nome, cor):
    """Badge numérico + rodapé no crop."""
    ch, cw = crop.shape[:2]
    bcx, bcy = min(22, cw-4), min(22, ch-4)
    cv2.circle(crop, (bcx, bcy), 16, (0,0,0), -1)
    cv2.circle(crop, (bcx, bcy), 16, cor, 3)
    cv2.putText(crop, str(numero),
                (bcx-6 if numero < 10 else bcx-10, bcy+6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    rod = 26
    cv2.rectangle(crop, (0, ch-rod), (cw, ch), (0,0,0), -1)
    cv2.putText(crop, ascii_nome(nome), (5, ch-7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2, cv2.LINE_AA)
    return crop


def crop_orientado(img_bgr, eixo, numero, nome, cor,
                   pad_comprimento=0.20, pad_largura=0.50):
    """Rotaciona a imagem pelo eixo da ferramenta e recorta."""
    pa, pb = eixo["ponta_a"], eixo["ponta_b"]
    larg_pct = eixo.get("largura_estimada_pct", 5)

    ax, ay = pa["x"] * W, pa["y"] * H
    bx, by = pb["x"] * W, pb["y"] * H

    cx, cy = (ax + bx) / 2.0, (ay + by) / 2.0
    comprimento = math.sqrt((bx-ax)**2 + (by-ay)**2)
    angle_deg = math.degrees(math.atan2(by - ay, bx - ax))

    largura_px = max(30, larg_pct / 100 * W)

    comp_padded = comprimento * (1 + 2 * pad_comprimento)
    larg_padded = largura_px  * (1 + 2 * pad_largura)

    # Rotaciona a imagem inteira para alinhar o eixo ao horizontal
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    rotated = cv2.warpAffine(img_bgr, M, (W, H),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(20, 20, 20))

    # Crop centrado no eixo (já horizontal após rotação)
    x1 = max(0, int(cx - comp_padded / 2))
    x2 = min(W, int(cx + comp_padded / 2))
    y1 = max(0, int(cy - larg_padded / 2))
    y2 = min(H, int(cy + larg_padded / 2))

    crop = rotated[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return None

    _adicionar_badge(crop, numero, nome, cor)
    return crop


def crop_bbox(img_bgr, bbox, numero, nome, cor, pad=0.15):
    """Fallback: crop da bbox com margens escurecidas."""
    bw = bbox["x2"] - bbox["x1"]
    bh = bbox["y2"] - bbox["y1"]
    pad_x = int(bw * pad * W)
    pad_y = int(bh * pad * H)

    x1 = max(0, int(bbox["x1"]*W) - pad_x)
    y1 = max(0, int(bbox["y1"]*H) - pad_y)
    x2 = min(W, int(bbox["x2"]*W) + pad_x)
    y2 = min(H, int(bbox["y2"]*H) + pad_y)

    escuro = cv2.addWeighted(img_bgr, 0.12, np.zeros_like(img_bgr), 0.88, 0)
    resultado = escuro.copy()
    bx1, by1 = int(bbox["x1"]*W), int(bbox["y1"]*H)
    bx2, by2 = int(bbox["x2"]*W), int(bbox["y2"]*H)
    resultado[by1:by2, bx1:bx2] = img_bgr[by1:by2, bx1:bx2]
    cv2.rectangle(resultado, (bx1-2,by1-2), (bx2+2,by2+2), (0,0,0), 5)
    cv2.rectangle(resultado, (bx1-2,by1-2), (bx2+2,by2+2), cor, 3)

    crop = resultado[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return None
    _adicionar_badge(crop, numero, nome, cor)
    return crop


# ── gera crops ────────────────────────────────────────────────────────────────
print("[2] Gerando crops...")
resultados = []

for i, obj in enumerate(objetos, 1):
    nome = obj["nome"]
    cor = PALETA[i % len(PALETA)]
    eixo = obj.get("eixo_principal")
    bbox = obj["bbox_normalizada"]

    if eixo:
        crop = crop_orientado(img_bgr, eixo, i, nome, cor)
        tag = "eixo"
    else:
        crop = crop_bbox(img_bgr, bbox, i, nome, cor)
        tag = "bbox"

    if crop is None:
        print(f"  [{i:2d}] {nome:<30} SKIP")
        continue

    fname = OUT_DIR / f"obj_{i:02d}_{ascii_nome(nome,16).replace(' ','_')}_{tag}.jpg"
    cv2.imwrite(str(fname), crop)
    print(f"  [{i:2d}] {nome:<30} [{tag}]  → {fname.name}")
    resultados.append((obj, crop, tag, cor, i))


# ── grid ──────────────────────────────────────────────────────────────────────
print("\n[3] Gerando grid...")
THUMB_W, THUMB_H = 480, 320
LABEL_H = 26
GAP = 6
N = len(resultados)
COLS = 4
ROWS = (N + COLS - 1) // COLS

gw = COLS * (THUMB_W + GAP) + GAP
gh = ROWS * (THUMB_H + LABEL_H + GAP*2) + GAP
grid = np.full((gh, gw, 3), 18, dtype=np.uint8)

for idx, (obj, crop, tag, cor, numero) in enumerate(resultados):
    col = idx % COLS
    row = idx // COLS
    bx = GAP + col*(THUMB_W+GAP)
    by = GAP + row*(THUMB_H+LABEL_H+GAP*2)

    label = f"{numero}. {ascii_nome(obj['nome'],20)}  [{tag}]"
    cv2.rectangle(grid,(bx,by),(bx+THUMB_W,by+LABEL_H),(40,40,40),-1)
    cv2.putText(grid, label, (bx+5, by+18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, cor, 2, cv2.LINE_AA)

    ch, cw = crop.shape[:2]
    scale = min(THUMB_W/cw, THUMB_H/ch)
    nw, nh = int(cw*scale), int(ch*scale)
    thumb = cv2.resize(crop, (nw,nh), interpolation=cv2.INTER_AREA)
    cell = np.full((THUMB_H,THUMB_W,3), 28, dtype=np.uint8)
    ox, oy = (THUMB_W-nw)//2, (THUMB_H-nh)//2
    cell[oy:oy+nh, ox:ox+nw] = thumb

    ty = by + LABEL_H + GAP
    grid[ty:ty+THUMB_H, bx:bx+THUMB_W] = cell
    border = cor if tag == "eixo" else (60,60,60)
    cv2.rectangle(grid,(bx,ty),(bx+THUMB_W-1,ty+THUMB_H-1), border, 2)

out_grid = Path(__file__).parent / "output" / "piloto_eixo_grid.jpg"
cv2.imwrite(str(out_grid), grid)

n_eixo = sum(1 for *_, t, _, _ in resultados if t == "eixo")
print(f"\n  Grid: {out_grid}")
print(f"  Crops: {OUT_DIR}/")
print(f"  {n_eixo}/{len(resultados)} com eixo orientado\n")
