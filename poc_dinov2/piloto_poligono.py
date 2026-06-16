"""
Piloto: crop orientado pelo polígono do contorno_aproximado.

Para cada objeto:
  1. Usa o polígono como máscara — pixels fora = quase preto (8%)
  2. Crop ao bounding rect do polígono + 10% de padding
  3. Salva individualmente + grid comparativo

Uso:
  conda run -n casaiq python poc_dinov2/piloto_poligono.py [caminho_foto]
"""
import sys, json, unicodedata, tempfile, os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

FOTO_PADRAO = ROOT / "storage/fotos_originais/20260602_191715_158037.jpg"
FOTO = Path(sys.argv[1]) if len(sys.argv) > 1 else FOTO_PADRAO
CACHE = Path(__file__).parent / "output" / "piloto_analise_cache.json"
OUT_DIR = Path(__file__).parent / "output" / "piloto_poligono"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PALETA_BGR = [
    (0, 210, 0), (255, 100, 0), (0, 120, 255), (180, 0, 255),
    (0, 210, 210), (255, 200, 0), (255, 0, 120), (0, 160, 80),
    (0, 210, 0), (255, 100, 0), (0, 120, 255), (180, 0, 255),
]

# ── EXIF ──────────────────────────────────────────────────────────────────────
_img_raw = Image.open(FOTO)
_img_corr = ImageOps.exif_transpose(_img_raw).convert("RGB")
_tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
FOTO_PROC = Path(_tmp.name)
_img_corr.save(FOTO_PROC, 'JPEG', quality=95)
_tmp.close()

# ── análise Claude (com cache) ────────────────────────────────────────────────
if CACHE.exists():
    print(f"[cache] {CACHE.name}")
    analise = json.loads(CACHE.read_text())
else:
    print("[1] Chamando Claude (novo schema com contorno_aproximado)...")
    from core.visao_global import analisar_foto_completa
    analise = analisar_foto_completa(str(FOTO_PROC))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(analise, ensure_ascii=False, indent=2))
    print(f"    Salvo em {CACHE.name}")

objetos = [o for o in analise.get("objetos", []) if o.get("bbox_normalizada")]
print(f"    {len(objetos)} objetos com bbox")
com_poly = sum(1 for o in objetos if o.get("contorno_aproximado"))
print(f"    {com_poly} com contorno_aproximado\n")

# ── imagem OpenCV ─────────────────────────────────────────────────────────────
img_rgb = np.array(_img_corr)
img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
H, W = img_bgr.shape[:2]
os.unlink(FOTO_PROC)


def ascii_nome(nome: str, n: int = 28) -> str:
    s = unicodedata.normalize("NFKD", nome).encode("ascii", "ignore").decode("ascii")
    return s[:n] + "..." if len(s) > n else s


def poly_to_px(pts, W, H):
    """Converte lista de {x,y} normalizados para array numpy de pixels."""
    return np.array([[int(p["x"] * W), int(p["y"] * H)] for p in pts], dtype=np.int32)


def crop_por_poligono(img_bgr, poly_norm, bbox_norm, numero, nome, cor, padding=0.12):
    """
    Crop orientado ao polígono:
    1. Máscara: pixels fora do polígono → 8% brilho
    2. Bounding rect do polígono + padding → crop
    3. Retorna imagem cropada com máscara aplicada
    """
    h, w = img_bgr.shape[:2]

    if poly_norm and len(poly_norm) >= 3:
        pts = poly_to_px(poly_norm, w, h)

        # Máscara binária do polígono
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)

        # Aplica máscara: fora do polígono = 8% de brilho
        overlay = np.zeros_like(img_bgr)
        resultado = cv2.addWeighted(img_bgr, 0.08, overlay, 0.92, 0)
        resultado[mask == 255] = img_bgr[mask == 255]

        # Borda colorida do polígono
        cv2.polylines(resultado, [pts], isClosed=True, color=(0, 0, 0), thickness=5)
        cv2.polylines(resultado, [pts], isClosed=True, color=cor, thickness=3)

        # Bounding rect do polígono para o crop
        rx, ry, rw, rh = cv2.boundingRect(pts)

        # Padding baseado nas dimensões do polígono bbox
        pad_x = int(rw * padding)
        pad_y = int(rh * padding)
        cx1 = max(0, rx - pad_x)
        cy1 = max(0, ry - pad_y)
        cx2 = min(w, rx + rw + pad_x)
        cy2 = min(h, ry + rh + pad_y)

    else:
        # Fallback para bbox quando não tem polígono
        cx1 = max(0, int(bbox_norm["x1"] * w))
        cy1 = max(0, int(bbox_norm["y1"] * h))
        cx2 = min(w, int(bbox_norm["x2"] * w))
        cy2 = min(h, int(bbox_norm["y2"] * h))

        # Escurece tudo, restaura bbox
        overlay = np.zeros_like(img_bgr)
        resultado = cv2.addWeighted(img_bgr, 0.12, overlay, 0.88, 0)
        resultado[cy1:cy2, cx1:cx2] = img_bgr[cy1:cy2, cx1:cx2]
        bx1, by1 = cx1, cy1; bx2, by2 = cx2, cy2
        cv2.rectangle(resultado, (bx1-2, by1-2), (bx2+2, by2+2), (0, 0, 0), 5)
        cv2.rectangle(resultado, (bx1-2, by1-2), (bx2+2, by2+2), cor, 3)

    crop = resultado[cy1:cy2, cx1:cx2]
    ch, cw = crop.shape[:2]
    if cw < 5 or ch < 5:
        return None

    # Badge no crop
    bcx = min(25, cw - 5)
    bcy = min(25, ch - 5)
    cv2.circle(crop, (bcx, bcy), 16, (0, 0, 0), -1)
    cv2.circle(crop, (bcx, bcy), 16, cor, 3)
    cv2.putText(crop, str(numero), (bcx-6 if numero < 10 else bcx-10, bcy+6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Rodapé
    rod = 26
    cv2.rectangle(crop, (0, ch-rod), (cw, ch), (0, 0, 0), -1)
    cv2.putText(crop, ascii_nome(nome), (5, ch-7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2, cv2.LINE_AA)

    return crop


# ── gera crops individuais ────────────────────────────────────────────────────
print("[2] Gerando crops por polígono...")

resultados = []
for i, obj in enumerate(objetos):
    nome = obj["nome"]
    cor = PALETA_BGR[i % len(PALETA_BGR)]
    numero = i + 1
    poly = obj.get("contorno_aproximado")
    bbox = obj["bbox_normalizada"]
    tem_poly = bool(poly and len(poly) >= 3)

    crop = crop_por_poligono(img_bgr, poly, bbox, numero, nome, cor)
    if crop is None:
        print(f"  [{numero:2d}] {nome:<30} SKIP (crop vazio)")
        continue

    tag = "poly" if tem_poly else "bbox"
    fname = OUT_DIR / f"obj_{numero:02d}_{ascii_nome(nome, 18).replace(' ','_')}_{tag}.jpg"
    cv2.imwrite(str(fname), crop)
    print(f"  [{numero:2d}] {nome:<30} {'✓ polígono' if tem_poly else '○ fallback bbox'}  → {fname.name}")
    resultados.append((obj, crop, tem_poly, cor, numero))


# ── grid comparativo ──────────────────────────────────────────────────────────
print("\n[3] Gerando grid...")

THUMB_W, THUMB_H = 480, 360
LABEL_H = 28
GAP = 6
N = len(resultados)
COLS = 4
ROWS = (N + COLS - 1) // COLS

grid_h = ROWS * (THUMB_H + LABEL_H + GAP * 2) + GAP
grid_w = COLS * (THUMB_W + GAP) + GAP
grid = np.full((grid_h, grid_w, 3), 18, dtype=np.uint8)

for idx, (obj, crop, tem_poly, cor, numero) in enumerate(resultados):
    col = idx % COLS
    row = idx // COLS
    bx = GAP + col * (THUMB_W + GAP)
    by = GAP + row * (THUMB_H + LABEL_H + GAP * 2)

    # Label
    label = f"{numero}. {ascii_nome(obj['nome'], 20)}  {'[poly]' if tem_poly else '[bbox]'}"
    cv2.rectangle(grid, (bx, by), (bx + THUMB_W, by + LABEL_H), (40, 40, 40), -1)
    cv2.putText(grid, label, (bx + 5, by + 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, cor, 2, cv2.LINE_AA)

    # Thumbnail mantendo aspecto
    ch, cw = crop.shape[:2]
    scale = min(THUMB_W / cw, THUMB_H / ch)
    nw, nh = int(cw * scale), int(ch * scale)
    thumb = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    cell = np.full((THUMB_H, THUMB_W, 3), 28, dtype=np.uint8)
    ox, oy = (THUMB_W - nw) // 2, (THUMB_H - nh) // 2
    cell[oy:oy+nh, ox:ox+nw] = thumb

    ty = by + LABEL_H + GAP
    grid[ty:ty+THUMB_H, bx:bx+THUMB_W] = cell
    border_col = cor if tem_poly else (60, 60, 60)
    cv2.rectangle(grid, (bx, ty), (bx+THUMB_W-1, ty+THUMB_H-1), border_col, 2)

out_grid = Path(__file__).parent / "output" / "piloto_poligono_grid.jpg"
cv2.imwrite(str(out_grid), grid)

print(f"\n  Grid: {out_grid}")
print(f"  Crops: {OUT_DIR}/")
poly_count = sum(1 for _, _, tp, _, _ in resultados if tp)
print(f"  {poly_count}/{len(resultados)} com polígono real\n")
