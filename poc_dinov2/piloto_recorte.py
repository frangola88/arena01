"""
Piloto: recorte expandido com spotlight interno.

Compara o approach atual (foto inteira + spotlight) com o novo
(crop da bbox + padding + spotlight interno).

Uso:
  conda run -n casaiq python poc_dinov2/piloto_recorte.py [caminho_foto]
"""
import sys, json, unicodedata
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

FOTO_PADRAO = ROOT / "storage/fotos_originais/20260602_191715_158037.jpg"
FOTO = Path(sys.argv[1]) if len(sys.argv) > 1 else FOTO_PADRAO
CACHE = Path(__file__).parent / "output" / "piloto_analise_cache.json"
OUT_DIR = Path(__file__).parent / "output" / "piloto_recortes"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PALETA_BGR = [
    (0, 210, 0), (255, 100, 0), (0, 120, 255), (180, 0, 255),
    (0, 210, 210), (255, 200, 0), (255, 0, 120), (0, 160, 80),
    (0, 210, 0), (255, 100, 0), (0, 120, 255), (180, 0, 255),
]

# ── pré-processamento EXIF ────────────────────────────────────────────────────
_img_raw = Image.open(FOTO)
_img_corr = ImageOps.exif_transpose(_img_raw).convert("RGB")
import tempfile, os
_tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
FOTO_PROC = Path(_tmp.name)
_img_corr.save(FOTO_PROC, 'JPEG', quality=95)
_tmp.close()

# ── análise Claude (com cache) ────────────────────────────────────────────────
if CACHE.exists():
    print(f"[cache] Carregando análise de {CACHE.name}")
    analise = json.loads(CACHE.read_text())
else:
    print("[1] Chamando Claude para análise...")
    from core.visao_global import analisar_foto_completa
    analise = analisar_foto_completa(str(FOTO_PROC))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(analise, ensure_ascii=False, indent=2))
    print(f"    Salvo em {CACHE.name}")

objetos = analise.get("objetos", [])
objetos = [o for o in objetos if o.get("bbox_normalizada")]
print(f"    {len(objetos)} objetos com bbox\n")

# ── refinamento DINOv2 ────────────────────────────────────────────────────────
print("[2] Carregando GazetteerMatcher...")
from core.gazetteer import GazetteerMatcher
gaz = GazetteerMatcher(
    emb_path=ROOT / "embeddings.npy",
    faiss_path=ROOT / "data/gazetteer/gazetteer.faiss",
    map_path=ROOT / "data/gazetteer/index_mapping.jsonl",
    text_idx_path=ROOT / "data/gazetteer/text_index.json",
)
print(f"    {gaz._idx.ntotal:,} embeddings\n")

bboxes_refinadas = []
for obj in objetos:
    bb = obj["bbox_normalizada"]
    refined = gaz.refinar_bbox(
        str(FOTO_PROC), bb,
        nome=obj.get("nome", ""),
        cores=obj.get("cores_dominantes") or [],
        centroide=obj.get("centroide_normalizado"),
    )
    bboxes_refinadas.append(refined)

os.unlink(FOTO_PROC)

# ── carrega imagem em OpenCV ──────────────────────────────────────────────────
img_rgb = np.array(_img_corr)
img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
H, W = img_bgr.shape[:2]


# ─────────────────────────────────────────────────────────────────────────────
# Funções de renderização
# ─────────────────────────────────────────────────────────────────────────────

def ascii_nome(nome: str, maxlen: int = 28) -> str:
    s = unicodedata.normalize("NFKD", nome).encode("ascii", "ignore").decode("ascii")
    return s[:maxlen] + "..." if len(s) > maxlen else s


def _spotlight_full(img_bgr, bbox_norm, numero, nome, cor):
    """Approach ATUAL: foto inteira com spotlight."""
    x1 = max(0, int(bbox_norm["x1"] * W))
    y1 = max(0, int(bbox_norm["y1"] * H))
    x2 = min(W, int(bbox_norm["x2"] * W))
    y2 = min(H, int(bbox_norm["y2"] * H))

    out = img_bgr.copy()
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (W, H), (0, 0, 0), -1)
    out = cv2.addWeighted(out, 0.35, overlay, 0.65, 0)
    out[y1:y2, x1:x2] = img_bgr[y1:y2, x1:x2]
    cv2.rectangle(out, (x1-2, y1-2), (x2+2, y2+2), (0, 0, 0), 5)
    cv2.rectangle(out, (x1-2, y1-2), (x2+2, y2+2), cor, 3)
    cx, cy = x1 + 20, y1 + 20
    cv2.circle(out, (cx, cy), 16, (0, 0, 0), -1)
    cv2.circle(out, (cx, cy), 16, cor, 3)
    cv2.putText(out, str(numero), (cx-6 if numero<10 else cx-10, cy+6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    rod = 28
    cv2.rectangle(out, (0, H-rod), (W, H), (0, 0, 0), -1)
    cv2.putText(out, ascii_nome(nome), (6, H-7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, cor, 2, cv2.LINE_AA)
    return out


def _spotlight_crop(img_bgr, bbox_norm, numero, nome, cor, padding=0.20):
    """Approach NOVO: crop da bbox expandida + spotlight interno."""
    bw = bbox_norm["x2"] - bbox_norm["x1"]
    bh = bbox_norm["y2"] - bbox_norm["y1"]
    bbox_area = bw * bh

    # Fallback para foto inteira se bbox for muito grande (>45%)
    if bbox_area > 0.45:
        return _spotlight_full(img_bgr, bbox_norm, numero, nome, cor)

    # Expande bbox proporcionalmente às dimensões da própria bbox
    pad_x = bw * padding
    pad_y = bh * padding
    ex = {
        "x1": max(0.0, bbox_norm["x1"] - pad_x),
        "y1": max(0.0, bbox_norm["y1"] - pad_y),
        "x2": min(1.0, bbox_norm["x2"] + pad_x),
        "y2": min(1.0, bbox_norm["y2"] + pad_y),
    }

    # Pixels da região expandida
    ex1 = max(0, int(ex["x1"] * W)); ey1 = max(0, int(ex["y1"] * H))
    ex2 = min(W, int(ex["x2"] * W)); ey2 = min(H, int(ex["y2"] * H))
    crop = img_bgr[ey1:ey2, ex1:ex2].copy()
    ch, cw = crop.shape[:2]

    if cw < 10 or ch < 10:
        return _spotlight_full(img_bgr, bbox_norm, numero, nome, cor)

    # Bbox original em coordenadas do crop
    ew = ex["x2"] - ex["x1"]; eh = ex["y2"] - ex["y1"]
    ox1 = max(0, int((bbox_norm["x1"] - ex["x1"]) / ew * cw))
    oy1 = max(0, int((bbox_norm["y1"] - ex["y1"]) / eh * ch))
    ox2 = min(cw, int((bbox_norm["x2"] - ex["x1"]) / ew * cw))
    oy2 = min(ch, int((bbox_norm["y2"] - ex["y1"]) / eh * ch))

    # Escurece a margem para 12% do brilho original (vizinhos ficam quase invisíveis)
    overlay = crop.copy()
    cv2.rectangle(overlay, (0, 0), (cw, ch), (0, 0, 0), -1)
    out = cv2.addWeighted(crop, 0.12, overlay, 0.88, 0)

    # Restaura região original (spotlight)
    out[oy1:oy2, ox1:ox2] = crop[oy1:oy2, ox1:ox2]

    # Retângulo
    cv2.rectangle(out, (ox1-2, oy1-2), (ox2+2, oy2+2), (0, 0, 0), 5)
    cv2.rectangle(out, (ox1-2, oy1-2), (ox2+2, oy2+2), cor, 3)

    # Badge
    bcx, bcy = ox1 + 20, oy1 + 20
    cv2.circle(out, (bcx, bcy), 16, (0, 0, 0), -1)
    cv2.circle(out, (bcx, bcy), 16, cor, 3)
    cv2.putText(out, str(numero), (bcx-6 if numero<10 else bcx-10, bcy+6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Rodapé
    rod = 28
    cv2.rectangle(out, (0, ch-rod), (cw, ch), (0, 0, 0), -1)
    cv2.putText(out, ascii_nome(nome), (6, ch-7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, cor, 2, cv2.LINE_AA)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Gera grid comparativo: cada objeto = 1 coluna, 2 linhas (atual | novo)
# ─────────────────────────────────────────────────────────────────────────────
print("[3] Gerando grid comparativo...")

THUMB_W, THUMB_H = 480, 360
LABEL_H = 30
GAP = 6
N = len(objetos)
COLS = 4
ROWS = (N + COLS - 1) // COLS

grid_h = ROWS * (THUMB_H * 2 + LABEL_H + GAP * 3) + GAP
grid_w = COLS * (THUMB_W + GAP) + GAP
grid = np.full((grid_h, grid_w, 3), 18, dtype=np.uint8)

for i, (obj, bb_refined) in enumerate(zip(objetos, bboxes_refinadas)):
    nome = obj["nome"]
    bb_claude = obj["bbox_normalizada"]
    cor = PALETA_BGR[i % len(PALETA_BGR)]
    numero = i + 1

    col = i % COLS
    row = i // COLS
    base_x = GAP + col * (THUMB_W + GAP)
    base_y = GAP + row * (THUMB_H * 2 + LABEL_H + GAP * 3)

    # Label do objeto
    label = f"{numero}. {ascii_nome(nome, 22)}"
    cv2.rectangle(grid,
                  (base_x, base_y), (base_x + THUMB_W, base_y + LABEL_H),
                  (40, 40, 40), -1)
    cv2.putText(grid, label, (base_x + 6, base_y + 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, cor, 2, cv2.LINE_AA)

    # ── atual (topo) ──
    full_img = _spotlight_full(img_bgr, bb_refined, numero, nome, cor)
    thumb_full = cv2.resize(full_img, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)
    y0 = base_y + LABEL_H + GAP
    grid[y0:y0+THUMB_H, base_x:base_x+THUMB_W] = thumb_full
    cv2.rectangle(grid, (base_x, y0), (base_x+THUMB_W-1, y0+THUMB_H-1), (80, 80, 80), 1)
    cv2.putText(grid, "ATUAL (foto inteira)", (base_x+4, y0+14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)

    # ── novo (baixo) ──
    crop_img = _spotlight_crop(img_bgr, bb_refined, numero, nome, cor)
    # redimensiona mantendo aspecto, pad com cinza escuro
    ch_orig, cw_orig = crop_img.shape[:2]
    scale = min(THUMB_W / cw_orig, THUMB_H / ch_orig)
    nw, nh = int(cw_orig * scale), int(ch_orig * scale)
    resized = cv2.resize(crop_img, (nw, nh), interpolation=cv2.INTER_AREA)
    pad_img = np.full((THUMB_H, THUMB_W, 3), 30, dtype=np.uint8)
    ox, oy = (THUMB_W - nw) // 2, (THUMB_H - nh) // 2
    pad_img[oy:oy+nh, ox:ox+nw] = resized

    y1 = y0 + THUMB_H + GAP
    grid[y1:y1+THUMB_H, base_x:base_x+THUMB_W] = pad_img
    cv2.rectangle(grid, (base_x, y1), (base_x+THUMB_W-1, y1+THUMB_H-1), cor, 2)
    cv2.putText(grid, "NOVO (crop+padding)", (base_x+4, y1+14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, cor, 1, cv2.LINE_AA)

    # Salva crop individual também
    out_individual = OUT_DIR / f"obj_{numero:02d}_{ascii_nome(nome, 20).replace(' ', '_')}.jpg"
    cv2.imwrite(str(out_individual), crop_img)

# Salva grid
out_grid = Path(__file__).parent / "output" / "piloto_recorte_grid.jpg"
cv2.imwrite(str(out_grid), grid)

print(f"\n  Grid salvo em: {out_grid}")
print(f"  Crops individuais em: {OUT_DIR}/")
print(f"  ({N} objetos × 2 abordagens)\n")
