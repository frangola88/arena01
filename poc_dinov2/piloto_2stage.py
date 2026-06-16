"""
Piloto 2-stage + gatilho 3º — bbox com spotlight.

Stage 1 — foto inteira (cache): Claude → bbox_normalizada
Stage 2 — crop generoso: Claude → bbox refinada dentro do crop
Stage 3 — ativado se qual='multiplos' ou conf < 0.70: sub-crop mais justo

Custo: 2 chamadas/objeto (caso típico), 3 (caso difícil).

Uso:
  conda run -n casaiq python poc_dinov2/piloto_2stage.py [caminho_foto]
"""
import sys, json, unicodedata, tempfile, os
from io import BytesIO
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

FOTO_PADRAO = ROOT / "storage/fotos_originais/20260602_191715_158037.jpg"
FOTO = Path(sys.argv[1]) if len(sys.argv) > 1 else FOTO_PADRAO
CACHE_S1 = Path(__file__).parent / "output" / "piloto_analise_cache.json"
OUT_DIR   = Path(__file__).parent / "output" / "piloto_2stage"
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

img_rgb = np.array(_img_corr)
img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
H, W = img_bgr.shape[:2]
os.unlink(FOTO_PROC)

# ── Stage 1: carrega cache ────────────────────────────────────────────────────
if CACHE_S1.exists():
    print(f"[S1] cache: {CACHE_S1.name}")
    analise_s1 = json.loads(CACHE_S1.read_text())
else:
    print("[S1] Chamando Claude (foto inteira)...")
    from core.visao_global import analisar_foto_completa
    _tmp2 = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    _img_corr.save(_tmp2.name, 'JPEG', quality=95); _tmp2.close()
    analise_s1 = analisar_foto_completa(_tmp2.name)
    os.unlink(_tmp2.name)
    CACHE_S1.write_text(json.dumps(analise_s1, ensure_ascii=False, indent=2))

objetos_s1 = [o for o in analise_s1.get("objetos", []) if o.get("bbox_normalizada")]
print(f"    {len(objetos_s1)} objetos com bbox\n")

# ── Prompt stage 2/3 ─────────────────────────────────────────────────────────
PROMPT_CROP = """Esta imagem é um recorte de uma foto de inventário doméstico.
Identifique a ferramenta ou objeto PRINCIPAL e responda APENAS com JSON válido:

{
  "nome": "nome simples do objeto principal",
  "bbox_normalizada": {"x1": float, "y1": float, "x2": float, "y2": float},
  "qualidade_crop": "ok",
  "confianca": float,
  "observacao": "string curta"
}

Regras:
- bbox_normalizada: retângulo da ferramenta PRINCIPAL em coords 0.0-1.0 DESTE CROP
- qualidade_crop:
    "ok"         = objeto único e completo dominando o frame
    "multiplos"  = mais de um objeto claramente visível
    "incompleto" = objeto cortado pela borda
- confianca: 0.0-1.0
- observacao: frase curta (ex: "formão completo isolado", "dois ponteiros presentes")"""


# ── Rembg singleton ──────────────────────────────────────────────────────────
_rembg_session = None

def _get_rembg():
    global _rembg_session
    if _rembg_session is None:
        try:
            from rembg import new_session
            print("[rembg] carregando u2net...", flush=True)
            _rembg_session = new_session("u2net")
            print("[rembg] pronto", flush=True)
        except Exception as e:
            print(f"[rembg] indisponível: {e}")
            _rembg_session = False
    return None if _rembg_session is False else _rembg_session


def _matting_blend(img_bgr, bbox, escuro):
    """Blends objeto com fundo escuro usando alpha do rembg. None = fallback."""
    session = _get_rembg()
    if session is None:
        return None
    from rembg import remove
    bx1 = max(0, int(bbox["x1"]*W)); by1 = max(0, int(bbox["y1"]*H))
    bx2 = min(W, int(bbox["x2"]*W)); by2 = min(H, int(bbox["y2"]*H))
    roi = img_bgr[by1:by2, bx1:bx2]
    if roi.size == 0:
        return None
    try:
        roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        buf = BytesIO(); roi_pil.save(buf, "PNG")
        alpha = np.array(Image.open(BytesIO(
            remove(buf.getvalue(), session=session)
        )).convert("RGBA"))[:, :, 3].astype(np.float32) / 255.0
    except Exception as e:
        print(f"  [matting] falhou: {e}")
        return None
    result = escuro.copy()
    a3 = alpha[:, :, np.newaxis]
    result[by1:by2, bx1:bx2] = (
        roi.astype(np.float32) * a3 +
        escuro[by1:by2, bx1:bx2].astype(np.float32) * (1 - a3)
    ).astype(np.uint8)
    return result


# ── Helpers ───────────────────────────────────────────────────────────────────
def ascii_nome(nome, n=24):
    s = unicodedata.normalize("NFKD", nome).encode("ascii","ignore").decode("ascii")
    return s[:n]+"..." if len(s) > n else s


def _crop_generoso(obj, padding=0.30):
    """Crop com padding generoso para dar contexto ao Claude."""
    bbox = obj["bbox_normalizada"]
    bw = (bbox["x2"]-bbox["x1"])*W
    bh = (bbox["y2"]-bbox["y1"])*H
    px1 = max(0, int(bbox["x1"]*W - bw*padding))
    py1 = max(0, int(bbox["y1"]*H - bh*padding))
    px2 = min(W, int(bbox["x2"]*W + bw*padding))
    py2 = min(H, int(bbox["y2"]*H + bh*padding))
    return img_bgr[py1:py2, px1:px2].copy(), (px1, py1, px2, py2)


def _mapear_para_original(resultado_crop, origem):
    """Mapeia bbox do crop de volta para coords da imagem original."""
    px1, py1, px2, py2 = origem
    cw, ch = px2-px1, py2-py1
    bb = resultado_crop.get("bbox_normalizada", {})
    return {
        "x1": (px1 + bb.get("x1",0)*cw) / W,
        "y1": (py1 + bb.get("y1",0)*ch) / H,
        "x2": (px1 + bb.get("x2",1)*cw) / W,
        "y2": (py1 + bb.get("y2",1)*ch) / H,
    }


def _crop_bbox_spotlight(bbox_orig, numero, nome, cor, pad=0.10):
    """Crop com matting (rembg) + fundo escuro + badge. Fallback para spotlight."""
    bw = (bbox_orig["x2"]-bbox_orig["x1"])*W
    bh = (bbox_orig["y2"]-bbox_orig["y1"])*H
    x1 = max(0, int(bbox_orig["x1"]*W - bw*pad))
    y1 = max(0, int(bbox_orig["y1"]*H - bh*pad))
    x2 = min(W, int(bbox_orig["x2"]*W + bw*pad))
    y2 = min(H, int(bbox_orig["y2"]*H + bh*pad))

    escuro = cv2.addWeighted(img_bgr, 0.10, np.zeros_like(img_bgr), 0.90, 0)
    out = _matting_blend(img_bgr, bbox_orig, escuro)
    if out is None:
        out = escuro.copy()
        bx1,by1 = int(bbox_orig["x1"]*W), int(bbox_orig["y1"]*H)
        bx2,by2 = int(bbox_orig["x2"]*W), int(bbox_orig["y2"]*H)
        out[by1:by2, bx1:bx2] = img_bgr[by1:by2, bx1:bx2]
    crop = out[y1:y2, x1:x2].copy()

    if crop.size == 0:
        return None

    ch, cw = crop.shape[:2]
    bcx, bcy = min(22, cw-4), min(22, ch-4)
    cv2.circle(crop, (bcx,bcy), 16, (0,0,0), -1)
    cv2.circle(crop, (bcx,bcy), 16, cor, 3)
    cv2.putText(crop, str(numero),
                (bcx-6 if numero<10 else bcx-10, bcy+6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    rod = 26
    cv2.rectangle(crop, (0,ch-rod),(cw,ch),(0,0,0),-1)
    cv2.putText(crop, ascii_nome(nome), (5,ch-7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2, cv2.LINE_AA)
    return crop


def _chamar_claude_crop(crop_bgr):
    """Salva crop em temp, chama Claude, retorna dict parseado."""
    from core.llm import chamar_visao, extrair_json
    tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    cv2.imwrite(tmp.name, crop_bgr)
    tmp.close()
    try:
        texto, _ = chamar_visao(PROMPT_CROP, tmp.name)
        return extrair_json(texto)
    except Exception as e:
        return {"qualidade_crop": "erro", "observacao": str(e), "confianca": 0}
    finally:
        os.unlink(tmp.name)


# ── Pipeline principal ────────────────────────────────────────────────────────
print("[S2+S3] Refinando objetos...\n")

resultados = []
total_s2 = total_s3 = 0

for i, obj in enumerate(objetos_s1, 1):
    nome_s1 = obj["nome"]
    cor = PALETA[i % len(PALETA)]

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    crop_rough, origem = _crop_generoso(obj, padding=0.30)
    if crop_rough.size == 0:
        print(f"  [{i:2d}] {nome_s1:<30} SKIP (crop vazio)")
        continue

    r2 = _chamar_claude_crop(crop_rough)
    total_s2 += 1
    qual2 = r2.get("qualidade_crop", "erro")
    conf2 = r2.get("confianca", 0)
    nome_s2 = r2.get("nome", nome_s1)
    bbox2 = _mapear_para_original(r2, origem)

    precisa_s3 = (qual2 == "multiplos") or (conf2 < 0.70)
    stage_usado = "S2"

    print(f"  [{i:2d}] {nome_s1:<28} S2→ qual={qual2:<11} conf={conf2:.2f}  nome='{nome_s2}'", end="")

    # ── Stage 3 (gatilho) ────────────────────────────────────────────────────
    nome_final, bbox_final = nome_s2, bbox2
    if precisa_s3:
        print(f"  ⚠ → S3", end="")
        crop_s3, origem_s3 = _crop_generoso(
            {"bbox_normalizada": bbox2},
            padding=0.08,
        )
        if crop_s3.size > 0:
            r3 = _chamar_claude_crop(crop_s3)
            total_s3 += 1
            qual3 = r3.get("qualidade_crop", "erro")
            conf3 = r3.get("confianca", 0)
            nome_final = r3.get("nome", nome_s2)
            bbox_final = _mapear_para_original(r3, origem_s3)
            print(f" qual={qual3} conf={conf3:.2f}", end="")
        stage_usado = "S3"

    print()

    # ── Crop final ────────────────────────────────────────────────────────────
    crop_final = _crop_bbox_spotlight(bbox_final, i, nome_final, cor)
    if crop_final is None:
        print(f"         SKIP (crop_final vazio)")
        continue

    tag = f"{stage_usado}_bbox"
    fname = OUT_DIR / f"obj_{i:02d}_{ascii_nome(nome_final,14).replace(' ','_')}_{tag}.jpg"
    cv2.imwrite(str(fname), crop_final)
    resultados.append((nome_final, crop_final, tag, cor, i, qual2 if stage_usado=="S2" else qual3))


# ── Grid ──────────────────────────────────────────────────────────────────────
print(f"\n[Grid] {len(resultados)} crops | S2={total_s2} calls | S3={total_s3} calls")

THUMB_W, THUMB_H = 480, 320
LABEL_H = 26
GAP = 6
N = len(resultados)
COLS = 4
ROWS = (N + COLS - 1) // COLS

gw = COLS*(THUMB_W+GAP)+GAP
gh = ROWS*(THUMB_H+LABEL_H+GAP*2)+GAP
grid = np.full((gh, gw, 3), 18, dtype=np.uint8)

for idx, (nome, crop, tag, cor, numero, qual) in enumerate(resultados):
    col = idx%COLS; row = idx//COLS
    bx = GAP+col*(THUMB_W+GAP)
    by = GAP+row*(THUMB_H+LABEL_H+GAP*2)

    qual_cor = (0,200,0) if qual=="ok" else (0,180,255) if qual=="incompleto" else (0,80,255)
    label = f"{numero}. {ascii_nome(nome,18)}  [{tag}]"
    cv2.rectangle(grid,(bx,by),(bx+THUMB_W,by+LABEL_H),(30,30,30),-1)
    cv2.putText(grid, label, (bx+5,by+18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, cor, 2, cv2.LINE_AA)
    cv2.circle(grid, (bx+THUMB_W-12, by+13), 7, qual_cor, -1)

    ch, cw = crop.shape[:2]
    scale = min(THUMB_W/cw, THUMB_H/ch)
    nw, nh = int(cw*scale), int(ch*scale)
    thumb = cv2.resize(crop, (nw,nh), interpolation=cv2.INTER_AREA)
    cell = np.full((THUMB_H,THUMB_W,3), 28, dtype=np.uint8)
    ox, oy = (THUMB_W-nw)//2, (THUMB_H-nh)//2
    cell[oy:oy+nh, ox:ox+nw] = thumb

    ty = by+LABEL_H+GAP
    grid[ty:ty+THUMB_H, bx:bx+THUMB_W] = cell
    cv2.rectangle(grid,(bx,ty),(bx+THUMB_W-1,ty+THUMB_H-1),cor,2)

out_grid = Path(__file__).parent/"output"/"piloto_2stage_grid.jpg"
cv2.imwrite(str(out_grid), grid)
print(f"Grid: {out_grid}")
print(f"Crops: {OUT_DIR}/")
