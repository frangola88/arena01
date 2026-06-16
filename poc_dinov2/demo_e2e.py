"""
Demo end-to-end REAL — Claude API → GazetteerMatcher → visualização.

Fluxo:
  1. Chama analisar_foto_completa() com foto real (Claude API).
  2. Imprime cada objeto: nome, bbox, centroide, cores.
  3. Roda refinar_bbox() com 3 sinais (DINOv2 + cor + centroide).
  4. Gera imagem comparativa: esquerda=Claude, direita=refinada.

Uso:
  conda run -n casaiq python poc_dinov2/demo_e2e.py [caminho_foto]
"""
import sys, time, tempfile, os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── foto ────────────────────────────────────────────────────────────────────
FOTO_PADRAO = ROOT / "storage/fotos_originais/20260602_191715_158037.jpg"
FOTO = Path(sys.argv[1]) if len(sys.argv) > 1 else FOTO_PADRAO

if not FOTO.exists():
    print(f"[erro] Foto não encontrada: {FOTO}")
    sys.exit(1)

print(f"\n{'='*60}")
print(f"  CasaIQ — Demo end-to-end real")
print(f"  Foto: {FOTO.name}")
print(f"{'='*60}\n")

# ── pré-processamento EXIF ───────────────────────────────────────────────────
# Corrige orientação antes de enviar ao Claude e ao DINOv2, garantindo que
# todos os módulos trabalhem no mesmo sistema de coordenadas.
_img_raw = Image.open(FOTO)
_img_corr = ImageOps.exif_transpose(_img_raw).convert("RGB")
_w_raw, _h_raw = _img_raw.size
_w_cor, _h_cor = _img_corr.size
if (_w_raw, _h_raw) != (_w_cor, _h_cor):
    print(f"  EXIF corrigido: {_w_raw}×{_h_raw} → {_w_cor}×{_h_cor} (orientação aplicada)\n")
_tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
FOTO_PROC = Path(_tmp.name)
_img_corr.save(FOTO_PROC, 'JPEG', quality=95)
_tmp.close()

# ── 1. Claude: análise da foto inteira ──────────────────────────────────────
from core.visao_global import analisar_foto_completa

print("[1/3] Chamando Claude para análise da foto...")
t0 = time.time()
analise = analisar_foto_completa(str(FOTO_PROC))
t_claude = time.time() - t0

objetos = analise.get("objetos", [])
contexto = analise.get("contexto_da_cena", "")
fundo = analise.get("fundo", "")

print(f"      Tempo: {t_claude:.1f}s")
print(f"      Contexto: {contexto}")
print(f"      Fundo: {fundo}")
print(f"      Objetos identificados: {len(objetos)}\n")

if not objetos:
    print("[erro] Claude não identificou objetos. Abortando.")
    sys.exit(1)

# Imprime detalhes de cada objeto
for i, obj in enumerate(objetos, 1):
    bb = obj.get("bbox_normalizada")
    ct = obj.get("centroide_normalizado")
    cores = obj.get("cores_dominantes") or []
    print(f"  [{i}] {obj['nome']}  (confiança={obj.get('confianca',0):.2f})")
    if bb:
        area = (bb["x2"]-bb["x1"])*(bb["y2"]-bb["y1"])*100
        print(f"       bbox: ({bb['x1']:.2f},{bb['y1']:.2f})→({bb['x2']:.2f},{bb['y2']:.2f})  área={area:.1f}%")
    else:
        print(f"       bbox: None (sem localização)")
    if ct:
        print(f"       centroide: ({ct['cx']:.2f}, {ct['cy']:.2f})")
    if cores:
        hexes = ", ".join(f"{c['hex']} {c['nome']} {c['area_pct']}%" for c in cores[:3])
        print(f"       cores: {hexes}")
    print()

# ── 2. GazetteerMatcher: init + refinamento ──────────────────────────────────
print("[2/3] Inicializando GazetteerMatcher...")
from core.gazetteer import GazetteerMatcher

t1 = time.time()
gaz = GazetteerMatcher(
    emb_path=ROOT / "embeddings.npy",
    faiss_path=ROOT / "data/gazetteer/gazetteer.faiss",
    map_path=ROOT / "data/gazetteer/index_mapping.jsonl",
    text_idx_path=ROOT / "data/gazetteer/text_index.json",
)
print(f"      Carregado em {time.time()-t1:.1f}s  ({gaz._idx.ntotal:,} embeddings)\n")

resultados = []
objetos_com_bbox = [o for o in objetos if o.get("bbox_normalizada")]

if not objetos_com_bbox:
    print("[aviso] Nenhum objeto com bbox — sem refinamento possível.")
    sys.exit(0)

print(f"      Refinando {len(objetos_com_bbox)} objetos com bbox...")
for obj in objetos_com_bbox:
    bb    = obj["bbox_normalizada"]
    nome  = obj.get("nome", "")
    brand = obj.get("marca", "")
    cores = obj.get("cores_dominantes") or []
    ct    = obj.get("centroide_normalizado") or None

    t2 = time.time()
    refined = gaz.refinar_bbox(
        str(FOTO_PROC), bb,
        nome=nome, brand=brand,
        cores=cores, centroide=ct,
    )
    elapsed = time.time() - t2

    area_c = (bb["x2"]-bb["x1"])*(bb["y2"]-bb["y1"])*100
    area_r = (refined["x2"]-refined["x1"])*(refined["y2"]-refined["y1"])*100
    delta  = area_c - area_r

    sinal  = "↓" if delta > 0 else ("↑" if delta < 0 else "=")
    print(f"  ✓ {nome[:32]:<32}  "
          f"Claude={area_c:.1f}%  →  refinada={area_r:.1f}%  "
          f"({sinal}{abs(delta):.1f}pp)  {elapsed:.1f}s")

    resultados.append({
        "obj":     obj,
        "refined": refined,
        "delta":   delta,
    })

# ── 3. Visualização ─────────────────────────────────────────────────────────
print(f"\n[3/3] Gerando visualização comparativa...")
os.unlink(FOTO_PROC)  # limpa temp
img = _img_corr       # já corrigido acima
W, H = img.size

PALETTE = [
    "#FF4444", "#44BB44", "#4488FF", "#FF9900",
    "#CC44CC", "#44CCCC", "#FF44AA", "#AAFF44",
]

# Lado a lado: esquerda=Claude, direita=Refinada
GAP    = 20
HEADER = 36
canvas = Image.new("RGB", (W * 2 + GAP, H + HEADER + 8), (22, 22, 22))
canvas.paste(img, (0, HEADER))
canvas.paste(img, (W + GAP, HEADER))

draw = ImageDraw.Draw(canvas)

# Cabeçalhos
draw.rectangle([0, 0, W - 1, HEADER - 1], fill=(40, 40, 40))
draw.text((10, 10), f"Claude  ({len(objetos_com_bbox)} objetos)", fill="#BBBBBB")

draw.rectangle([W + GAP, 0, W * 2 + GAP - 1, HEADER - 1], fill=(40, 40, 40))
draw.text((W + GAP + 10, 10),
          f"DINOv2 refinado  (36k gazetteer · 896px)", fill="#88CCFF")

for i, res in enumerate(resultados):
    obj     = res["obj"]
    bb      = obj["bbox_normalizada"]
    refined = res["refined"]
    cor     = PALETTE[i % len(PALETTE)]

    nome_label = obj["nome"][:28]

    # — esquerda: bbox Claude —
    x1 = int(bb["x1"] * W);      y1 = int(bb["y1"] * H) + HEADER
    x2 = int(bb["x2"] * W);      y2 = int(bb["y2"] * H) + HEADER
    for t in range(4):
        draw.rectangle([x1-t, y1-t, x2+t, y2+t], outline=cor)
    draw.text((x1 + 4, y1 + 4), nome_label, fill=cor)

    # centroide (círculo branco)
    ct = obj.get("centroide_normalizado")
    if ct:
        cx = int(ct["cx"] * W);  cy = int(ct["cy"] * H) + HEADER
        draw.ellipse([cx-7, cy-7, cx+7, cy+7], outline="#FFFFFF", width=2)
        draw.ellipse([cx-2, cy-2, cx+2, cy+2], fill="#FFFFFF")

    # — direita: bbox refinada —
    ox = W + GAP
    rx1 = int(refined["x1"] * W) + ox;  ry1 = int(refined["y1"] * H) + HEADER
    rx2 = int(refined["x2"] * W) + ox;  ry2 = int(refined["y2"] * H) + HEADER
    for t in range(4):
        draw.rectangle([rx1-t, ry1-t, rx2+t, ry2+t], outline=cor)
    area_r = (refined["x2"]-refined["x1"])*(refined["y2"]-refined["y1"])*100
    draw.text((rx1 + 4, ry1 + 4), f"{nome_label}  {area_r:.0f}%", fill=cor)

OUT = ROOT / "poc_dinov2/output/e2e_result.jpg"
OUT.parent.mkdir(parents=True, exist_ok=True)
canvas.save(OUT, "JPEG", quality=93)

# ── sumário ──────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Foto     : {FOTO.name}  ({W}×{H})")
print(f"  Claude   : {t_claude:.1f}s  →  {len(objetos)} objetos")
print(f"  Gazetteer: {gaz._idx.ntotal:,} embeddings")
if resultados:
    deltas = [r["delta"] for r in resultados]
    print(f"  Δ área   : {np.mean(deltas):+.1f}pp médio  "
          f"(range {min(deltas):+.1f} … {max(deltas):+.1f}pp)")
print(f"  Output   : {OUT}")
print(f"{'='*60}\n")
