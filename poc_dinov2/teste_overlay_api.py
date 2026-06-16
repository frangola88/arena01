"""
poc_dinov2/teste_overlay_api.py — teste A/B/C real contra a Claude API pra
decidir se o overlay (2ª imagem com regiões candidatas numeradas) faz o modelo
contar/identificar melhor do que a foto sozinha.

Condições, por foto:
    A) só a foto real
    B) foto real + hint numérico no prompt ("~N objetos")
    C) foto real + overlay numerado (2ª imagem) + hint

Mede: nº de objetos que o modelo reporta em cada condição, comparado à
verdade (~13-14 ferramentas nas mestra04). Uma chamada por condição.

Reaproveita ANTHROPIC_API_KEY / ANTHROPIC_MODEL do projeto. Monta o corpo
multi-imagem na mão (o _claude_visao do projeto só manda 1 imagem).
"""
from __future__ import annotations

import base64
import io
import sys
import time
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import ANTHROPIC_API_KEY  # noqa: E402
# config do projeto aponta p/ um ID que a chave não acessa (404); usa o Sonnet atual
ANTHROPIC_MODEL = "claude-sonnet-4-6"
from core.analise_cena import _carregar_cinza  # noqa: E402
from poc_dinov2.demo_grid_celulas import (  # noqa: E402
    grid_coerencia, classificar_celulas, componentes, gerar_overlay,
    N_ROWS, N_COLS, AREA_MIN_CELULAS, CLOSING_ITER,
)
from scipy.ndimage import binary_closing  # noqa: E402

MAX_EDGE = 1568  # Anthropic reescala acima disso; reduzimos antes p/ economizar


def _b64_jpeg(img: Image.Image, max_edge: int = MAX_EDGE) -> str:
    w, h = img.size
    if max(w, h) > max_edge:
        s = max_edge / max(w, h)
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=90)
    return base64.standard_b64encode(buf.getvalue()).decode()


def _b64_png(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return base64.standard_b64encode(buf.getvalue()).decode()


def _img_block(b64: str, media_type: str) -> dict:
    return {"type": "image", "source": {"type": "base64",
            "media_type": media_type, "data": b64}}


def chamar(content: list) -> str:
    headers = {"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01",
               "content-type": "application/json"}
    body = {"model": ANTHROPIC_MODEL, "max_tokens": 2048,
            "messages": [{"role": "user", "content": content}]}
    r = httpx.post("https://api.anthropic.com/v1/messages",
                   headers=headers, json=body, timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f"API {r.status_code}: {r.text[:300]}")
    return r.json()["content"][0]["text"]


PERGUNTA_BASE = (
    "Esta é uma foto de ferramentas dispostas sobre tapetes escuros numa mesa. "
    "Conte e liste TODOS os objetos/ferramentas individuais visíveis. "
    "Inclua itens pequenos e os que estão próximos uns dos outros. "
    "Responda em JSON: {\"n_objetos\": <int>, \"objetos\": [{\"nome\": \"...\", "
    "\"posicao\": \"<descrição curta de onde está>\"}]}."
)
HINT = (" Uma pré-análise automática estimou aproximadamente {n} objetos na cena "
        "(apenas uma estimativa, pode estar errada — confie na imagem).")
INSTRUCAO_OVERLAY = (
    " A segunda imagem é um mapa de referência gerado automaticamente: marca "
    "regiões candidatas a objeto com caixas numeradas. Use como GUIA de onde "
    "olhar, mas confie na primeira imagem (a foto real) para identificar o que "
    "é cada objeto. As caixas podem fundir objetos próximos ou marcar coisas que "
    "não são objetos — corrija conforme o que você vê na foto real, e reporte "
    "objetos que existam fora das caixas.")


def construir_overlay(caminho: str):
    img_gray = _carregar_cinza(caminho)
    H, W = img_gray.shape
    sinal, (bh, bw) = grid_coerencia(img_gray, N_ROWS, N_COLS)
    mask, th = classificar_celulas(sinal)
    mask = binary_closing(mask, structure=np.ones((3, 3)), iterations=CLOSING_ITER)
    objs, _ = componentes(mask, AREA_MIN_CELULAS)
    overlay = gerar_overlay((H, W), objs, bh, bw, escala_overlay=0.30)
    return overlay, len(objs)


def avaliar(caminho: str):
    print(f"\n{'='*60}\n=== {Path(caminho).name} ===\n{'='*60}")
    foto = Image.open(caminho)
    foto_b64 = _b64_jpeg(foto)
    overlay, n_est = construir_overlay(caminho)
    overlay_b64 = _b64_png(overlay)

    resultados = {}

    # A) só a foto
    print("\n[A] só a foto...")
    rA = chamar([_img_block(foto_b64, "image/jpeg"), {"type": "text", "text": PERGUNTA_BASE}])
    resultados["A"] = rA

    # B) foto + hint numérico
    print("[B] foto + hint numérico...")
    rB = chamar([_img_block(foto_b64, "image/jpeg"),
                 {"type": "text", "text": PERGUNTA_BASE + HINT.format(n=n_est)}])
    resultados["B"] = rB

    # C) foto + overlay + hint
    print("[C] foto + overlay numerado + hint...")
    rC = chamar([_img_block(foto_b64, "image/jpeg"),
                 _img_block(overlay_b64, "image/png"),
                 {"type": "text", "text": PERGUNTA_BASE + HINT.format(n=n_est) + INSTRUCAO_OVERLAY}])
    resultados["C"] = rC

    print(f"\n  (overlay tinha {n_est} regiões candidatas)")
    for cond, txt in resultados.items():
        n = _extrair_n(txt)
        print(f"\n  --- Condição {cond}: n_objetos reportado = {n} ---")
        print("  " + txt.strip().replace("\n", "\n  ")[:1500])
    return resultados, n_est


def _extrair_n(txt: str):
    import json, re
    t = re.sub(r'^```[a-z]*\s*\n?', '', txt.strip())
    if t.endswith('```'):
        t = t[:-3].rstrip()
    m = re.search(r'(\{.*\})', t, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1)).get("n_objetos", "?")
        except Exception:
            pass
    m2 = re.search(r'"n_objetos"\s*:\s*(\d+)', t)
    return m2.group(1) if m2 else "?"


if __name__ == "__main__":
    alvos = sys.argv[1:] or [
        "data/fotos_mestras/IMG_mestra04semflash.jpg",
        "data/fotos_mestras/IMG_mestra04comflash.jpg",
    ]
    resumo = {}
    for a in alvos:
        res, n_est = avaliar(a)
        resumo[Path(a).name] = {c: _extrair_n(t) for c, t in res.items()} | {"overlay_n": n_est}
        time.sleep(1)
    print(f"\n\n{'#'*60}\n# RESUMO (n_objetos por condição)\n{'#'*60}")
    print(f"{'foto':<32} {'A(só foto)':>11} {'B(+hint)':>9} {'C(+overlay)':>12} {'overlay':>8}")
    for foto, d in resumo.items():
        print(f"{foto:<32} {str(d['A']):>11} {str(d['B']):>9} {str(d['C']):>12} {str(d['overlay_n']):>8}")
