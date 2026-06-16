"""
poc_dinov2/teste_overlay_api_regra.py — teste de controle pro confound do
teste A/B/C: re-roda A e B AGORA COM a regra de contagem explícita no texto
("embalagem/kit fechado = 1 item"), que o teste original omitia.

Objetivo: isolar quanto do ganho do overlay (condição C) era só comunicar
visualmente essa regra. Se A'/B' com a regra já chegarem perto de 14
(verdade verificada), o overlay tem pouco valor marginal e basta corrigir o
prompt. Se continuarem inflados, o overlay se prova essencial.

Verdade verificada (contagem manual sob "fechada=1"): 14 objetos em ambas.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from poc_dinov2.teste_overlay_api import (  # noqa: E402
    _b64_jpeg, _img_block, chamar, _extrair_n, PERGUNTA_BASE, HINT,
    construir_overlay,
)

REGRA = (" REGRA DE CONTAGEM: uma embalagem, pacote ou kit fechado conta como "
         "1 único objeto (NÃO conte as peças individuais dentro de uma "
         "embalagem fechada). Ferramentas soltas contam individualmente.")


def avaliar(caminho: str):
    print(f"\n{'='*60}\n=== {Path(caminho).name} (verdade=14) ===\n{'='*60}")
    foto_b64 = _b64_jpeg(Image.open(caminho))
    _, n_est = construir_overlay(caminho)

    print("\n[A'] só foto + REGRA...")
    rA = chamar([_img_block(foto_b64, "image/jpeg"),
                 {"type": "text", "text": PERGUNTA_BASE + REGRA}])

    print("[B'] foto + REGRA + hint numérico...")
    rB = chamar([_img_block(foto_b64, "image/jpeg"),
                 {"type": "text", "text": PERGUNTA_BASE + REGRA + HINT.format(n=n_est)}])

    nA, nB = _extrair_n(rA), _extrair_n(rB)
    print(f"\n  A' (só foto + regra)        : n_objetos = {nA}")
    print(f"  B' (foto + regra + hint)    : n_objetos = {nB}")
    print(f"\n  --- A' completo ---\n  " + rA.strip().replace('\n', '\n  ')[:1200])
    return {"A_regra": nA, "B_regra": nB}


if __name__ == "__main__":
    alvos = sys.argv[1:] or [
        "data/fotos_mestras/IMG_mestra04semflash.jpg",
        "data/fotos_mestras/IMG_mestra04comflash.jpg",
    ]
    resumo = {}
    for a in alvos:
        resumo[Path(a).name] = avaliar(a)
    print(f"\n\n{'#'*70}")
    print("# RESUMO FINAL (verdade=14; A/B/C originais p/ comparar)")
    print(f"{'#'*70}")
    print(f"{'foto':<30} {'A(orig)':>8} {'B(orig)':>8} {'A+regra':>8} {'B+regra':>8} {'C(overlay)':>11}")
    orig = {"IMG_mestra04semflash.jpg": (25, 22, 13),
            "IMG_mestra04comflash.jpg": (28, 26, 16)}
    for foto, d in resumo.items():
        a0, b0, c0 = orig.get(foto, ("?", "?", "?"))
        print(f"{foto:<30} {a0:>8} {b0:>8} {str(d['A_regra']):>8} {str(d['B_regra']):>8} {c0:>11}")
