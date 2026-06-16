"""
poc_dinov2/demo_analise_cena_rotada.py — avalia o ganho de somar grids de
seções a 45° e -45° à tomografia de core/analise_cena.py.

Ideia do usuário: além das 48 seções H + 48 V no eixo original, rodar o
mesmo processo de seções sobre o mapa de variância ROTACIONADO em +45° e
-45°, e comparar/combinar os resultados.

Equivalência usada aqui (mais rápida que rotar a imagem 4000x3000 crua):
rotacionar o MAPA DE VARIÂNCIA já suavizado (pequeno, ~250x187 blocos) em
vez da imagem inteira. Amostrar 48 seções H+V nesse mapa rotacionado é
matematicamente equivalente (a menos de erro de interpolação) a amostrar
seções diagonais no mapa original — é o que o pedido descreve.

Saída: n_objetos por eixo (0°, 45°, -45°), por combinação (max/mean), preview
PNG lado a lado, e custo de tempo.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, rotate as nd_rotate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.analise_cena import (  # noqa: E402
    BLOCK_SZ, N_SECOES, M_SECOES, PASSO_SAMP, SIGMA_SUAV,
    SIGMOID_K, SIGMOID_MU, TH_OBJ_PCT,
    _variancia_local, _normalizar, _amostrar_secoes, _reconstruir,
    _sigmoid, _detectar_objetos, _carregar_cinza,
)


def _cortar_centro(arr: np.ndarray, H: int, W: int) -> np.ndarray:
    h, w = arr.shape
    y0 = max(0, (h - H) // 2)
    x0 = max(0, (w - W) // 2)
    out = arr[y0:y0 + H, x0:x0 + W]
    if out.shape != (H, W):
        # canvas rotado ficou menor que o alvo por arredondamento — pad
        pad_h = H - out.shape[0]
        pad_w = W - out.shape[1]
        out = np.pad(out, ((0, max(0, pad_h)), (0, max(0, pad_w))))
        out = out[:H, :W]
    return out


def superficie_por_eixo(var_norm: np.ndarray, angle: float) -> tuple[np.ndarray, float]:
    """Reconstrói a superfície de object-ness amostrando seções H+V no
    referencial rotacionado por `angle`, devolvendo já no referencial
    original (Hb x Wb)."""
    t0 = time.time()
    Hb, Wb = var_norm.shape
    if angle == 0:
        base = var_norm
    else:
        base = nd_rotate(var_norm, angle, reshape=True, order=1,
                          mode="constant", cval=float(var_norm.mean()))

    pts, vals = _amostrar_secoes(base, N_SECOES, M_SECOES, PASSO_SAMP)
    sup_rot = _reconstruir(base.shape, pts, vals)
    sup_rot = _sigmoid(sup_rot, SIGMOID_K, SIGMOID_MU)

    if angle == 0:
        sup_orig = sup_rot
    else:
        canvas_back = nd_rotate(sup_rot, -angle, reshape=True, order=1,
                                 mode="constant", cval=0.0)
        sup_orig = _cortar_centro(canvas_back, Hb, Wb)
    return sup_orig, time.time() - t0


def avaliar(caminho: str):
    print(f"\n=== {Path(caminho).name} ===")
    img_gray = _carregar_cinza(caminho)
    var_map = _variancia_local(img_gray, BLOCK_SZ)
    var_norm = _normalizar(gaussian_filter(var_map, sigma=SIGMA_SUAV))
    Hb, Wb = var_norm.shape

    sups = {}
    tempos = {}
    for angulo, nome in [(0, "0°"), (45, "45°"), (-45, "-45°")]:
        sup, dt = superficie_por_eixo(var_norm, angulo)
        sups[nome] = sup
        tempos[nome] = dt
        picos, _ = _detectar_objetos(sup, TH_OBJ_PCT)
        print(f"  eixo {nome:>5s}: n_obj={len(picos):2d}  tempo={dt:.2f}s")

    combo_max = np.maximum(np.maximum(sups["0°"], sups["45°"]), sups["-45°"])
    combo_mean = (sups["0°"] + sups["45°"] + sups["-45°"]) / 3.0

    picos_max, _ = _detectar_objetos(combo_max, TH_OBJ_PCT)
    picos_mean, _ = _detectar_objetos(combo_mean, TH_OBJ_PCT)
    picos_base, _ = _detectar_objetos(sups["0°"], TH_OBJ_PCT)

    tempo_total_3eixos = sum(tempos.values())
    print(f"  --- combinação ---")
    print(f"  baseline (só 0°)      : n_obj={len(picos_base):2d}  tempo={tempos['0°']:.2f}s")
    print(f"  combo max(0,45,-45)   : n_obj={len(picos_max):2d}  tempo_total={tempo_total_3eixos:.2f}s "
          f"({tempo_total_3eixos / tempos['0°']:.1f}x o custo)")
    print(f"  combo mean(0,45,-45)  : n_obj={len(picos_mean):2d}")

    # preview lado a lado
    try:
        from PIL import Image
        painel = np.concatenate(
            [sups["0°"], sups["45°"], sups["-45°"], combo_max], axis=1
        )
        painel_img = Image.fromarray((np.clip(painel, 0, 1) * 255).astype(np.uint8))
        out = f"/tmp/rotada_{Path(caminho).stem}.png"
        painel_img.save(out)
        print(f"  preview: {out}  (0° | 45° | -45° | combo_max)")
    except Exception as e:
        print(f"  (preview falhou: {e})")

    return {
        "base": len(picos_base), "45": len(picos_max), "combo_max": len(picos_max),
        "combo_mean": len(picos_mean), "tempo_base": tempos["0°"], "tempo_total": tempo_total_3eixos,
    }


if __name__ == "__main__":
    alvos = sys.argv[1:] or [
        "data/fotos_mestras/IMG_mestra04semflash.jpg",
        "data/fotos_mestras/IMG_mestra04comflash.jpg",
    ]
    for a in alvos:
        avaliar(a)
