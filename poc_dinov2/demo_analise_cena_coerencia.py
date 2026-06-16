"""
poc_dinov2/demo_analise_cena_coerencia.py — testa um sinal de "agitação" por
coerência de gradiente (structure tensor) no lugar de variância local pura,
pra ver se discrimina textura (grão de madeira, tecido) de borda real de
objeto melhor do que variância — sem precisar do blur pesado (σ=4) que hoje
sustenta a robustez do baseline às custas de fundir objetos próximos.

Sinal por bloco:
    Ix, Iy = sobel(img)
    Jxx, Jyy, Jxy = médias de Ix², Iy², IxIy dentro do bloco  (structure tensor 2x2)
    autovalores λ1>=λ2
    energia    = λ1 + λ2          (quanta variação de borda tem no bloco)
    coerência  = (λ1-λ2)/(λ1+λ2)  (0=isotrópico/ruído, 1=borda bem definida/linear)
    sinal      = energia * coerência

Ideia: textura fina (grão de madeira) tende a ter energia espalhada em
várias direções dentro do bloco (coerência baixa); borda real de ferramenta
(linha reta/contínua) tem alta coerência. Multiplicar por energia evita que
ruído de sensor com coerência espúria conte como objeto.

Roda baseline (variância, block=16) vs coerência em 3 resoluções, nas duas
fotos-mestre.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, sobel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.analise_cena import (  # noqa: E402
    BLOCK_SZ, N_SECOES, M_SECOES, PASSO_SAMP, SIGMA_SUAV, SIGMA_SUP,
    SIGMOID_K, SIGMOID_MU, TH_OBJ_PCT, AREA_MIN,
    _variancia_local, _normalizar, _amostrar_secoes, _sigmoid, _carregar_cinza,
)
from poc_dinov2.demo_analise_cena_multipass import reconstruir, detectar_bbox  # noqa: E402


def _signal_coerencia(img_gray: np.ndarray, block_sz: int):
    Ix = sobel(img_gray.astype(np.float32), axis=1)
    Iy = sobel(img_gray.astype(np.float32), axis=0)
    Ixx, Iyy, Ixy = Ix * Ix, Iy * Iy, Ix * Iy

    H, W = img_gray.shape
    Hb, Wb = H // block_sz, W // block_sz

    def block_mean(a):
        a = a[:Hb * block_sz, :Wb * block_sz].reshape(Hb, block_sz, Wb, block_sz)
        return a.mean(axis=(1, 3))

    Jxx, Jyy, Jxy = block_mean(Ixx), block_mean(Iyy), block_mean(Ixy)
    trace = Jxx + Jyy
    disc = np.sqrt(np.clip((trace / 2) ** 2 - (Jxx * Jyy - Jxy ** 2), 0, None))
    lam1, lam2 = trace / 2 + disc, trace / 2 - disc
    energia = lam1 + lam2
    coerencia = (lam1 - lam2) / (lam1 + lam2 + 1e-8)
    sinal = energia * coerencia
    return sinal.astype(np.float32), energia.astype(np.float32), coerencia.astype(np.float32)


def pipeline(mapa_bruto, *, sigma_pre, n_sec, m_sec, passo, th_obj_pct, area_min, sigma_pos=SIGMA_SUP):
    t0 = time.time()
    mapa = _normalizar(gaussian_filter(mapa_bruto, sigma=sigma_pre)) if sigma_pre > 0 \
        else _normalizar(mapa_bruto)
    Hb, Wb = mapa.shape
    pts, vals = _amostrar_secoes(mapa, n_sec, m_sec, passo)
    sup = reconstruir((Hb, Wb), pts, vals, sigma_pos)
    sup = _sigmoid(sup, SIGMOID_K, SIGMOID_MU)
    objs, _ = detectar_bbox(sup, th_obj_pct, area_min)
    return {"n_obj": len(objs), "tempo_s": round(time.time() - t0, 3), "sup": sup}


def avaliar(caminho: str):
    print(f"\n=== {Path(caminho).name} ===")
    img_gray = _carregar_cinza(caminho)

    var_map = _variancia_local(img_gray, BLOCK_SZ)
    r_base = pipeline(var_map, sigma_pre=SIGMA_SUAV, n_sec=N_SECOES, m_sec=M_SECOES,
                       passo=PASSO_SAMP, th_obj_pct=TH_OBJ_PCT, area_min=AREA_MIN)
    print(f"  baseline variância   (block=16, sigma=4.0)  : n_obj={r_base['n_obj']:2d}  "
          f"tempo={r_base['tempo_s']:.2f}s")

    sinal16, energia16, coer16 = _signal_coerencia(img_gray, 16)
    r_coer_sigma4 = pipeline(sinal16, sigma_pre=SIGMA_SUAV, n_sec=N_SECOES, m_sec=M_SECOES,
                              passo=PASSO_SAMP, th_obj_pct=TH_OBJ_PCT, area_min=AREA_MIN)
    print(f"  coerência  (block=16, sigma=4.0, mesmo th)  : n_obj={r_coer_sigma4['n_obj']:2d}  "
          f"tempo={r_coer_sigma4['tempo_s']:.2f}s")

    r_coer_sigma0 = pipeline(sinal16, sigma_pre=0.0, n_sec=N_SECOES, m_sec=M_SECOES,
                              passo=PASSO_SAMP, th_obj_pct=TH_OBJ_PCT, area_min=AREA_MIN)
    print(f"  coerência  (block=16, sigma=0, sem preblur) : n_obj={r_coer_sigma0['n_obj']:2d}  "
          f"tempo={r_coer_sigma0['tempo_s']:.2f}s")

    sinal8, energia8, coer8 = _signal_coerencia(img_gray, 8)
    r_coer_hires = pipeline(sinal8, sigma_pre=1.0, n_sec=64, m_sec=64, passo=2,
                             th_obj_pct=TH_OBJ_PCT, area_min=AREA_MIN)
    print(f"  coerência  (block=8,  sigma=1.0, hires)     : n_obj={r_coer_hires['n_obj']:2d}  "
          f"tempo={r_coer_hires['tempo_s']:.2f}s")

    r_coer_hires2 = pipeline(sinal8, sigma_pre=1.0, n_sec=64, m_sec=64, passo=2,
                              th_obj_pct=70, area_min=6)
    print(f"  coerência  (block=8,  sigma=1.0, th=70/a=6) : n_obj={r_coer_hires2['n_obj']:2d}  "
          f"tempo={r_coer_hires2['tempo_s']:.2f}s")

    # preview lado a lado: var_norm vs coerencia (block=16, ambos sem blur) e supercifies finais
    try:
        from PIL import Image
        var_norm_raw = _normalizar(var_map)
        coer_norm_raw = _normalizar(sinal16)
        painel_raw = np.concatenate([var_norm_raw, coer_norm_raw], axis=1)
        painel_sup = np.concatenate([r_base["sup"], r_coer_sigma4["sup"], r_coer_sigma0["sup"]], axis=1)
        Image.fromarray((np.clip(painel_raw, 0, 1) * 255).astype(np.uint8)) \
            .save(f"/tmp/coer_raw_{Path(caminho).stem}.png")
        Image.fromarray((np.clip(painel_sup, 0, 1) * 255).astype(np.uint8)) \
            .save(f"/tmp/coer_sup_{Path(caminho).stem}.png")
        print(f"  preview raw: /tmp/coer_raw_{Path(caminho).stem}.png (variância | coerência)")
        print(f"  preview sup: /tmp/coer_sup_{Path(caminho).stem}.png "
              f"(baseline | coer sigma4 | coer sigma0)")
    except Exception as e:
        print(f"  (preview falhou: {e})")


if __name__ == "__main__":
    alvos = sys.argv[1:] or [
        "data/fotos_mestras/IMG_mestra04semflash.jpg",
        "data/fotos_mestras/IMG_mestra04comflash.jpg",
    ]
    for a in alvos:
        avaliar(a)
