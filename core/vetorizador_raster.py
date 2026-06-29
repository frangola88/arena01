"""
core/vetorizador_raster.py — conversão raster→vector estilo GIS.

Converte a superfície de object-ness (`AnaliseCena.superficie`, grid de blocos
(Hb,Wb) com valores [0,1]) em polígonos vetoriais Shapely via isocontours
(marching squares do scikit-image).

Pipeline:
    1. Binariza superficie com threshold (ISOCONTOUR_THRESHOLD).
    2. Rotula componentes conectados (scipy.ndimage.label).
    3. Para cada região, extrai contorno externo via skimage.measure.find_contours.
    4. Valida e filtra polígonos (is_valid, área mínima, vértices mínimos).
    5. Calcula atributos por polígono (id, area, centroid, vertices, confidence).
    6. Calcula iou_vs_claude quando bboxes_claude são fornecidas.
    7. Exporta FeatureCollection GeoJSON.
    8. Mede e loga latência.

Uso típico (aditivo, sem alterar pipeline existente):
    from core.vetorizador_raster import vetorizar_superficie
    resultado = vetorizar_superficie(cena.superficie, bboxes_claude=bboxes)
"""
from __future__ import annotations

import json
import logging
import time
from typing import Optional

import numpy as np
import scipy.ndimage
import skimage.measure
from shapely.geometry import Polygon, box, mapping

from core.config import ISOCONTOUR_THRESHOLD, MIN_AREA_PIXEL, MIN_POLYGON_VERTICES

_log = logging.getLogger("casaiq.vetorizador_raster")


def _calcular_iou_vs_claude(
    poly: Polygon,
    bboxes_claude: list[dict],
    shape: tuple[int, int],
) -> Optional[float]:
    """Calcula o melhor IoU entre o polígono e as bboxes normalizadas do Claude.

    As bboxes vêm normalizadas {x1, y1, x2, y2} em [0, 1]; são convertidas
    para coordenadas de blocos usando `shape` = (Hb, Wb). Retorna o maior IoU
    encontrado sobre todas as bboxes, ou None se lista vazia.
    """
    if not bboxes_claude:
        return None

    Hb, Wb = shape
    melhor_iou = 0.0

    for bb in bboxes_claude:
        # Converte coords normalizadas → coords de blocos (x=col, y=row)
        x1 = bb.get("x1", 0.0) * Wb
        y1 = bb.get("y1", 0.0) * Hb
        x2 = bb.get("x2", 1.0) * Wb
        y2 = bb.get("y2", 1.0) * Hb

        bbox_poly = box(x1, y1, x2, y2)

        try:
            inter = poly.intersection(bbox_poly).area
            uniao = poly.union(bbox_poly).area
            iou = float(inter / uniao) if uniao > 0 else 0.0
        except Exception:
            iou = 0.0

        if iou > melhor_iou:
            melhor_iou = iou

    return round(melhor_iou, 4)


def vetorizar_superficie(
    superficie: np.ndarray,
    *,
    threshold: float = ISOCONTOUR_THRESHOLD,
    min_area: float = MIN_AREA_PIXEL,
    bboxes_claude: Optional[list[dict]] = None,
) -> dict:
    """Converte a superfície de object-ness em polígonos vetoriais.

    Args:
        superficie: array float32 (Hb, Wb) com valores [0, 1] representando
                    a intensidade de "object-ness" em cada bloco.
        threshold:  limiar de binarização para a isocontorno (default ISOCONTOUR_THRESHOLD).
        min_area:   área mínima em pixels de bloco para aceitar um polígono
                    (default MIN_AREA_PIXEL).
        bboxes_claude: lista de bboxes normalizadas {x1, y1, x2, y2} ∈ [0,1]
                       retornadas pelo Claude. Quando fornecidas, calcula
                       iou_vs_claude para cada polígono.

    Returns:
        dict com chaves:
            'geometries':  list[Polygon]  — polígonos Shapely válidos.
            'attributes':  list[dict]     — atributos por polígono (mesma ordem).
            'geojson':     str            — JSON de FeatureCollection parseável.
            'stats':       dict           — total_polys, total_area, coverage_pct.
            'performance': dict           — latency_ms, threshold_used.
    """
    t0 = time.perf_counter()

    superficie = np.asarray(superficie, dtype=np.float32)
    Hb, Wb = superficie.shape

    # ── 1. Binarização ───────────────────────────────────────────────────────
    mask = superficie > threshold

    # Estrutura de retorno vazia para caso degenerado
    def _vazio(latencia_ms: float) -> dict:
        return {
            "geometries": [],
            "attributes": [],
            "geojson": json.dumps({"type": "FeatureCollection", "features": []}),
            "stats": {"total_polys": 0, "total_area": 0.0, "coverage_pct": 0.0},
            "performance": {"latency_ms": round(latencia_ms, 3), "threshold_used": threshold},
        }

    if not mask.any():
        _log.info("vetorizar_raster_superficie_vazia", extra={"threshold": threshold})
        return _vazio((time.perf_counter() - t0) * 1000)

    # ── 2. Rotulação de componentes conectados ───────────────────────────────
    labeled, n_labels = scipy.ndimage.label(mask)

    geometries: list[Polygon] = []
    attributes: list[dict] = []

    poly_id = 0

    for label_val in range(1, n_labels + 1):
        regiao_mask = labeled == label_val

        # ── 3. Extração de contornos (isocontours) ───────────────────────────
        # Usa a máscara binária (0/1) da região para garantir transição no nível
        # 0.5, independentemente dos valores absolutos da superfície.
        # find_contours devolve coordenadas em (row, col) → (y, x).
        contornos = skimage.measure.find_contours(regiao_mask.astype(np.float32), 0.5)

        if not contornos:
            continue

        # Pega o contorno externo (mais longo)
        contorno = max(contornos, key=len)

        if len(contorno) < MIN_POLYGON_VERTICES:
            continue

        # Converte (row, col) → (x=col, y=row) para Shapely
        coords = [(float(pt[1]), float(pt[0])) for pt in contorno]

        try:
            poly = Polygon(coords)
        except Exception:
            continue

        # ── 4. Validação do polígono ─────────────────────────────────────────
        if not poly.is_valid:
            poly = poly.buffer(0)  # tenta corrigir auto-interseção

        if not poly.is_valid or poly.is_empty:
            continue

        if poly.area < min_area:
            continue

        # Recontagem de vértices após possível buffer
        n_vertices = len(poly.exterior.coords)
        if n_vertices < MIN_POLYGON_VERTICES:
            continue

        # ── 5. Atributos ─────────────────────────────────────────────────────
        cx, cy = poly.centroid.x, poly.centroid.y

        # Confiança: média da superfície dentro da máscara da região
        vals_regiao = superficie[regiao_mask]
        confianca = float(np.clip(vals_regiao.mean(), 0.0, 1.0))

        # IoU vs Claude
        iou = _calcular_iou_vs_claude(poly, bboxes_claude or [], (Hb, Wb))

        poly_id += 1
        atributos = {
            "id":          poly_id,
            "area":        round(float(poly.area), 4),
            "centroid":    (round(cx, 4), round(cy, 4)),
            "vertices":    n_vertices,
            "confidence":  round(confianca, 4),
            "iou_vs_claude": iou,
        }

        _log.info("poligono_vetorizado", extra={
            "id":         poly_id,
            "area":       atributos["area"],
            "centroid":   atributos["centroid"],
            "vertices":   atributos["vertices"],
            "confidence": atributos["confidence"],
        })

        geometries.append(poly)
        attributes.append(atributos)

    # ── 6. GeoJSON FeatureCollection ─────────────────────────────────────────
    features = []
    for poly, attr in zip(geometries, attributes):
        features.append({
            "type": "Feature",
            "geometry": mapping(poly),
            "properties": attr,
        })
    geojson_str = json.dumps({"type": "FeatureCollection", "features": features})

    # ── 7. Estatísticas ───────────────────────────────────────────────────────
    total_area = sum(a["area"] for a in attributes)
    area_total_superficie = float(Hb * Wb)
    coverage_pct = round(100.0 * total_area / area_total_superficie, 4) if area_total_superficie > 0 else 0.0

    latencia_ms = (time.perf_counter() - t0) * 1000

    _log.info("vetorizar_raster_resumo", extra={
        "total_polys":   len(geometries),
        "total_area":    round(total_area, 4),
        "coverage_pct":  coverage_pct,
        "latency_ms":    round(latencia_ms, 3),
        "threshold_used": threshold,
        "geojson_len":   len(geojson_str),
    })

    return {
        "geometries": geometries,
        "attributes": attributes,
        "geojson":    geojson_str,
        "stats": {
            "total_polys":  len(geometries),
            "total_area":   round(total_area, 4),
            "coverage_pct": coverage_pct,
        },
        "performance": {
            "latency_ms":    round(latencia_ms, 3),
            "threshold_used": threshold,
        },
    }
