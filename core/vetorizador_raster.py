"""
core/vetorizador_raster.py — conversão raster→vector estilo GIS.

Converte a superfície de object-ness (`AnaliseCena.superficie`, grid de blocos
(Hb,Wb) com valores [0,1]) em polígonos vetoriais Shapely via isocontours
(marching squares do scikit-image).

Pipeline:
    1. Binariza superficie com threshold (ISOCONTOUR_THRESHOLD).
    2. Rotula componentes conectados (scipy.ndimage.label).
    3. Para cada região, extrai contorno externo (shell) e furos (holes)
       via cv2.findContours com RETR_CCOMP (hierarquia pai/filho).
    4. Monta Shapely Polygon(shell, holes=[...]) — furos abaixo de
       MIN_AREA_PIXEL são ignorados.
    5. Valida e filtra polígonos (is_valid, área mínima, vértices mínimos).
    6. Calcula atributos por polígono (id, area, centroid, vertices,
       confidence, n_holes).
    7. Calcula iou_vs_claude quando bboxes_claude são fornecidas.
    8. Exporta FeatureCollection GeoJSON.
    9. Mede e loga latência.

Novo helper público:
    casar_poligono_a_bbox(resultado_vet, bbox, shape) -> str
        Casa o melhor polígono de `resultado_vet` a `bbox` por IoU e retorna
        o GeoJSON Feature correspondente, ou "" se nenhum casar acima de
        IOU_MATCH_MIN.

Uso típico (aditivo, sem alterar pipeline existente):
    from core.vetorizador_raster import vetorizar_superficie, casar_poligono_a_bbox
    resultado = vetorizar_superficie(cena.superficie, bboxes_claude=bboxes)
    for obj_bbox in bboxes:
        geom = casar_poligono_a_bbox(resultado, obj_bbox, cena.superficie.shape)
"""
from __future__ import annotations

import json
import logging
import time
from typing import Optional

import cv2
import numpy as np
import scipy.ndimage
from shapely.geometry import Polygon, box, mapping

from core.config import (
    IOU_MATCH_MIN,
    ISOCONTOUR_THRESHOLD,
    MIN_AREA_PIXEL,
    MIN_POLYGON_VERTICES,
)

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


def _extrair_poligono_com_furos(
    regiao_mask: np.ndarray,
    min_area: float,
) -> Optional[Polygon]:
    """Extrai um Shapely Polygon com furos (holes) a partir da máscara binária.

    Usa cv2.findContours com RETR_CCOMP para separar contorno externo (shell,
    parent == -1 na hierarquia) dos contornos internos (holes, parent >= 0).
    Furos com área < min_area são ignorados. O polígono retornado já desconta
    os furos em sua área (poly.area = área líquida).

    Args:
        regiao_mask: array bool/uint8 (H, W) com True/1 na região de interesse.
        min_area:    área mínima em pixels de bloco para aceitar um furo.

    Returns:
        Polygon Shapely válido com possíveis interiores, ou None se não for
        possível construir polígono válido com vértices suficientes.
    """
    # cv2.findContours exige uint8
    mask_u8 = regiao_mask.astype(np.uint8)

    # RETR_CCOMP: 2 níveis — nível 0 = contornos externos, nível 1 = buracos
    # hierarchy[0][i] = [next_sibling, prev_sibling, first_child, parent]
    # parent == -1 → shell; parent >= 0 → hole do contorno parent
    contours, hierarchy = cv2.findContours(
        mask_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours or hierarchy is None:
        return None

    hier = hierarchy[0]  # shape (N, 4)

    # Agrupa: shell_idx → lista de hole_idxs
    shells_holes: dict[int, list[int]] = {}
    for i, h in enumerate(hier):
        parent = int(h[3])
        if parent == -1:
            # Shell: nó raiz na hierarquia CCOMP
            if i not in shells_holes:
                shells_holes[i] = []
        else:
            # Hole: filho de 'parent'
            if parent not in shells_holes:
                shells_holes[parent] = []
            shells_holes[parent].append(i)

    if not shells_holes:
        return None

    # Pega o maior shell por número de pontos (contorno externo principal)
    shell_idx = max(shells_holes.keys(), key=lambda idx: len(contours[idx]))
    hole_idxs = shells_holes[shell_idx]

    shell_cnt = contours[shell_idx]
    if len(shell_cnt) < MIN_POLYGON_VERTICES:
        return None

    # cv2 dá coordenadas (x, y) — diretamente usáveis no Shapely (x=col, y=row)
    shell_coords = [(float(pt[0][0]), float(pt[0][1])) for pt in shell_cnt]

    # Furos: filtra por área mínima
    holes_coords: list[list[tuple[float, float]]] = []
    for hi in hole_idxs:
        hole_cnt = contours[hi]
        hole_pts = [(float(pt[0][0]), float(pt[0][1])) for pt in hole_cnt]
        if len(hole_pts) < MIN_POLYGON_VERTICES:
            continue
        # Área aproximada do furo (usando Shapely temporariamente)
        try:
            area_furo = Polygon(hole_pts).area
        except Exception:
            continue
        if area_furo < min_area:
            continue  # furo muito pequeno — ignora
        holes_coords.append(hole_pts)

    try:
        poly = Polygon(shell_coords, holes_coords)
    except Exception:
        return None

    return poly


def casar_poligono_a_bbox(
    resultado_vet: dict,
    bbox: dict,
    shape: tuple[int, int],
) -> str:
    """Casa o melhor polígono de `resultado_vet` a `bbox` por IoU.

    Percorre todos os polígonos em resultado_vet['geometries'], calcula o IoU
    de cada um contra `bbox` (normalizada {x1,y1,x2,y2} ∈ [0,1]) usando
    `shape` = (Hb, Wb) para des-normalizar. Retorna o GeoJSON Feature do
    polígono com maior IoU se esse IoU ≥ IOU_MATCH_MIN; caso contrário "".

    Args:
        resultado_vet: dict retornado por vetorizar_superficie.
        bbox:          bbox normalizada {x1, y1, x2, y2} ∈ [0,1].
        shape:         (Hb, Wb) da superfície original (para des-normalizar).

    Returns:
        String GeoJSON de um Feature único, ou "" se nenhum polígono casar.
    """
    geometries: list[Polygon] = resultado_vet.get("geometries", [])
    attributes: list[dict] = resultado_vet.get("attributes", [])

    if not geometries:
        return ""

    Hb, Wb = shape
    x1 = bbox.get("x1", 0.0) * Wb
    y1 = bbox.get("y1", 0.0) * Hb
    x2 = bbox.get("x2", 1.0) * Wb
    y2 = bbox.get("y2", 1.0) * Hb
    bbox_poly = box(x1, y1, x2, y2)

    melhor_iou = 0.0
    melhor_idx = -1

    for i, poly in enumerate(geometries):
        try:
            inter = poly.intersection(bbox_poly).area
            uniao = poly.union(bbox_poly).area
            iou = float(inter / uniao) if uniao > 0 else 0.0
        except Exception:
            iou = 0.0

        if iou > melhor_iou:
            melhor_iou = iou
            melhor_idx = i

    if melhor_idx < 0 or melhor_iou < IOU_MATCH_MIN:
        _log.debug("casar_poligono_sem_match", extra={
            "melhor_iou": round(melhor_iou, 4),
            "limiar": IOU_MATCH_MIN,
        })
        return ""

    poly_casado = geometries[melhor_idx]
    attr_casado = attributes[melhor_idx]

    feature = {
        "type": "Feature",
        "geometry": mapping(poly_casado),
        "properties": attr_casado,
    }
    geom_str = json.dumps(feature)

    _log.debug("casar_poligono_match", extra={
        "poly_id": attr_casado.get("id"),
        "iou": round(melhor_iou, 4),
    })
    return geom_str


def vetorizar_superficie(
    superficie: np.ndarray,
    *,
    threshold: float = ISOCONTOUR_THRESHOLD,
    min_area: float = MIN_AREA_PIXEL,
    bboxes_claude: Optional[list[dict]] = None,
) -> dict:
    """Converte a superfície de object-ness em polígonos vetoriais.

    Modela furos internos (holes) via cv2.findContours com RETR_CCOMP:
    regiões de baixa object-ness cercadas por shell de alta intensidade são
    representadas como interiores (Polygon.interiors) e descontadas da área.

    Args:
        superficie: array float32 (Hb, Wb) com valores [0, 1] representando
                    a intensidade de "object-ness" em cada bloco.
        threshold:  limiar de binarização para a isocontorno (default ISOCONTOUR_THRESHOLD).
        min_area:   área mínima em pixels de bloco para aceitar um polígono
                    (default MIN_AREA_PIXEL). Furos menores que este valor
                    também são ignorados.
        bboxes_claude: lista de bboxes normalizadas {x1, y1, x2, y2} ∈ [0,1]
                       retornadas pelo Claude. Quando fornecidas, calcula
                       iou_vs_claude para cada polígono.

    Returns:
        dict com chaves:
            'geometries':  list[Polygon]  — polígonos Shapely válidos (com furos).
            'attributes':  list[dict]     — atributos por polígono (mesma ordem).
                           Inclui 'n_holes' (int) além dos campos v1.
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

        # Filtro de área por contagem de pixels da região (antes de vetorizar).
        # Mais confiável que poly.area: o contorno cv2 é traçado pelo interior
        # dos pixels, então poly.area ≈ area_pixels − perímetro/2 para formas
        # pequenas. Usar regiao_mask.sum() garante que MIN_AREA_PIXEL reflete
        # literalmente "número de pixels de bloco" da região.
        area_pixels = int(regiao_mask.sum())
        if area_pixels < min_area:
            continue

        # ── 3. Extração de contornos com suporte a furos ─────────────────────
        # cv2.findContours(RETR_CCOMP) distingue shell (parent==-1) de holes
        # (parent>=0). Furos com área < min_area são ignorados.
        poly = _extrair_poligono_com_furos(regiao_mask, min_area)
        if poly is None:
            continue

        # ── 4. Validação do polígono ─────────────────────────────────────────
        if not poly.is_valid:
            poly = poly.buffer(0)  # tenta corrigir auto-interseção

        # Se buffer(0) resultou em MultiPolygon, usar o maior
        from shapely.geometry import MultiPolygon
        if isinstance(poly, MultiPolygon):
            poly = max(poly.geoms, key=lambda p: p.area)

        if not poly.is_valid or poly.is_empty:
            continue

        if poly.is_empty or poly.area <= 0:
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

        # Número de furos internos (holes) do polígono
        n_holes = len(list(poly.interiors))

        poly_id += 1
        atributos = {
            "id":          poly_id,
            "area":        round(float(poly.area), 4),
            "centroid":    (round(cx, 4), round(cy, 4)),
            "vertices":    n_vertices,
            "confidence":  round(confianca, 4),
            "iou_vs_claude": iou,
            "n_holes":     n_holes,
        }

        _log.info("poligono_vetorizado", extra={
            "id":         poly_id,
            "area":       atributos["area"],
            "centroid":   atributos["centroid"],
            "vertices":   atributos["vertices"],
            "confidence": atributos["confidence"],
            "n_holes":    n_holes,
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
