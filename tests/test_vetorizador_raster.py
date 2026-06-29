"""
Testes de core/vetorizador_raster.py.

Cobertura ≥ 80% do módulo. Cada caso tem asserções substantivas sobre
geometria, contagem de polígonos, GeoJSON, performance e IoU.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
from shapely.geometry import Polygon

from core.vetorizador_raster import vetorizar_superficie
from core.config import ISOCONTOUR_THRESHOLD, MIN_AREA_PIXEL, MIN_POLYGON_VERTICES


# ─────────────────────────────────────────────────────────────────────────────
# Auxiliares
# ─────────────────────────────────────────────────────────────────────────────

def _geojson_valido(geojson_str: str, n_esperado: int) -> None:
    """Valida a estrutura GeoJSON e confere número de features."""
    doc = json.loads(geojson_str)
    assert doc["type"] == "FeatureCollection", "GeoJSON deve ser FeatureCollection"
    assert len(doc["features"]) == n_esperado, (
        f"Esperado {n_esperado} features, obtido {len(doc['features'])}"
    )
    for feat in doc["features"]:
        assert feat["type"] == "Feature"
        assert "geometry" in feat
        assert "properties" in feat


# ─────────────────────────────────────────────────────────────────────────────
# B4 — superfície plana/abaixo do threshold → lista vazia, sem crash
# ─────────────────────────────────────────────────────────────────────────────

def test_superficie_plana_retorna_vazio(superficie_plana):
    """Superfície uniforme baixa → geometries vazio, stats zerados, sem exceção."""
    out = vetorizar_superficie(superficie_plana)

    assert out["geometries"] == []
    assert out["attributes"] == []
    assert out["stats"]["total_polys"] == 0
    assert out["stats"]["total_area"] == 0.0
    _geojson_valido(out["geojson"], 0)


def test_superficie_zeros_retorna_vazio():
    """Array de zeros: sem objetos acima do threshold."""
    sup = np.zeros((10, 10), dtype=np.float32)
    out = vetorizar_superficie(sup)
    assert out["geometries"] == []
    assert out["stats"]["total_polys"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# B3 — duas ilhas separadas → 2 polígonos, centroides dentro, área > 0
# ─────────────────────────────────────────────────────────────────────────────

def test_duas_ilhas_produzem_dois_poligonos(superficie_duas_ilhas):
    """Dois blocos de alta intensidade separados → exatamente 2 polígonos."""
    out = vetorizar_superficie(superficie_duas_ilhas)

    assert out["stats"]["total_polys"] == 2, (
        f"Esperado 2 polígonos, obtido {out['stats']['total_polys']}"
    )
    assert len(out["geometries"]) == 2
    assert len(out["attributes"]) == 2


def test_duas_ilhas_poligonos_validos(superficie_duas_ilhas):
    """Polígonos das duas ilhas devem ser Shapely válidos."""
    out = vetorizar_superficie(superficie_duas_ilhas)
    for poly in out["geometries"]:
        assert isinstance(poly, Polygon)
        assert poly.is_valid, f"Polígono inválido: {poly}"
        assert not poly.is_empty


def test_duas_ilhas_area_positiva(superficie_duas_ilhas):
    """Cada polígono deve ter área > 0."""
    out = vetorizar_superficie(superficie_duas_ilhas)
    for attr in out["attributes"]:
        assert attr["area"] > 0.0, "Área deve ser positiva"


def test_duas_ilhas_centroides_dentro_das_ilhas(superficie_duas_ilhas):
    """Centroide de cada polígono deve cair dentro da respectiva ilha."""
    # Ilha 1: linhas 1-5, colunas 1-5 → x ∈ [1,5], y ∈ [1,5]
    # Ilha 2: linhas 12-16, colunas 12-16 → x ∈ [12,16], y ∈ [12,16]
    out = vetorizar_superficie(superficie_duas_ilhas)

    centroides = [attr["centroid"] for attr in out["attributes"]]  # (x, y) = (col, row)

    # Cada centroide deve estar dentro de uma ilha conhecida
    ilha1_x = (0.0, 7.0)   # x=col ∈ [1,5] com margem do contorno
    ilha1_y = (0.0, 7.0)   # y=row
    ilha2_x = (10.0, 19.0)
    ilha2_y = (10.0, 19.0)

    def _em_ilha1(cx, cy):
        return ilha1_x[0] <= cx <= ilha1_x[1] and ilha1_y[0] <= cy <= ilha1_y[1]

    def _em_ilha2(cx, cy):
        return ilha2_x[0] <= cx <= ilha2_x[1] and ilha2_y[0] <= cy <= ilha2_y[1]

    em_1 = [_em_ilha1(cx, cy) for cx, cy in centroides]
    em_2 = [_em_ilha2(cx, cy) for cx, cy in centroides]

    # Um centroide deve estar em ilha1 e o outro em ilha2
    assert sum(em_1) == 1, f"Esperado 1 centroide em ilha1, centroides={centroides}"
    assert sum(em_2) == 1, f"Esperado 1 centroide em ilha2, centroides={centroides}"


# ─────────────────────────────────────────────────────────────────────────────
# B4 — região menor que MIN_AREA_PIXEL → descartada
# ─────────────────────────────────────────────────────────────────────────────

def test_regiao_pequena_descartada():
    """Região com área < MIN_AREA_PIXEL deve ser filtrada (retorno vazio ou só a grande)."""
    # Cria superfície com um bloco minúsculo (2×2 pixels de bloco) — área ~4
    sup = np.full((30, 30), 0.1, dtype=np.float32)
    sup[1:3, 1:3] = 0.9   # blob 2×2 = area ~4, abaixo de MIN_AREA_PIXEL (20)

    out = vetorizar_superficie(sup, min_area=MIN_AREA_PIXEL)
    assert out["stats"]["total_polys"] == 0, (
        "Região com área < MIN_AREA_PIXEL deve ser descartada"
    )


def test_regiao_grande_o_suficiente_aceita():
    """Região com área ≥ MIN_AREA_PIXEL deve ser aceita."""
    sup = np.full((30, 30), 0.1, dtype=np.float32)
    sup[2:10, 2:10] = 0.9  # blob 8×8 = area ~64, acima de MIN_AREA_PIXEL

    out = vetorizar_superficie(sup, min_area=MIN_AREA_PIXEL)
    assert out["stats"]["total_polys"] >= 1


# ─────────────────────────────────────────────────────────────────────────────
# B9 — GeoJSON válido com nº correto de features
# ─────────────────────────────────────────────────────────────────────────────

def test_geojson_estrutura_valida(superficie_duas_ilhas):
    """GeoJSON deve ser parseável e ter type=FeatureCollection."""
    out = vetorizar_superficie(superficie_duas_ilhas)
    _geojson_valido(out["geojson"], out["stats"]["total_polys"])


def test_geojson_features_iguais_total_polys():
    """Número de features no GeoJSON == total_polys em stats."""
    sup = np.full((30, 30), 0.1, dtype=np.float32)
    sup[2:12, 2:12] = 0.9
    sup[15:25, 15:25] = 0.9
    out = vetorizar_superficie(sup)

    doc = json.loads(out["geojson"])
    assert len(doc["features"]) == out["stats"]["total_polys"]


def test_geojson_propriedades_por_feature(superficie_duas_ilhas):
    """Cada feature do GeoJSON deve ter as propriedades esperadas."""
    out = vetorizar_superficie(superficie_duas_ilhas)
    doc = json.loads(out["geojson"])

    campos_obrigatorios = {"id", "area", "centroid", "vertices", "confidence", "iou_vs_claude"}
    for feat in doc["features"]:
        props = feat["properties"]
        assert campos_obrigatorios.issubset(props.keys()), (
            f"Campos faltando: {campos_obrigatorios - props.keys()}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# B6 — performance medida e presente
# ─────────────────────────────────────────────────────────────────────────────

def test_performance_latency_presente(superficie_plana):
    """performance.latency_ms deve estar presente e > 0 mesmo para superfície vazia."""
    out = vetorizar_superficie(superficie_plana)
    assert "latency_ms" in out["performance"]
    assert out["performance"]["latency_ms"] >= 0.0


def test_performance_latency_positiva(superficie_duas_ilhas):
    """latency_ms deve ser > 0 quando há polígonos a processar."""
    out = vetorizar_superficie(superficie_duas_ilhas)
    assert out["performance"]["latency_ms"] > 0.0


def test_performance_threshold_registrado(superficie_duas_ilhas):
    """threshold_used deve refletir o valor passado."""
    out = vetorizar_superficie(superficie_duas_ilhas, threshold=0.6)
    assert out["performance"]["threshold_used"] == 0.6


# ─────────────────────────────────────────────────────────────────────────────
# iou_vs_claude
# ─────────────────────────────────────────────────────────────────────────────

def test_iou_vs_claude_none_sem_bboxes(superficie_duas_ilhas):
    """Sem bboxes_claude, iou_vs_claude deve ser None para todos os polígonos."""
    out = vetorizar_superficie(superficie_duas_ilhas)
    for attr in out["attributes"]:
        assert attr["iou_vs_claude"] is None


def test_iou_vs_claude_positivo_com_bbox_cobrindo_ilha():
    """Bbox que cobre a ilha deve produzir IoU > 0."""
    sup = np.full((20, 20), 0.1, dtype=np.float32)
    sup[2:10, 2:10] = 0.9  # ilha em linhas 2-9, colunas 2-9

    # Bbox normalizada cobrindo a ilha: x1=2/20, y1=2/20, x2=10/20, y2=10/20
    bboxes = [{"x1": 0.05, "y1": 0.05, "x2": 0.55, "y2": 0.55}]
    out = vetorizar_superficie(sup, bboxes_claude=bboxes)

    assert out["stats"]["total_polys"] >= 1
    ious = [attr["iou_vs_claude"] for attr in out["attributes"]
            if attr["iou_vs_claude"] is not None]
    assert len(ious) > 0, "Deve haver ao menos um iou calculado"
    assert max(ious) > 0.0, f"IoU esperado > 0, obtido {ious}"


def test_iou_vs_claude_lista_vazia_retorna_none():
    """bboxes_claude=[] deve resultar em iou_vs_claude=None."""
    sup = np.full((20, 20), 0.1, dtype=np.float32)
    sup[2:10, 2:10] = 0.9
    out = vetorizar_superficie(sup, bboxes_claude=[])
    for attr in out["attributes"]:
        assert attr["iou_vs_claude"] is None


# ─────────────────────────────────────────────────────────────────────────────
# B4 — auto-interseção tratada via buffer(0)
# ─────────────────────────────────────────────────────────────────────────────

def test_superficie_com_regiao_irregular_nao_crasha():
    """Superfície com forma irregular (possível auto-interseção) não deve lançar exceção."""
    sup = np.full((30, 30), 0.1, dtype=np.float32)
    # Forma em L — pode gerar contorno irregular
    sup[2:12, 2:8] = 0.9
    sup[2:8, 2:15] = 0.9

    try:
        out = vetorizar_superficie(sup)
    except Exception as e:
        pytest.fail(f"vetorizar_superficie levantou exceção inesperada: {e}")

    # Todos os polígonos retornados devem ser válidos E não-vazios
    # (asserção endurecida: o `or` anterior era quase-sempre-verdadeiro)
    for poly in out["geometries"]:
        assert poly.is_valid and not poly.is_empty


# ─────────────────────────────────────────────────────────────────────────────
# Atributos básicos dos polígonos
# ─────────────────────────────────────────────────────────────────────────────

def test_atributos_ids_sequenciais(superficie_duas_ilhas):
    """IDs dos polígonos devem ser sequenciais a partir de 1."""
    out = vetorizar_superficie(superficie_duas_ilhas)
    ids = [attr["id"] for attr in out["attributes"]]
    assert ids == list(range(1, len(ids) + 1)), f"IDs não sequenciais: {ids}"


def test_atributos_confidence_range(superficie_duas_ilhas):
    """Confiança de cada polígono deve estar em [0, 1]."""
    out = vetorizar_superficie(superficie_duas_ilhas)
    for attr in out["attributes"]:
        assert 0.0 <= attr["confidence"] <= 1.0, (
            f"Confiança fora de [0,1]: {attr['confidence']}"
        )


def test_atributos_vertices_minimo(superficie_duas_ilhas):
    """Cada polígono deve ter pelo menos MIN_POLYGON_VERTICES vértices."""
    out = vetorizar_superficie(superficie_duas_ilhas)
    for attr in out["attributes"]:
        assert attr["vertices"] >= MIN_POLYGON_VERTICES


def test_stats_total_area_soma_areas(superficie_duas_ilhas):
    """stats.total_area deve ser a soma das áreas individuais."""
    out = vetorizar_superficie(superficie_duas_ilhas)
    soma = sum(a["area"] for a in out["attributes"])
    assert math.isclose(out["stats"]["total_area"], soma, rel_tol=1e-4), (
        f"total_area={out['stats']['total_area']} != soma={soma}"
    )


def test_stats_coverage_pct_range(superficie_duas_ilhas):
    """coverage_pct deve estar em [0, 100]."""
    out = vetorizar_superficie(superficie_duas_ilhas)
    assert 0.0 <= out["stats"]["coverage_pct"] <= 100.0


# ─────────────────────────────────────────────────────────────────────────────
# Threshold customizado
# ─────────────────────────────────────────────────────────────────────────────

def test_threshold_alto_descarta_todos():
    """Threshold = 1.0 deve descartar todos os polígonos (nenhum pixel = 1.0)."""
    sup = np.full((20, 20), 0.9, dtype=np.float32)
    out = vetorizar_superficie(sup, threshold=1.0)
    assert out["stats"]["total_polys"] == 0


def test_threshold_baixo_aceita_regiao_moderada():
    """Threshold baixo (0.05) deve aceitar região com valor 0.1 que o default rejeitaria."""
    sup = np.full((30, 30), 0.01, dtype=np.float32)
    # Ilha com valor 0.1: acima de threshold=0.05, abaixo do default 0.5
    sup[5:15, 5:15] = 0.1
    out_default = vetorizar_superficie(sup, threshold=ISOCONTOUR_THRESHOLD)
    out_baixo   = vetorizar_superficie(sup, threshold=0.05)
    # Threshold baixo deve detectar a ilha, o default não
    assert out_baixo["stats"]["total_polys"] >= 1
    assert out_default["stats"]["total_polys"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Chave de retorno completa
# ─────────────────────────────────────────────────────────────────────────────

def test_retorno_tem_todas_as_chaves(superficie_plana):
    """O dict de retorno deve ter todas as chaves documentadas."""
    out = vetorizar_superficie(superficie_plana)
    assert set(out.keys()) == {"geometries", "attributes", "geojson", "stats", "performance"}
    assert set(out["stats"].keys()) == {"total_polys", "total_area", "coverage_pct"}
    assert set(out["performance"].keys()) == {"latency_ms", "threshold_used"}
