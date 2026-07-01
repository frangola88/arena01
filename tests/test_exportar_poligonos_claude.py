"""
Testes de core/exportar_poligonos_claude.py.

Isola a dependência de core.alpha_pipeline.processar_imagem (U2Net/ONNX) via
monkeypatch — testa só a normalização de coordenadas e a exportação dos
artefatos (PNG + JSON), sem exigir o modelo ONNX no ambiente de CI.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from shapely.geometry import box

import core.exportar_poligonos_claude as mod
from core.exportar_poligonos_claude import gerar_pacote_claude, _normalizar_poligonos


GRID = 100


def _resultado_fake():
    """Resultado sintético equivalente ao retorno de processar_imagem: 2
    polígonos (um centralizado, um encostado na borda) num grid 100x100."""
    poly_a = box(10, 20, 30, 40)   # área 400, centroide (20,30)
    poly_b = box(90, 90, 100, 100)  # área 100, encostado no canto (borda)

    return {
        "alpha_grid": np.zeros((GRID, GRID), dtype=np.float32),
        "sup_morph":  np.zeros((GRID, GRID), dtype=np.float32),
        "poligonos": [
            {"geometry": poly_a, "area": poly_a.area, "n_vertices": 5,
             "centroid": (poly_a.centroid.x, poly_a.centroid.y)},
            {"geometry": poly_b, "area": poly_b.area, "n_vertices": 5,
             "centroid": (poly_b.centroid.x, poly_b.centroid.y)},
        ],
        "n_poligonos": 2,
        "area_total": poly_a.area + poly_b.area,
        "cobertura_pct": 100.0 * (poly_a.area + poly_b.area) / (GRID * GRID),
        "tempo_ms": {"u2net": 500.0, "morph": 0.2, "vetorizar": 1.0,
                     "simplify": 0.5, "total": 501.7},
    }


@pytest.fixture(autouse=True)
def _mock_processar_imagem(monkeypatch):
    """Substitui a chamada real ao U2Net por um resultado determinístico."""
    monkeypatch.setattr(mod, "processar_imagem", lambda *a, **k: _resultado_fake())


# ─────────────────────────────────────────────────────────────────────────────
# _normalizar_poligonos — coordenadas devem cair em [0,1]
# ─────────────────────────────────────────────────────────────────────────────

def test_normalizar_poligonos_bbox_em_0_1():
    resultado = _resultado_fake()
    norm = _normalizar_poligonos(resultado["poligonos"], GRID)

    assert len(norm) == 2
    for p in norm:
        for k in ("x1", "y1", "x2", "y2"):
            assert 0.0 <= p["bbox"][k] <= 1.0, f"{k}={p['bbox'][k]} fora de [0,1]"
        assert 0.0 <= p["centroid"]["x"] <= 1.0
        assert 0.0 <= p["centroid"]["y"] <= 1.0


def test_normalizar_poligonos_bbox_correto():
    """poly_a = box(10,20,30,40) em grid 100 -> bbox normalizado exato."""
    resultado = _resultado_fake()
    norm = _normalizar_poligonos(resultado["poligonos"], GRID)

    bbox_a = norm[0]["bbox"]
    assert bbox_a == {"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4}


def test_normalizar_poligonos_area_pct_condiz_com_area_pixel():
    resultado = _resultado_fake()
    norm = _normalizar_poligonos(resultado["poligonos"], GRID)

    # poly_a: área 400 px² num grid 100x100=10000 px² -> 4%
    assert norm[0]["area_pct"] == pytest.approx(4.0)
    # poly_b: área 100 px² -> 1%
    assert norm[1]["area_pct"] == pytest.approx(1.0)


def test_normalizar_poligonos_ids_sequenciais_desde_1():
    resultado = _resultado_fake()
    norm = _normalizar_poligonos(resultado["poligonos"], GRID)
    assert [p["id"] for p in norm] == [1, 2]


# ─────────────────────────────────────────────────────────────────────────────
# gerar_pacote_claude — artefatos gravados em disco
# ─────────────────────────────────────────────────────────────────────────────

def test_gerar_pacote_cria_png_e_json(tmp_path):
    out = gerar_pacote_claude("fake_img.jpg", str(tmp_path))

    assert (tmp_path / "poligonos.png").exists()
    assert (tmp_path / "poligonos.json").exists()
    assert out["png_path"] == str(tmp_path / "poligonos.png")
    assert out["json_path"] == str(tmp_path / "poligonos.json")


def test_gerar_pacote_json_tem_estrutura_esperada(tmp_path):
    out = gerar_pacote_claude("fake_img.jpg", str(tmp_path))

    with open(out["json_path"]) as f:
        payload = json.load(f)

    assert payload["imagem_origem"] == "fake_img.jpg"
    assert payload["grid"] == 280  # default de gerar_pacote_claude
    assert payload["n_poligonos"] == 2
    assert len(payload["poligonos"]) == 2
    assert "tempo_ms" in payload
    assert "cobertura_pct" in payload


def test_gerar_pacote_retorna_poligonos_normalizados(tmp_path):
    out = gerar_pacote_claude("fake_img.jpg", str(tmp_path))

    assert out["poligonos"] == _normalizar_poligonos(
        _resultado_fake()["poligonos"], grid=280
    )


def test_gerar_pacote_resumo_bate_com_resultado(tmp_path):
    out = gerar_pacote_claude("fake_img.jpg", str(tmp_path))

    assert out["resumo"]["n_poligonos"] == 2
    assert out["resumo"]["cobertura_pct"] == pytest.approx(5.0)  # (400+100)/10000*100


def test_gerar_pacote_cria_out_dir_se_nao_existir(tmp_path):
    novo_dir = tmp_path / "subdir_inexistente"
    assert not novo_dir.exists()

    gerar_pacote_claude("fake_img.jpg", str(novo_dir))
    assert novo_dir.exists()
    assert (novo_dir / "poligonos.json").exists()
