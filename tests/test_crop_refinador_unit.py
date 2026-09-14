"""Testes unitários para o core/crop_refinador.py."""
import numpy as np
import pytest
from pathlib import Path
from PIL import Image

from core.crop_refinador import (
    _bbox_tocando_bordas,
    _expandir_bbox_por_corte,
    refinar_crop,
)


def test_bbox_tocando_bordas():
    # Bbox centralizada (não toca nenhuma borda)
    bbox_meio = {"x1": 0.2, "y1": 0.2, "x2": 0.8, "y2": 0.8}
    assert _bbox_tocando_bordas(bbox_meio, (100, 100)) == []

    # Bbox colada na esquerda e topo
    bbox_canto = {"x1": 0.02, "y1": 0.01, "x2": 0.5, "y2": 0.5}
    lados = _bbox_tocando_bordas(bbox_canto, (100, 100), tolerancia=0.05)
    assert "left" in lados
    assert "top" in lados
    assert "right" not in lados
    assert "bottom" not in lados

    # Bbox colada na direita e embaixo
    bbox_fim = {"x1": 0.5, "y1": 0.5, "x2": 0.98, "y2": 0.99}
    lados_fim = _bbox_tocando_bordas(bbox_fim, (100, 100), tolerancia=0.05)
    assert "right" in lados_fim
    assert "bottom" in lados_fim


def test_expandir_bbox_por_corte():
    bbox = {"x1": 0.2, "y1": 0.2, "x2": 0.8, "y2": 0.8}
    W, H = 200, 100
    expandida = _expandir_bbox_por_corte(bbox, W, H, ["left", "right"], expansao=0.10)
    assert expandida["x1"] < bbox["x1"]
    assert expandida["x2"] > bbox["x2"]
    assert expandida["y1"] == bbox["y1"]
    assert expandida["y2"] == bbox["y2"]


def test_refinar_crop_fast_path(tmp_path):
    # Cria imagem temporária
    img_path = tmp_path / "teste_foto.jpg"
    img = Image.new("RGB", (200, 200), color=(128, 128, 128))
    img.save(img_path)

    obj = {
        "nome": "martelo",
        "bbox_normalizada": {"x1": 0.1, "y1": 0.1, "x2": 0.6, "y2": 0.6},
        "confianca": 0.92,
    }
    saida_path = tmp_path / "crop_out.jpg"

    saida, tag = refinar_crop(
        caminho_foto=str(img_path),
        obj=obj,
        saida=str(saida_path),
        numero=1,
        total_objetos=1,
        nome="martelo",
    )

    assert tag == "S1_bbox_direta"
    assert Path(saida).exists()
    assert saida_path.exists()


def test_refinar_crop_fallback_quando_confianca_baixa(tmp_path, mocker):
    img_path = tmp_path / "teste_foto2.jpg"
    img = Image.new("RGB", (200, 200), color=(100, 150, 200))
    img.save(img_path)

    # Mock da chamada do Claude para Stage 2
    mock_claude = mocker.patch(
        "core.crop_refinador._chamar_claude",
        return_value={
            "qualidade_crop": "ok",
            "confianca": 0.88,
            "bbox_normalizada": {"x1": 0.1, "y1": 0.1, "x2": 0.8, "y2": 0.8},
        }
    )

    obj = {
        "nome": "chave",
        "bbox_normalizada": {"x1": 0.2, "y1": 0.2, "x2": 0.7, "y2": 0.7},
        "confianca": 0.45,  # < 0.70 força chamada ao Stage 2
    }
    saida_path = tmp_path / "crop_fallback.jpg"

    saida, tag = refinar_crop(
        caminho_foto=str(img_path),
        obj=obj,
        saida=str(saida_path),
        numero=1,
        total_objetos=1,
        nome="chave",
    )

    assert tag in ("S2_bbox", "S3_bbox")
    assert mock_claude.called
    assert Path(saida).exists()
