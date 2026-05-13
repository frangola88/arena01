"""
Testes unitários para agent_1_segmentador — segmentação em 2 etapas.

Foco: comportamento isolado das funções principais.
"""
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from PIL import Image

from agents.agent_1_segmentador import _recortar


@pytest.fixture
def temp_recortes_dir():
    """Diretório temporário para recortes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# Testes: _recortar()
# ============================================================================


def test_recortar_cria_arquivo_jpeg(temp_recortes_dir):
    """_recortar() cria arquivo JPEG com as coordenadas normalizadas."""
    with patch("agents.agent_1_segmentador.RECORTES_DIR", temp_recortes_dir):
        imagem = Image.new("RGB", (400, 300), color=(255, 0, 0))
        bbox = {"x1": 0.1, "y1": 0.2, "x2": 0.5, "y2": 0.8}
        
        caminho = _recortar(imagem, bbox, foto_id=1, idx=0)
        
        assert Path(caminho).exists()
        assert Path(caminho).suffix == ".jpg"
        assert "foto_1_obj_00" in caminho


def test_recortar_clipa_bordas(temp_recortes_dir):
    """_recortar() clipa coordenadas fora da imagem."""
    with patch("agents.agent_1_segmentador.RECORTES_DIR", temp_recortes_dir):
        imagem = Image.new("RGB", (400, 300), color=(0, 255, 0))
        bbox = {"x1": -0.1, "y1": -0.1, "x2": 1.5, "y2": 1.5}
        
        caminho = _recortar(imagem, bbox, foto_id=2, idx=0)
        
        recorte = Image.open(caminho)
        assert recorte.size[0] == 400
        assert recorte.size[1] == 300


def test_recortar_minimo_tamanho(temp_recortes_dir):
    """_recortar() garante mínimo de 20px."""
    with patch("agents.agent_1_segmentador.RECORTES_DIR", temp_recortes_dir):
        imagem = Image.new("RGB", (400, 300), color=(0, 0, 255))
        bbox = {"x1": 0.5, "y1": 0.5, "x2": 0.51, "y2": 0.51}
        
        caminho = _recortar(imagem, bbox, foto_id=3, idx=0)
        
        recorte = Image.open(caminho)
        assert recorte.size[0] >= 20
        assert recorte.size[1] >= 20


def test_recortar_diferentes_indices(temp_recortes_dir):
    """_recortar() gera nomes diferentes para índices diferentes."""
    with patch("agents.agent_1_segmentador.RECORTES_DIR", temp_recortes_dir):
        imagem = Image.new("RGB", (100, 100), color=(100, 100, 100))
        bbox = {"x1": 0.1, "y1": 0.1, "x2": 0.5, "y2": 0.5}
        
        caminho0 = _recortar(imagem, bbox, foto_id=5, idx=0)
        caminho1 = _recortar(imagem, bbox, foto_id=5, idx=1)
        
        assert "obj_00" in caminho0
        assert "obj_01" in caminho1
        assert caminho0 != caminho1
