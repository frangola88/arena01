"""
Testes unitários para agent_4_icone — 5 estratégias em cascata.

Cobertura: cada estratégia isolada + fluxo completo + edge cases.
"""
import io
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
from PIL import Image

from agents.agent_4_icone import (
    _recorte_como_icone,
    _buscar_imagem_web,
    _executar_desenho_PIL,
    _claude_desenha,
    _placeholder_PIL,
    _nome_util,
    gerar_icone,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_icone_dir():
    """Diretório temporário para ícones."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def imagem_teste(temp_icone_dir):
    """Cria uma imagem de teste 100x100 RGB."""
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    path = temp_icone_dir / "teste.png"
    img.save(str(path), "PNG")
    return str(path)


@pytest.fixture
def imagem_recorte(temp_icone_dir):
    """Cria um recorte de teste."""
    img = Image.new("RGB", (128, 128), color=(0, 255, 0))
    path = temp_icone_dir / "recorte.png"
    img.save(str(path), "PNG")
    return str(path)


# ============================================================================
# Testes: _nome_util()
# ============================================================================

class TestNomeUtil:
    """Validação de nomes úteis (não genéricos)."""

    def test_nome_util_valido(self):
        assert _nome_util("Chave de fenda") is True
        assert _nome_util("Livro de Python") is True
        assert _nome_util("Martelo amarelo") is True

    def test_nome_util_generico(self):
        assert _nome_util("objeto") is False
        assert _nome_util("coisa") is False
        assert _nome_util("item") is False
        assert _nome_util("peca") is False
        assert _nome_util("varios") is False

    def test_nome_util_muito_curto(self):
        assert _nome_util("abc") is False
        assert _nome_util("") is False


# ============================================================================
# Testes: Estratégia 1 — recorte_como_icone()
# ============================================================================

class TestRecorteComoIcone:
    """Redimensionar recorte para 256x256."""

    def test_recorte_valido(self, imagem_recorte, temp_icone_dir):
        saida = temp_icone_dir / "icone.png"
        ok = _recorte_como_icone(imagem_recorte, str(saida))
        assert ok is True
        assert saida.exists()
        img = Image.open(str(saida))
        assert img.size == (256, 256)

    def test_recorte_nao_existe(self, temp_icone_dir):
        saida = temp_icone_dir / "icone.png"
        ok = _recorte_como_icone("/nao/existe/recorte.png", str(saida))
        assert ok is False
        assert not saida.exists()

    def test_recorte_arquivo_corrupto(self, temp_icone_dir):
        corrupto = temp_icone_dir / "corrupto.png"
        corrupto.write_text("not a valid image")
        saida = temp_icone_dir / "icone.png"
        ok = _recorte_como_icone(str(corrupto), str(saida))
        assert ok is False


# ============================================================================
# Testes: Estratégia 2 — buscar_imagem_web()
# ============================================================================

class TestBuscarImagemWeb:
    """DuckDuckGo Images com resiliência."""

    @patch("ddgs.DDGS")
    @patch("requests.get")
    def test_busca_web_sucesso(self, mock_get, mock_ddgs, temp_icone_dir):
        # Mock do DDGS — usar side_effect com função que retorna novo generator
        def mock_images_gen(*args, **kwargs):
            yield {"image": "https://example.com/img1.jpg"}
            yield {"image": "https://example.com/img2.jpg"}
        
        mock_ddgs.return_value.__enter__.return_value.images.side_effect = lambda *a, **k: mock_images_gen()
        
        # Mock da resposta HTTP — precisa ser PNG válido com > 2000 bytes
        # Criar imagem com gradiente para garantir tamanho > 2000 bytes
        img = Image.new("RGB", (1024, 1024))
        pixels = img.load()
        for i in range(1024):
            for j in range(1024):
                pixels[i, j] = (i % 256, j % 256, (i+j) % 256)  # Gradiente multicolor
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = img_bytes.getvalue()
        mock_get.return_value = mock_response

        saida = temp_icone_dir / "icone.png"
        ok = _buscar_imagem_web("Chave de fenda", str(saida))
        assert ok is True
        assert saida.exists()
        mock_get.assert_called()

    @patch("ddgs.DDGS")
    def test_busca_web_ddgs_falha(self, mock_ddgs, temp_icone_dir):
        mock_ddgs.return_value.__enter__.return_value.images.side_effect = Exception("API erro")
        saida = temp_icone_dir / "icone.png"
        ok = _buscar_imagem_web("Chave de fenda", str(saida))
        assert ok is False

    @patch("ddgs.DDGS")
    @patch("requests.get")
    def test_busca_web_conteudo_pequeno(self, mock_get, mock_ddgs, temp_icone_dir):
        # Conteúdo muito pequeno (< 2000 bytes)
        mock_ddgs.return_value.__enter__.return_value.images.return_value = [
            {"image": "https://example.com/img.jpg"},
        ]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"x" * 100  # Muito pequeno
        mock_get.return_value = mock_response

        saida = temp_icone_dir / "icone.png"
        ok = _buscar_imagem_web("Chave de fenda", str(saida))
        assert ok is False

    @patch("ddgs.DDGS")
    @patch("requests.get")
    def test_busca_web_tenta_multiplas_urls(self, mock_get, mock_ddgs, temp_icone_dir):
        # Primeira falha, segunda sucede
        def mock_images_gen2(*args, **kwargs):
            yield {"image": "https://example.com/broken.jpg"}
            yield {"image": "https://example.com/ok.jpg"}
        
        mock_ddgs.return_value.__enter__.return_value.images.side_effect = lambda *a, **k: mock_images_gen2()
        
        # Primeira falha, segunda sucede — gerar PNG válido com > 2000 bytes
        # Criar imagem com gradiente para garantir tamanho > 2000 bytes
        img = Image.new("RGB", (1024, 1024))
        pixels = img.load()
        for i in range(1024):
            for j in range(1024):
                pixels[i, j] = (i % 256, j % 256, (i+j) % 256)  # Gradiente multicolor
        img_bytes_io = io.BytesIO()
        img.save(img_bytes_io, format="PNG")
        img_bytes_io.seek(0)
        mock_get.side_effect = [
            Exception("timeout"),
            MagicMock(status_code=200, content=img_bytes_io.getvalue()),
        ]

        saida = temp_icone_dir / "icone.png"
        ok = _buscar_imagem_web("Produto", str(saida))
        assert ok is True


# ============================================================================
# Testes: Estratégia 3 — _executar_desenho_PIL()
# ============================================================================

class TestExecutarDesenhoPIL:
    """Renderizar JSON de formas geométricas em PNG."""

    def test_desenho_retangulo(self, temp_icone_dir):
        instrucoes = {
            "fundo": [255, 255, 255],
            "formas": [
                {"tipo": "retangulo", "x1": 50, "y1": 50, "x2": 200, "y2": 150,
                 "cor": [0, 0, 0], "espessura": 2},
            ],
        }
        saida = temp_icone_dir / "icone.png"
        ok = _executar_desenho_PIL(instrucoes, str(saida))
        assert ok is True
        assert saida.exists()
        img = Image.open(str(saida))
        assert img.size == (256, 256)

    def test_desenho_elipse(self, temp_icone_dir):
        instrucoes = {
            "fundo": [255, 255, 255],
            "formas": [
                {"tipo": "elipse", "x1": 50, "y1": 50, "x2": 200, "y2": 150,
                 "cor": [255, 0, 0], "preenchido": True},
            ],
        }
        saida = temp_icone_dir / "icone.png"
        ok = _executar_desenho_PIL(instrucoes, str(saida))
        assert ok is True

    def test_desenho_linha(self, temp_icone_dir):
        instrucoes = {
            "fundo": [255, 255, 255],
            "formas": [
                {"tipo": "linha", "x1": 10, "y1": 10, "x2": 250, "y2": 250,
                 "cor": [0, 0, 255], "espessura": 3},
            ],
        }
        saida = temp_icone_dir / "icone.png"
        ok = _executar_desenho_PIL(instrucoes, str(saida))
        assert ok is True

    def test_desenho_texto(self, temp_icone_dir):
        instrucoes = {
            "fundo": [255, 255, 255],
            "formas": [
                {"tipo": "texto", "x": 128, "y": 128, "texto": "A",
                 "cor": [0, 0, 0], "ancora": "mm"},
            ],
        }
        saida = temp_icone_dir / "icone.png"
        ok = _executar_desenho_PIL(instrucoes, str(saida))
        assert ok is True

    def test_desenho_poligono(self, temp_icone_dir):
        instrucoes = {
            "fundo": [255, 255, 255],
            "formas": [
                {"tipo": "poligono", "pontos": [50, 50, 200, 50, 125, 150],
                 "cor": [0, 255, 0]},
            ],
        }
        saida = temp_icone_dir / "icone.png"
        ok = _executar_desenho_PIL(instrucoes, str(saida))
        assert ok is True

    def test_desenho_invalido(self, temp_icone_dir):
        instrucoes = {"fundo": "nao_e_lista"}
        saida = temp_icone_dir / "icone.png"
        ok = _executar_desenho_PIL(instrucoes, str(saida))
        assert ok is False


# ============================================================================
# Testes: Estratégia 4 — _claude_desenha()
# ============================================================================

class TestClaudeDesenha:
    """Claude gera JSON, PIL executa."""

    @patch("agents.agent_4_icone.ANTHROPIC_API_KEY", "fake-key")
    @patch("anthropic.Anthropic")
    @patch("agents.agent_4_icone._executar_desenho_PIL")
    def test_claude_desenha_sucesso(self, mock_pil, mock_client_class, temp_icone_dir):
        # Mock da resposta do Claude
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"fundo":[255,255,255],"formas":[]}')]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Mock do PIL
        mock_pil.return_value = True

        saida = temp_icone_dir / "icone.png"
        ok = _claude_desenha("Chave de fenda", str(saida))
        assert ok is True
        mock_client.messages.create.assert_called()

    @patch("agents.agent_4_icone.ANTHROPIC_API_KEY", None)
    def test_claude_desenha_sem_api_key(self, temp_icone_dir):
        saida = temp_icone_dir / "icone.png"
        ok = _claude_desenha("Chave de fenda", str(saida))
        assert ok is False

    @patch("agents.agent_4_icone.ANTHROPIC_API_KEY", "fake-key")
    @patch("anthropic.Anthropic")
    def test_claude_desenha_api_erro(self, mock_client_class, temp_icone_dir):
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API error")
        mock_client_class.return_value = mock_client

        saida = temp_icone_dir / "icone.png"
        ok = _claude_desenha("Chave de fenda", str(saida))
        assert ok is False


# ============================================================================
# Testes: Estratégia 5 — _placeholder_PIL()
# ============================================================================

class TestPlaceholderPIL:
    """Fallback final que sempre funciona."""

    def test_placeholder_com_emoji(self, temp_icone_dir):
        saida = temp_icone_dir / "icone.png"
        ok = _placeholder_PIL("Chave de fenda", "🔧", "Manutenção", str(saida))
        assert ok is True
        assert saida.exists()
        img = Image.open(str(saida))
        assert img.size == (256, 256)

    def test_placeholder_nome_longo(self, temp_icone_dir):
        saida = temp_icone_dir / "icone.png"
        nome_longo = "A" * 50
        ok = _placeholder_PIL(nome_longo, "📦", "Geral", str(saida))
        assert ok is True

    def test_placeholder_cores_diferentes(self, temp_icone_dir):
        for grupo in ["Casa", "Tecnologia", "Saúde", "Infantil"]:
            saida = temp_icone_dir / f"icone_{grupo}.png"
            ok = _placeholder_PIL(f"Objeto {grupo}", "📦", grupo, str(saida))
            assert ok is True


# ============================================================================
# Testes: Fluxo em cascata — gerar_icone()
# ============================================================================

class TestGerarIcone:
    """Fluxo completo em cascata."""

    def test_cascata_recorte_alta_confianca(self, imagem_recorte, temp_icone_dir, monkeypatch):
        # Mock ICONES_DIR para usar temp_icone_dir
        monkeypatch.setattr("agents.agent_4_icone.ICONES_DIR", temp_icone_dir)

        caminho, fonte = gerar_icone(
            recorte_path=imagem_recorte,
            nome="Chave de fenda",
            categoria_nome="Ferramentas",
            icone_emoji="🔧",
            grupo="Manutenção",
            confianca=0.80,
            objeto_id=1,
        )
        assert Path(caminho).exists()
        assert fonte == "recorte"

    def test_cascata_recorte_falha_web_sucede(self, temp_icone_dir, monkeypatch):
        monkeypatch.setattr("agents.agent_4_icone.ICONES_DIR", temp_icone_dir)

        # Mock: recorte falha, web sucede
        with patch("agents.agent_4_icone._recorte_como_icone", return_value=False):
            with patch("agents.agent_4_icone._buscar_imagem_web", return_value=True):
                caminho, fonte = gerar_icone(
                    recorte_path="/inexistente.png",
                    nome="Chave de fenda",
                    categoria_nome="Ferramentas",
                    icone_emoji="🔧",
                    grupo="Manutenção",
                    confianca=0.70,
                    objeto_id=2,
                )
                assert fonte == "web"

    def test_cascata_tudo_falha_placeholder(self, temp_icone_dir, monkeypatch):
        monkeypatch.setattr("agents.agent_4_icone.ICONES_DIR", temp_icone_dir)

        # Mock: tudo falha
        with patch("agents.agent_4_icone._recorte_como_icone", return_value=False):
            with patch("agents.agent_4_icone._buscar_imagem_web", return_value=False):
                with patch("agents.agent_4_icone._claude_desenha", return_value=False):
                    caminho, fonte = gerar_icone(
                        recorte_path="/inexistente.png",
                        nome="Objeto desconhecido",
                        categoria_nome="Geral",
                        icone_emoji="📦",
                        grupo="Geral",
                        confianca=0.20,
                        objeto_id=3,
                    )
                    assert fonte == "placeholder"
                    assert Path(caminho).exists()

    def test_cascata_confianca_baixa_pula_recorte1(self, temp_icone_dir, monkeypatch):
        monkeypatch.setattr("agents.agent_4_icone.ICONES_DIR", temp_icone_dir)

        # Confiança < 0.75: pula estratégia 1
        with patch("agents.agent_4_icone._recorte_como_icone") as mock_recorte:
            with patch("agents.agent_4_icone._buscar_imagem_web", return_value=True):
                gerar_icone(
                    recorte_path="/existe.png",
                    nome="Produto válido",
                    categoria_nome="Casa",
                    icone_emoji="🏠",
                    grupo="Casa",
                    confianca=0.50,
                    objeto_id=4,
                )
                # _recorte_como_icone não deve ser chamado (confiança < 0.75)
                assert not mock_recorte.called

    def test_cascata_nome_generico_pula_web(self, temp_icone_dir, monkeypatch):
        monkeypatch.setattr("agents.agent_4_icone.ICONES_DIR", temp_icone_dir)

        # Nome genérico: pula web e Claude
        with patch("agents.agent_4_icone._recorte_como_icone", return_value=False):
            with patch("agents.agent_4_icone._buscar_imagem_web") as mock_web:
                with patch("agents.agent_4_icone._claude_desenha") as mock_claude:
                    gerar_icone(
                        recorte_path="/existe.png",
                        nome="coisa",  # genérico
                        categoria_nome="Geral",
                        icone_emoji="?",
                        grupo="Geral",
                        confianca=0.70,
                        objeto_id=5,
                    )
                    assert not mock_web.called
                    assert not mock_claude.called

    def test_cascata_ordem_correta(self, temp_icone_dir, monkeypatch):
        """Verifica que a ordem de tentativa é: recorte > web > recorte_baixa > claude > placeholder."""
        monkeypatch.setattr("agents.agent_4_icone.ICONES_DIR", temp_icone_dir)

        ordem_chamadas = []

        def mock_recorte(path, saida):
            ordem_chamadas.append("recorte")
            return False

        def mock_web(nome, saida):
            ordem_chamadas.append("web")
            return False

        def mock_claude(nome, saida):
            ordem_chamadas.append("claude")
            return False

        with patch("agents.agent_4_icone._recorte_como_icone", side_effect=mock_recorte):
            with patch("agents.agent_4_icone._buscar_imagem_web", side_effect=mock_web):
                with patch("agents.agent_4_icone._claude_desenha", side_effect=mock_claude):
                    gerar_icone(
                        recorte_path="/existe.png",
                        nome="Produto válido",
                        categoria_nome="Casa",
                        icone_emoji="📦",
                        grupo="Casa",
                        confianca=0.80,
                        objeto_id=6,
                    )

        # Verifica ordem: recorte (alta), web, recorte (baixa), claude
        assert ordem_chamadas == ["recorte", "web", "recorte", "claude"]
