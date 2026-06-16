"""
Testes unitários para agent_1_segmentador v5 — centroide-matching.

Testa as funções auxiliares puras sem dependência de OpenCV, PIL ou Claude.
"""
import pytest

# Importa diretamente só as funções que não dependem de cv2 no nível de módulo
# usando importação lazy para evitar o cv2 na carga do módulo.
import importlib, sys


def _get_fn(name):
    """Importa funções do agent_1 sem acionar o import de cv2 no topo."""
    # Substitui temporariamente cv2 por um mock no sys.modules
    import types
    cv2_mock = types.ModuleType("cv2")
    numpy_mock = types.ModuleType("numpy")
    had_cv2 = "cv2" in sys.modules
    had_np = "numpy" in sys.modules
    # Não substituir numpy real; só cv2
    if not had_cv2:
        sys.modules["cv2"] = cv2_mock
    # Reimportar o módulo se necessário
    if "agents.agent_1_segmentador" in sys.modules:
        mod = sys.modules["agents.agent_1_segmentador"]
    else:
        import agents.agent_1_segmentador as mod
    return getattr(mod, name)


# ============================================================================
# Testes: _encontrar_contorno_cv_em_bbox()
# ============================================================================

class TestEncontrarContornoCvEmBbox:

    def setup_method(self):
        import types, sys
        # Mock cv2 para permitir import do módulo agent_1
        if "cv2" not in sys.modules:
            sys.modules["cv2"] = types.ModuleType("cv2")
        # Importa a função diretamente
        from agents.agent_1_segmentador import _encontrar_contorno_cv_em_bbox
        self.fn = _encontrar_contorno_cv_em_bbox

    def _contornos(self):
        return [
            {"bbox": (50, 100, 200, 400), "area_pct": 0.15, "centro": (150, 300)},   # esquerda
            {"bbox": (500, 80, 150, 380), "area_pct": 0.10, "centro": (575, 270)},   # direita
        ]

    def test_match_esquerda(self):
        """Bbox cobrindo metade esquerda → encontra contorno 0 (centro em 150,300)."""
        idx, match = self.fn(
            {"x1": 0.0, "y1": 0.0, "x2": 0.5, "y2": 1.0},
            self._contornos(), largura=1000, altura=800, indices_usados=set()
        )
        assert idx == 0
        assert match["centro"] == (150, 300)

    def test_match_direita(self):
        """Bbox cobrindo metade direita → encontra contorno 1 (centro em 575,270)."""
        idx, match = self.fn(
            {"x1": 0.5, "y1": 0.0, "x2": 1.0, "y2": 1.0},
            self._contornos(), largura=1000, altura=800, indices_usados=set()
        )
        assert idx == 1
        assert match["centro"] == (575, 270)

    def test_indices_usados_excluidos(self):
        """Contorno já usado deve ser ignorado; resultado recai no outro."""
        idx, match = self.fn(
            {"x1": 0.0, "y1": 0.0, "x2": 0.5, "y2": 1.0},
            self._contornos(), largura=1000, altura=800, indices_usados={0}
        )
        # Contorno 0 excluído; o único centro em x=[0,500] é... nenhum na direita
        assert idx is None
        assert match is None

    def test_sem_match_bbox_vazia(self):
        """Bbox em canto sem objetos → retorna None, None."""
        idx, match = self.fn(
            {"x1": 0.9, "y1": 0.9, "x2": 1.0, "y2": 1.0},
            self._contornos(), largura=1000, altura=800, indices_usados=set()
        )
        assert idx is None
        assert match is None

    def test_dois_candidatos_preferencia_maior_area(self):
        """Dois contornos dentro do bbox → retorna o de maior área."""
        idx, match = self.fn(
            {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0},  # bbox toda a foto
            self._contornos(), largura=1000, altura=800, indices_usados=set()
        )
        assert idx == 0  # area_pct 0.15 > 0.10

    def test_lista_vazia(self):
        """Sem contornos → retorna None, None."""
        idx, match = self.fn(
            {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0},
            [], largura=1000, altura=800, indices_usados=set()
        )
        assert idx is None
        assert match is None

    def test_todos_usados(self):
        """Todos os índices já usados → retorna None, None."""
        idx, match = self.fn(
            {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0},
            self._contornos(), largura=1000, altura=800, indices_usados={0, 1}
        )
        assert idx is None
        assert match is None


# ============================================================================
# Testes: _bbox_normalizada_para_pixels()
# ============================================================================

class TestBboxNormalizadaParaPixels:

    def setup_method(self):
        import types, sys
        if "cv2" not in sys.modules:
            sys.modules["cv2"] = types.ModuleType("cv2")
        from agents.agent_1_segmentador import _bbox_normalizada_para_pixels
        self.fn = _bbox_normalizada_para_pixels

    def test_conversao_basica(self):
        """Bbox centrada na imagem → coordenadas corretas."""
        x, y, w, h = self.fn(
            {"x1": 0.25, "y1": 0.25, "x2": 0.75, "y2": 0.75},
            largura=1000, altura=800, margem_pct=0.0
        )
        assert x == 250
        assert y == 200
        assert w == 500
        assert h == 400

    def test_margem_aplicada(self):
        """Margem de 2% expande o bbox."""
        x0, y0, w0, h0 = self.fn(
            {"x1": 0.1, "y1": 0.1, "x2": 0.9, "y2": 0.9},
            largura=1000, altura=1000, margem_pct=0.0
        )
        x1, y1, w1, h1 = self.fn(
            {"x1": 0.1, "y1": 0.1, "x2": 0.9, "y2": 0.9},
            largura=1000, altura=1000, margem_pct=0.02
        )
        assert x1 < x0   # margem expande à esquerda
        assert y1 < y0   # margem expande acima
        assert w1 > w0   # largura maior
        assert h1 > h0   # altura maior

    def test_margem_clampada_bordas(self):
        """Bbox nas bordas com margem não vai além de [0, dimensão]."""
        x, y, w, h = self.fn(
            {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0},
            largura=500, altura=400, margem_pct=0.05
        )
        # x e y devem ser >= 0
        assert x >= 0
        assert y >= 0
        # w e h não devem ultrapassar a imagem
        assert x + w <= 500
        assert y + h <= 400


# ============================================================================
# Testes: _nome_eh_aceitavel()
# ============================================================================

class TestNomeEhAceitavel:

    def setup_method(self):
        import types, sys
        if "cv2" not in sys.modules:
            sys.modules["cv2"] = types.ModuleType("cv2")
        from agents.agent_1_segmentador import _nome_eh_aceitavel
        self.fn = _nome_eh_aceitavel

    def test_nomes_validos(self):
        assert self.fn("chave de fenda") is True
        assert self.fn("régua") is True
        assert self.fn("tesoura") is True
        assert self.fn("martelo") is True

    def test_nomes_bloqueados(self):
        assert self.fn("mesa") is False
        assert self.fn("tampo da mesa") is False
        assert self.fn("chão") is False
        assert self.fn("superfície") is False
        assert self.fn("reflexo da janela") is False

    def test_nome_vazio_ou_none(self):
        assert self.fn("") is False
        assert self.fn(None) is False
        assert self.fn("   ") is False
