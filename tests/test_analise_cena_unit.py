"""
Testes unitários para core.analise_cena — pré-análise de cena por seções ortogonais.

A análise depende de scipy (interpolação/morfologia). Onde scipy falta, o módulo
degrada graciosamente (n=0, complexidade='simples') — também coberto aqui.
"""
import numpy as np
import pytest

from core.analise_cena import (
    analisar_cena, AnaliseCena,
    _variancia_local, _normalizar, _amostrar_secoes, _sigmoid,
)


# ─── helpers puros (sem scipy) ───────────────────────────────────────────────────

def test_normalizar_intervalo_0_1():
    arr = np.array([[2.0, 4.0], [6.0, 10.0]], dtype=np.float32)
    out = _normalizar(arr)
    assert out.min() == pytest.approx(0.0)
    assert out.max() == pytest.approx(1.0)


def test_normalizar_constante_nao_estoura():
    arr = np.full((4, 4), 5.0, dtype=np.float32)
    out = _normalizar(arr)
    assert np.all(np.isfinite(out))          # não divide por zero
    assert out.max() <= 1.0


def test_variancia_local_zero_em_imagem_uniforme():
    img = np.full((64, 64), 128, dtype=np.float32)
    vmap = _variancia_local(img, 16)
    assert vmap.shape == (4, 4)
    assert np.allclose(vmap, 0.0)


def test_variancia_local_detecta_textura():
    img = np.zeros((64, 64), dtype=np.float32)
    img[:32, :32] = np.random.RandomState(0).randint(0, 255, (32, 32))  # quadrante texturado
    vmap = _variancia_local(img, 16)
    # canto superior-esquerdo tem variância; canto inferior-direito não
    assert vmap[0, 0] > vmap[3, 3]


def test_amostrar_secoes_conta_pontos():
    mapa = np.zeros((20, 30), dtype=np.float32)
    pts, vals = _amostrar_secoes(mapa, n_h=5, m_v=5, passo=2)
    assert len(pts) == len(vals)
    assert pts.shape[1] == 2
    # coords normalizadas em [0,1]
    assert pts.min() >= 0.0 and pts.max() <= 1.0


def test_sigmoid_exacerba_contraste():
    pytest.importorskip("scipy")
    sup = np.linspace(0, 1, 100).reshape(10, 10).astype(np.float32)
    out = _sigmoid(sup, k=12.0, mu=0.45)
    # sigmoid empurra para os extremos: massa nas pontas maior que no meio
    centro = np.mean((out > 0.4) & (out < 0.6))
    extremos = np.mean((out < 0.1) | (out > 0.9))
    assert extremos > centro


# ─── pipeline completo (requer scipy) ────────────────────────────────────────────

@pytest.fixture
def foto_sintetica():
    """Imagem RGB com 3 blocos texturados sobre fundo liso."""
    rng = np.random.RandomState(42)
    img = np.full((300, 400, 3), 200, dtype=np.uint8)   # fundo cinza liso
    for (y, x) in [(40, 40), (40, 300), (200, 180)]:
        img[y:y+50, x:x+50] = rng.randint(0, 255, (50, 50, 3))  # objeto texturado
    return img


def test_analisar_cena_retorna_dataclass(foto_sintetica):
    pytest.importorskip("scipy")
    r = analisar_cena(foto_sintetica)
    assert isinstance(r, AnaliseCena)
    assert r.shape_blocos[0] > 0 and r.shape_blocos[1] > 0
    assert r.complexidade in {"simples", "media", "densa"}
    assert r.tempo_s >= 0.0


def test_analisar_cena_detecta_objetos(foto_sintetica):
    pytest.importorskip("scipy")
    r = analisar_cena(foto_sintetica)
    # 3 blocos sintéticos → espera-se detectar alguns objetos (>=1)
    assert r.n_objetos_estimado >= 1
    assert len(r.picos) == r.n_objetos_estimado
    for p in r.picos:
        assert 0.0 <= p["cx"] <= 1.0
        assert 0.0 <= p["cy"] <= 1.0
        assert p["area_blocos"] >= 1


def test_analisar_cena_bg_mask_existe(foto_sintetica):
    pytest.importorskip("scipy")
    r = analisar_cena(foto_sintetica)
    # fundo liso domina → bg_mask deve marcar bastante área
    assert r.bg_mask.dtype == bool
    assert r.bg_mask.any()


def test_resumo_prompt_formato(foto_sintetica):
    pytest.importorskip("scipy")
    r = analisar_cena(foto_sintetica)
    txt = r.resumo_prompt()
    assert "objetos" in txt
    assert r.complexidade in txt


def test_analisar_cena_fonte_invalida_degrada():
    # caminho inexistente → não levanta, devolve análise vazia
    r = analisar_cena("/tmp/__nao_existe_casaiq__.jpg")
    assert isinstance(r, AnaliseCena)
    assert r.n_objetos_estimado == 0
    assert r.complexidade == "simples"
    assert r.picos == []


def test_analisar_cena_aceita_cinza(foto_sintetica):
    pytest.importorskip("scipy")
    cinza = foto_sintetica.mean(axis=2).astype(np.uint8)
    r = analisar_cena(cinza)
    assert isinstance(r, AnaliseCena)
    assert r.shape_blocos[0] > 0
