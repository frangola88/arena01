"""Testes das verificações cruzadas clássico × Claude (core/verificacoes_cruzadas)."""
import numpy as np
import pytest

from core.analise_cena import AnaliseCena
from core.verificacoes_cruzadas import (
    ratio_contagem,
    centroide_em_bg,
    centroide_em_obj,
    forca_no_centroide,
    w_borda_bbox,
    concordancia_mapas,
    score_qualidade,
    verificar_cena,
    RATIO_PERDA,
    RATIO_FRAG,
)


def _cena(bg_mask: np.ndarray, n_estimado: int = 4,
          superficie: np.ndarray | None = None) -> AnaliseCena:
    """Monta uma AnaliseCena mínima com bg_mask (e opcionalmente superficie)."""
    Hb, Wb = bg_mask.shape
    if superficie is None:
        superficie = np.where(bg_mask, 0.1, 0.8).astype(np.float32)
    return AnaliseCena(
        n_objetos_estimado=n_estimado,
        complexidade="media",
        picos=[],
        bordas_ativas={"top": False, "bottom": False, "left": False, "right": False},
        superficie=superficie,
        obj_mask=~bg_mask,
        bg_mask=bg_mask,
        shape_blocos=(Hb, Wb),
        shape_original=(Hb * 16, Wb * 16),
        tempo_s=0.0,
    )


# ─── ratio_contagem ──────────────────────────────────────────────────────────

def test_ratio_sem_sinal_quando_estimado_zero():
    r = ratio_contagem(n_claude=5, n_estimado=0)
    assert r["veredito"] == "sem_sinal"
    assert r["ratio"] is None


def test_ratio_ok_quando_proximo():
    r = ratio_contagem(n_claude=10, n_estimado=10)
    assert r["veredito"] == "ok"
    assert r["ratio"] == 1.0


def test_ratio_perda_quando_claude_conta_muito_menos():
    # 2 / 10 = 0.2 < RATIO_PERDA
    r = ratio_contagem(n_claude=2, n_estimado=10)
    assert r["veredito"] == "claude_perdeu_objetos"
    assert r["ratio"] < RATIO_PERDA


def test_ratio_fragmentando_quando_claude_conta_muito_mais():
    # 25 / 10 = 2.5 > RATIO_FRAG
    r = ratio_contagem(n_claude=25, n_estimado=10)
    assert r["veredito"] == "claude_fragmentando"
    assert r["ratio"] > RATIO_FRAG


def test_ratio_limites_inclusivos():
    # exatamente no limiar não dispara (usa estritamente < e >)
    n_est = 10
    r_perda = ratio_contagem(int(RATIO_PERDA * n_est), n_est)   # 5/10 = 0.5
    r_frag = ratio_contagem(int(RATIO_FRAG * n_est), n_est)     # 20/10 = 2.0
    assert r_perda["veredito"] == "ok"
    assert r_frag["veredito"] == "ok"


# ─── centroide_em_bg ─────────────────────────────────────────────────────────

def test_centroide_em_regiao_de_fundo_retorna_true():
    bg = np.zeros((4, 4), dtype=bool)
    bg[0, 0] = True                       # canto superior-esquerdo é fundo
    cena = _cena(bg)
    assert centroide_em_bg({"cx": 0.0, "cy": 0.0}, cena) is True


def test_centroide_em_regiao_de_objeto_retorna_false():
    bg = np.zeros((4, 4), dtype=bool)
    bg[0, 0] = True
    cena = _cena(bg)
    # centro da imagem → bloco (1 ou 2, 1 ou 2), que NÃO é fundo
    assert centroide_em_bg({"cx": 0.5, "cy": 0.5}, cena) is False


def test_centroide_none_retorna_false():
    cena = _cena(np.ones((4, 4), dtype=bool))
    assert centroide_em_bg(None, cena) is False


def test_centroide_mascara_degradada_retorna_false():
    # analise_cena falhou → bg_mask 1×1 → sem sinal, mesmo tudo "fundo"
    cena = _cena(np.ones((1, 1), dtype=bool), n_estimado=0)
    assert centroide_em_bg({"cx": 0.5, "cy": 0.5}, cena) is False


def test_centroide_fora_de_faixa_e_clampeado():
    bg = np.zeros((4, 4), dtype=bool)
    bg[3, 3] = True                       # canto inferior-direito é fundo
    cena = _cena(bg)
    # cx/cy > 1 devem ser clampeados para o último bloco (fundo)
    assert centroide_em_bg({"cx": 1.5, "cy": 1.5}, cena) is True


# ─── verificar_cena (integração das duas checagens) ──────────────────────────

def test_verificar_cena_reporta_contagem_e_suspeitos():
    bg = np.zeros((4, 4), dtype=bool)
    bg[0, 0] = True
    cena = _cena(bg, n_estimado=4)
    objetos = [
        {"nome": "chave", "centroide_normalizado": {"cx": 0.0, "cy": 0.0}},   # cai em bg
        {"nome": "alicate", "centroide_normalizado": {"cx": 0.5, "cy": 0.5}}, # objeto
        {"nome": "sem_centroide"},                                            # sem dado
    ]
    rel = verificar_cena(cena, objetos)
    assert rel["contagem"]["n_claude"] == 3
    assert rel["contagem"]["n_estimado"] == 4
    assert rel["centroide_em_bg"] == ["chave"]
    assert rel["n_suspeitos_bg"] == 1


def test_verificar_cena_lista_vazia():
    cena = _cena(np.zeros((4, 4), dtype=bool), n_estimado=3)
    rel = verificar_cena(cena, [])
    assert rel["contagem"]["n_claude"] == 0
    assert rel["contagem"]["veredito"] == "claude_perdeu_objetos"
    assert rel["n_suspeitos_bg"] == 0


# ─── centroide_em_obj ────────────────────────────────────────────────────────

def test_centroide_em_obj_true_no_objeto():
    bg = np.zeros((4, 4), dtype=bool)
    bg[0, 0] = True
    cena = _cena(bg)
    assert centroide_em_obj({"cx": 0.5, "cy": 0.5}, cena) is True


def test_centroide_em_obj_false_no_fundo():
    bg = np.zeros((4, 4), dtype=bool)
    bg[0, 0] = True
    cena = _cena(bg)
    assert centroide_em_obj({"cx": 0.0, "cy": 0.0}, cena) is False


def test_centroide_em_obj_none_e_degradado():
    cena = _cena(np.ones((1, 1), dtype=bool))
    assert centroide_em_obj(None, cena) is False
    assert centroide_em_obj({"cx": 0.5, "cy": 0.5}, cena) is False


# ─── forca_no_centroide ──────────────────────────────────────────────────────

def test_forca_alta_em_objeto_baixa_em_fundo():
    bg = np.zeros((4, 4), dtype=bool)
    bg[0, 0] = True
    cena = _cena(bg)   # superficie: 0.8 em objeto, 0.1 em fundo
    assert forca_no_centroide({"cx": 0.5, "cy": 0.5}, cena) > 0.5
    assert forca_no_centroide({"cx": 0.0, "cy": 0.0}, cena) < 0.5


def test_forca_zero_sem_centroide_ou_degradado():
    assert forca_no_centroide(None, _cena(np.zeros((4, 4), dtype=bool))) == 0.0
    assert forca_no_centroide({"cx": 0.5, "cy": 0.5},
                              _cena(np.ones((1, 1), dtype=bool))) == 0.0


# ─── w_borda_bbox ────────────────────────────────────────────────────────────

def test_w_borda_detecta_objeto_continuando_alem_da_bbox():
    # superficie toda "objeto" (0.9): qualquer bbox interna terá overflow alto
    sup = np.full((10, 10), 0.9, dtype=np.float32)
    cena = _cena(np.zeros((10, 10), dtype=bool), superficie=sup)
    wb = w_borda_bbox({"x1": 0.3, "y1": 0.3, "x2": 0.6, "y2": 0.6}, cena)
    assert wb["max"] > 0.5
    assert len(wb["cortado"]) > 0


def test_w_borda_sem_overflow_objeto_isolado():
    # objeto só no centro; fora da bbox é fundo → sem corte
    sup = np.full((10, 10), 0.05, dtype=np.float32)
    sup[4:6, 4:6] = 0.9
    cena = _cena(np.zeros((10, 10), dtype=bool), superficie=sup)
    wb = w_borda_bbox({"x1": 0.2, "y1": 0.2, "x2": 0.8, "y2": 0.8}, cena)
    assert wb["cortado"] == []


def test_w_borda_degradado_ou_sem_bbox():
    cena = _cena(np.ones((1, 1), dtype=bool))
    assert w_borda_bbox(None, cena)["cortado"] == []
    assert w_borda_bbox({"x1": 0, "y1": 0, "x2": 1, "y2": 1}, cena)["max"] == 0.0


# ─── concordancia_mapas ──────────────────────────────────────────────────────

def test_concordancia_alta_para_mapas_iguais():
    m = np.random.RandomState(0).rand(8, 8)
    assert concordancia_mapas(m, m.copy()) > 0.95


def test_concordancia_baixa_para_mapas_invertidos():
    m = np.linspace(0, 1, 64).reshape(8, 8)
    assert concordancia_mapas(m, 1.0 - m) < 0.1


def test_concordancia_neutra_para_degenerado():
    assert concordancia_mapas(np.zeros((8, 8)), np.ones((8, 8))) == 0.5
    assert concordancia_mapas(np.array([[1.0]]), np.array([[2.0]])) == 0.5


# ─── score_qualidade (combinação) ────────────────────────────────────────────

def test_score_alto_objeto_bem_posicionado_confiante():
    bg = np.zeros((10, 10), dtype=bool)
    sup = np.full((10, 10), 0.05, dtype=np.float32)
    sup[4:6, 4:6] = 0.9
    cena = _cena(bg, superficie=sup)
    obj = {
        "nome": "chave",
        "centroide_normalizado": {"cx": 0.5, "cy": 0.5},
        "bbox_normalizada": {"x1": 0.2, "y1": 0.2, "x2": 0.8, "y2": 0.8},
        "confianca": 0.95,
    }
    res = score_qualidade(obj, cena)
    assert res["score"] > 0.7
    assert res["flags"] == []
    assert "mapa" not in res["componentes"]      # sem heatmap → sem componente mapa


def test_score_baixo_centroide_em_fundo_e_pouca_confianca():
    bg = np.zeros((10, 10), dtype=bool)
    bg[0, 0] = True
    sup = np.where(bg, 0.05, 0.5).astype(np.float32)
    cena = _cena(bg, superficie=sup)
    obj = {
        "nome": "fantasma",
        "centroide_normalizado": {"cx": 0.0, "cy": 0.0},   # canto = fundo
        "bbox_normalizada": {"x1": 0.0, "y1": 0.0, "x2": 0.1, "y2": 0.1},
        "confianca": 0.3,
    }
    res = score_qualidade(obj, cena)
    assert res["score"] < 0.5
    assert "centroide_em_bg" in res["flags"]
    assert "baixa_confianca_claude" in res["flags"]


def test_score_inclui_componente_mapa_quando_heatmap_dado():
    bg = np.zeros((8, 8), dtype=bool)
    sup = np.full((8, 8), 0.05, dtype=np.float32)
    sup[3:5, 3:5] = 0.9
    cena = _cena(bg, superficie=sup)
    obj = {
        "nome": "chave",
        "centroide_normalizado": {"cx": 0.5, "cy": 0.5},
        "bbox_normalizada": {"x1": 0.2, "y1": 0.2, "x2": 0.8, "y2": 0.8},
        "confianca": 0.9,
    }
    heatmap = sup.copy()                          # heatmap concorda com a superficie
    res = score_qualidade(obj, cena, heatmap=heatmap)
    assert "mapa" in res["componentes"]
    assert res["componentes"]["mapa"] > 0.5
