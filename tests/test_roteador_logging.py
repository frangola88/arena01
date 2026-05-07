"""Testes de instrumentação: cada decisão de roteamento emite um log JSON
com modo, tem_api, decisao e motivo.

Não testa a matriz de decisão (já coberta em test_roteador.py) — só que
cada caminho gera o evento certo com os campos certos.
"""
from __future__ import annotations

import logging

import pytest


# ---------------------------------------------------------------------------
# deve_usar_claude_visao — eventos por motivo
# ---------------------------------------------------------------------------

@pytest.fixture
def caplog_info(caplog):
    caplog.set_level(logging.INFO, logger="casaiq.roteador")
    return caplog


def _ultimo(caplog) -> logging.LogRecord:
    """Devolve o último record emitido pelo roteador (ignora outros loggers)."""
    do_roteador = [r for r in caplog.records if r.name == "casaiq.roteador"]
    assert do_roteador, "Nenhum record do roteador capturado"
    return do_roteador[-1]


@pytest.mark.parametrize("modo,confianca,esperado_decisao,esperado_motivo", [
    ("offline",        None, False, "modo_offline"),
    ("offline",        0.30, False, "modo_offline"),
    ("claude",         None, True,  "modo_claude"),
    ("claude",         0.95, True,  "modo_claude"),
    ("hibrido",        None, False, "modo_hibrido_visao_local"),
    ("inteligente",    None, False, "primeira_tentativa_local"),
    ("inteligente",    0.30, True,  "confianca_abaixo_limiar"),
    ("inteligente",    0.95, False, "confianca_acima_limiar"),
    ("local_primeiro", 0.30, True,  "confianca_abaixo_limiar"),
    ("local_primeiro", 0.95, False, "confianca_acima_limiar"),
])
def test_visao_emite_evento_com_motivo_correto(
    caplog_info, set_modo, com_api, modo, confianca, esperado_decisao, esperado_motivo
):
    set_modo(modo)
    from core.roteador import deve_usar_claude_visao
    decisao = deve_usar_claude_visao(confianca_anterior=confianca)
    rec = _ultimo(caplog_info)

    assert rec.message == "decisao_visao"
    assert rec.decisao is esperado_decisao
    assert rec.motivo == esperado_motivo
    assert rec.modo == modo
    assert rec.tem_api is True
    assert rec.confianca == confianca
    assert decisao is esperado_decisao


def test_visao_sem_api_emite_motivo_sem_api_key(caplog_info, set_modo, sem_api):
    set_modo("inteligente")
    from core.roteador import deve_usar_claude_visao
    deve_usar_claude_visao(confianca_anterior=0.30)
    rec = _ultimo(caplog_info)
    assert rec.motivo == "sem_api_key"
    assert rec.decisao is False
    assert rec.tem_api is False


def test_visao_sempre_inclui_limiar_no_log(caplog_info, set_modo, com_api, set_limiar):
    set_modo("inteligente")
    set_limiar(0.42)
    from core.roteador import deve_usar_claude_visao
    deve_usar_claude_visao(confianca_anterior=0.5)
    rec = _ultimo(caplog_info)
    assert rec.limiar == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# deve_usar_claude_texto — eventos por motivo
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("modo,tarefa,esperado_decisao,esperado_motivo", [
    # OFFLINE bloqueia tudo
    ("offline",        "text_to_sql",     False, "modo_offline"),
    ("offline",        "classificacao",   False, "modo_offline"),
    # SEMPRE_LOCAL > tudo (exceto sem_api / offline)
    ("claude",         "classificacao",   False, "tarefa_sempre_local"),
    ("claude",         "enriquecimento",  False, "tarefa_sempre_local"),
    # CLAUDE para tarefas livres
    ("claude",         "text_to_sql",     True,  "modo_claude"),
    ("claude",         "guia",            True,  "modo_claude"),
    # HIBRIDO/INTELIGENTE: só preferenciais vão pra Claude
    ("hibrido",        "text_to_sql",     True,  "tarefa_preferencial_claude"),
    ("inteligente",    "resposta_chat",   True,  "tarefa_preferencial_claude"),
    ("hibrido",        "text_to_sql",     True,  "tarefa_preferencial_claude"),
    # LOCAL_PRIMEIRO mantém local mesmo nas preferenciais
    ("local_primeiro", "text_to_sql",     False, "modo_local_primeiro"),
    ("local_primeiro", "guia",            False, "modo_local_primeiro"),
])
def test_texto_emite_evento_com_motivo_correto(
    caplog_info, set_modo, com_api, modo, tarefa, esperado_decisao, esperado_motivo
):
    set_modo(modo)
    from core.roteador import deve_usar_claude_texto, TarefaTexto
    decisao = deve_usar_claude_texto(TarefaTexto(tarefa))
    rec = _ultimo(caplog_info)

    assert rec.message == "decisao_texto"
    assert rec.decisao is esperado_decisao
    assert rec.motivo == esperado_motivo
    assert rec.modo == modo
    assert rec.tarefa == tarefa
    assert decisao is esperado_decisao


def test_texto_sem_api_emite_motivo_sem_api_key(caplog_info, set_modo, sem_api):
    set_modo("claude")
    from core.roteador import deve_usar_claude_texto, TarefaTexto
    deve_usar_claude_texto(TarefaTexto.TEXT_TO_SQL)
    rec = _ultimo(caplog_info)
    assert rec.motivo == "sem_api_key"
    assert rec.tem_api is False


# ---------------------------------------------------------------------------
# Modo inválido emite warning estruturado
# ---------------------------------------------------------------------------

def test_modo_invalido_emite_warning_estruturado(caplog, monkeypatch):
    caplog.set_level(logging.WARNING, logger="casaiq.roteador")
    monkeypatch.setattr("core.roteador.CASAIQ_MODO", "modo_que_nao_existe")
    from core.roteador import _modo
    _modo()
    warn = [r for r in caplog.records if r.levelname == "WARNING"
            and r.message == "modo_invalido"]
    assert warn, "warning de modo inválido não foi emitido"
    assert warn[-1].valor_recebido == "modo_que_nao_existe"
    assert warn[-1].fallback == "inteligente"
