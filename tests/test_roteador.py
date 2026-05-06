"""Testes do roteador inteligente.

O roteador é lógica pura — sem I/O, sem rede. Toda a matriz de decisão
é coberta aqui via parametrização. Quando mudar o LIMIAR_CONFIANCA padrão
ou as regras de roteamento, este arquivo precisa refletir antes do código.
"""
from __future__ import annotations

import pytest

from core.roteador import (
    Modo,
    TarefaTexto,
    descricao_modo,
    deve_usar_claude_texto,
    deve_usar_claude_visao,
)


# ---------------------------------------------------------------------------
# deve_usar_claude_visao — matriz: 5 modos × {com, sem} api × 3 confianças
# ---------------------------------------------------------------------------

# Tabela canônica de expectativas para visão.
# Cada tupla: (modo, tem_api, confianca, esperado)
VISAO_MATRIZ = [
    # OFFLINE — nunca usa Claude, mesmo com API ou confiança baixa
    ("offline",        True,  None, False),
    ("offline",        True,  0.30, False),
    ("offline",        True,  0.95, False),
    ("offline",        False, None, False),

    # CLAUDE — sempre vai pra Claude se houver API; sem API, degrada pra local
    ("claude",         True,  None, True),
    ("claude",         True,  0.30, True),
    ("claude",         True,  0.95, True),
    ("claude",         False, None, False),
    ("claude",         False, 0.30, False),

    # HIBRIDO — visão fica SEMPRE local (Claude só em texto)
    ("hibrido",        True,  None, False),
    ("hibrido",        True,  0.30, False),
    ("hibrido",        True,  0.95, False),
    ("hibrido",        False, None, False),

    # LOCAL_PRIMEIRO — Claude só se confianca local < limiar
    ("local_primeiro", True,  None, False),  # primeira tentativa: local
    ("local_primeiro", True,  0.30, True),   # baixa confiança: segunda opinião
    ("local_primeiro", True,  0.64, True),   # logo abaixo do limiar default 0.65
    ("local_primeiro", True,  0.65, False),  # exatamente no limiar: NÃO usa
    ("local_primeiro", True,  0.95, False),
    ("local_primeiro", False, 0.30, False),  # sem API: nunca

    # INTELIGENTE — mesma regra de visão que LOCAL_PRIMEIRO
    ("inteligente",    True,  None, False),
    ("inteligente",    True,  0.30, True),
    ("inteligente",    True,  0.64, True),
    ("inteligente",    True,  0.65, False),
    ("inteligente",    True,  0.95, False),
    ("inteligente",    False, 0.30, False),
]


@pytest.mark.parametrize("modo,tem_api,confianca,esperado", VISAO_MATRIZ)
def test_deve_usar_claude_visao(set_modo, set_api, modo, tem_api, confianca, esperado):
    set_modo(modo)
    set_api("sk-test" if tem_api else "")
    assert deve_usar_claude_visao(confianca) is esperado


def test_visao_respeita_limiar_customizado(set_modo, com_api, set_limiar):
    """Mudar LIMIAR_CONFIANCA muda a fronteira de decisão."""
    set_modo("inteligente")
    set_limiar(0.80)

    assert deve_usar_claude_visao(0.79) is True   # agora 0.79 está abaixo
    assert deve_usar_claude_visao(0.80) is False  # exato no novo limiar
    assert deve_usar_claude_visao(0.65) is True   # antes do default era limite


# ---------------------------------------------------------------------------
# deve_usar_claude_texto — matriz: 5 modos × {com, sem} api × 5 tarefas
# ---------------------------------------------------------------------------

# Tarefas que SEMPRE ficam locais (independente de modo/api)
TAREFAS_SEMPRE_LOCAL = [TarefaTexto.CLASSIFICACAO, TarefaTexto.ENRIQUECIMENTO]

# Tarefas que preferem Claude quando o modo permite
TAREFAS_PREF_CLAUDE = [TarefaTexto.TEXT_TO_SQL, TarefaTexto.GUIA, TarefaTexto.RESPOSTA_CHAT]

TODAS_TAREFAS = TAREFAS_SEMPRE_LOCAL + TAREFAS_PREF_CLAUDE


@pytest.mark.parametrize("tarefa", TAREFAS_SEMPRE_LOCAL)
@pytest.mark.parametrize("modo", ["offline", "local_primeiro", "hibrido", "claude", "inteligente"])
def test_texto_classificacao_e_enriquecimento_sao_sempre_locais(set_modo, com_api, modo, tarefa):
    """Tarefas simples nunca devem ir pra Claude — alta frequência, sem ganho."""
    set_modo(modo)
    assert deve_usar_claude_texto(tarefa) is False


@pytest.mark.parametrize("tarefa", TODAS_TAREFAS)
def test_texto_offline_nunca_usa_claude(set_modo, com_api, tarefa):
    set_modo("offline")
    assert deve_usar_claude_texto(tarefa) is False


@pytest.mark.parametrize("tarefa", TODAS_TAREFAS)
def test_texto_sem_api_nunca_usa_claude(set_modo, sem_api, tarefa):
    set_modo("inteligente")
    assert deve_usar_claude_texto(tarefa) is False


@pytest.mark.parametrize("tarefa", TAREFAS_PREF_CLAUDE)
def test_texto_modo_claude_usa_claude_em_tarefas_preferenciais(set_modo, com_api, tarefa):
    set_modo("claude")
    assert deve_usar_claude_texto(tarefa) is True


@pytest.mark.parametrize("modo", ["hibrido", "inteligente"])
@pytest.mark.parametrize("tarefa", TAREFAS_PREF_CLAUDE)
def test_texto_hibrido_e_inteligente_preferem_claude_em_sql_guia_chat(
    set_modo, com_api, modo, tarefa
):
    set_modo(modo)
    assert deve_usar_claude_texto(tarefa) is True


@pytest.mark.parametrize("tarefa", TAREFAS_PREF_CLAUDE)
def test_texto_local_primeiro_nao_usa_claude_proativamente(set_modo, com_api, tarefa):
    """LOCAL_PRIMEIRO só sobe pra Claude em caso de falha — decidido em llm.py, não no roteador."""
    set_modo("local_primeiro")
    assert deve_usar_claude_texto(tarefa) is False


# ---------------------------------------------------------------------------
# descricao_modo — não pode quebrar para nenhum modo válido
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("modo", ["offline", "local_primeiro", "hibrido", "claude", "inteligente"])
def test_descricao_modo_nao_vazia_em_todos_os_modos(set_modo, com_api, modo):
    set_modo(modo)
    desc = descricao_modo()
    assert isinstance(desc, str)
    assert len(desc) > 0


def test_descricao_modo_diferente_para_cada_modo(set_modo, com_api):
    """Cada modo deve produzir descrição distinta — protege contra colapso acidental."""
    descricoes = set()
    for modo in ["offline", "local_primeiro", "hibrido", "claude", "inteligente"]:
        set_modo(modo)
        descricoes.add(descricao_modo())
    assert len(descricoes) == 5


def test_descricao_modo_indica_falta_de_api(set_modo, sem_api):
    """Quando o modo é claude mas falta API, descrição deve sinalizar degradação."""
    set_modo("claude")
    desc = descricao_modo().lower()
    assert "api" in desc or "local" in desc or "anthropic" in desc


# ---------------------------------------------------------------------------
# Modo inválido cai em INTELIGENTE silenciosamente
# ---------------------------------------------------------------------------

def test_modo_invalido_degrada_para_inteligente(set_modo, com_api):
    set_modo("modo_que_nao_existe")
    # Comportamento de visão deve ser o mesmo que INTELIGENTE: sem confianca → False
    assert deve_usar_claude_visao(None) is False
    # E confianca baixa → True (regra do INTELIGENTE)
    assert deve_usar_claude_visao(0.30) is True


def test_modo_invalido_descricao_nao_quebra(set_modo, com_api):
    set_modo("xpto")
    # Não deve levantar — só retorna a string do modo resolvido
    desc = descricao_modo()
    assert isinstance(desc, str) and desc


# ---------------------------------------------------------------------------
# Sanity: enums expõem todos os valores esperados (proteção contra renomeação)
# ---------------------------------------------------------------------------

def test_enum_modo_cobre_todos_os_valores_documentados():
    assert {m.value for m in Modo} == {
        "offline", "local_primeiro", "hibrido", "claude", "inteligente",
    }


def test_enum_tarefa_texto_cobre_todos_os_valores_documentados():
    assert {t.value for t in TarefaTexto} == {
        "classificacao", "enriquecimento", "text_to_sql", "guia", "resposta_chat",
    }
