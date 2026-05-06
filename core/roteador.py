"""
Roteador Inteligente do CasaIQ

Decide qual backend usar (Ollama local vs Claude API) baseado em:
  1. CASAIQ_MODO configurado no .env
  2. Tipo da tarefa (visão, classificação, SQL, guia, chat)
  3. Confiança retornada pela tentativa anterior

MODOS:
  offline        -> 100% local, Claude nunca chamado (mesmo com API key)
  local_primeiro -> local padrão; Claude só se local falhar (timeout/erro)
  hibrido        -> local para VISÃO; Claude para TEXTO/CHAT/SQL
  claude         -> Claude para tudo (máxima qualidade, tem custo por token)
  inteligente    -> roteador decide por tarefa + confiança (PADRÃO RECOMENDADO)

MODO INTELIGENTE — regras:
  Visão:
    -> local se confiança esperada >= LIMIAR_CONFIANCA (objeto comum, foto boa)
    -> Claude se confiança retornada < LIMIAR_CONFIANCA (segunda opinião de qualidade)
  Texto/SQL:
    -> local para classificação simples e enriquecimento de metadados
    -> Claude para: text-to-SQL, guias "como fazer", respostas de chat complexas
"""
from enum import Enum
from typing import Optional
from core.config import CASAIQ_MODO, ANTHROPIC_API_KEY, LIMIAR_CONFIANCA


class Modo(str, Enum):
    OFFLINE         = "offline"
    LOCAL_PRIMEIRO  = "local_primeiro"
    HIBRIDO         = "hibrido"
    CLAUDE          = "claude"
    INTELIGENTE     = "inteligente"


class TarefaTexto(str, Enum):
    CLASSIFICACAO   = "classificacao"    # mapear categoria — sempre local
    ENRIQUECIMENTO  = "enriquecimento"   # sinônimos, normalização — sempre local
    TEXT_TO_SQL     = "text_to_sql"      # converter pergunta em SQL — Claude preferido
    GUIA            = "guia"             # "como trocar chuveiro?" — Claude preferido
    RESPOSTA_CHAT   = "resposta_chat"    # formular resposta final — Claude preferido


# Tarefas de texto que se beneficiam significativamente do Claude
_TAREFAS_PREFERENCIALMENTE_CLAUDE = {
    TarefaTexto.TEXT_TO_SQL,
    TarefaTexto.GUIA,
    TarefaTexto.RESPOSTA_CHAT,
}

# Tarefas que devem ficar sempre locais (simples, alta frequência, sem ganho em usar API)
_TAREFAS_SEMPRE_LOCAL = {
    TarefaTexto.CLASSIFICACAO,
    TarefaTexto.ENRIQUECIMENTO,
}


def _tem_api() -> bool:
    return bool(ANTHROPIC_API_KEY)


def _modo() -> Modo:
    try:
        return Modo(CASAIQ_MODO)
    except ValueError:
        print(f"[Roteador] CASAIQ_MODO='{CASAIQ_MODO}' inválido. Usando 'inteligente'.")
        return Modo.INTELIGENTE


def deve_usar_claude_visao(confianca_anterior: Optional[float] = None) -> bool:
    """
    Decide se a chamada de VISÃO deve ir para Claude.
    confianca_anterior: resultado de uma tentativa local prévia (None = primeira tentativa).
    """
    modo = _modo()
    if not _tem_api():
        return False
    if modo == Modo.OFFLINE:
        return False
    if modo == Modo.CLAUDE:
        return True
    if modo == Modo.HIBRIDO:
        # No modo híbrido: visão fica sempre local; Claude só para texto
        return False
    # Modos LOCAL_PRIMEIRO e INTELIGENTE:
    # -> só vai para Claude se a tentativa local retornou baixa confiança
    if confianca_anterior is not None and confianca_anterior < LIMIAR_CONFIANCA:
        return True
    return False


def deve_usar_claude_texto(tarefa: TarefaTexto) -> bool:
    """
    Decide se a chamada de TEXTO deve ir para Claude.
    Não depende de confiança — depende do tipo da tarefa.
    """
    modo = _modo()
    if not _tem_api():
        return False
    if modo == Modo.OFFLINE:
        return False
    if tarefa in _TAREFAS_SEMPRE_LOCAL:
        return False
    if modo == Modo.CLAUDE:
        return True
    if modo in (Modo.HIBRIDO, Modo.INTELIGENTE):
        return tarefa in _TAREFAS_PREFERENCIALMENTE_CLAUDE
    # LOCAL_PRIMEIRO: usa Claude para texto preferencial só se local falhar
    # (a lógica de fallback em caso de falha está em llm.py)
    return False


def descricao_modo() -> str:
    """Retorna string legível do modo ativo (para logs e interface)."""
    modo = _modo()
    tem = _tem_api()
    descricoes = {
        Modo.OFFLINE:        "offline (100% local, Claude desabilitado)",
        Modo.LOCAL_PRIMEIRO: f"local_primeiro ({'Claude como fallback' if tem else 'sem Claude'})",
        Modo.HIBRIDO:        f"híbrido (visão=local, texto/SQL/chat={'Claude' if tem else 'local'})",
        Modo.CLAUDE:         f"claude ({'API ativa' if tem else 'sem ANTHROPIC_API_KEY — degradando para local'})",
        Modo.INTELIGENTE:    f"inteligente ({'Claude disponível' if tem else 'sem Claude — usando local para tudo'})",
    }
    return descricoes.get(modo, str(modo))
