"""Testes de core/llm.py — execução de chamadas LLM com fallback.

Mockamos:
- ollama.chat (importado top-level em core.llm)
- anthropic.Anthropic (importado lazy dentro de _claude_*)

A lógica de roteamento NÃO é mockada — usamos o roteador real configurado
via fixtures set_modo/set_api do conftest. Isso garante que mudanças no
roteador sejam refletidas aqui automaticamente.
"""
from __future__ import annotations

import time

import pytest

from core.llm import (
    OllamaIndisponivel,
    _ollama_visao,
    chamar_texto,
    chamar_visao,
    extrair_json,
)
from core.roteador import TarefaTexto


# ---------------------------------------------------------------------------
# Mocks compartilhados
# ---------------------------------------------------------------------------

def _ollama_resposta(texto: str):
    return {"message": {"content": texto}}


@pytest.fixture
def mock_ollama_ok(mocker):
    return mocker.patch("core.llm.ollama.chat",
                        return_value=_ollama_resposta("resposta ollama"))


@pytest.fixture
def mock_ollama_falha(mocker):
    return mocker.patch("core.llm.ollama.chat",
                        side_effect=RuntimeError("ollama down"))


def _build_claude_response(texto: str, mocker):
    msg = mocker.MagicMock()
    msg.content = [mocker.MagicMock(text=texto)]
    return msg


@pytest.fixture
def mock_claude_ok(mocker):
    """anthropic.Anthropic() retorna client cuja .messages.create devolve resposta válida."""
    client = mocker.MagicMock()
    client.messages.create.return_value = _build_claude_response("resposta claude", mocker)
    mocker.patch("anthropic.Anthropic", return_value=client)
    return client


@pytest.fixture
def mock_claude_falha(mocker):
    """anthropic.Anthropic() instancia, mas .messages.create levanta."""
    client = mocker.MagicMock()
    client.messages.create.side_effect = RuntimeError("claude api down")
    mocker.patch("anthropic.Anthropic", return_value=client)
    return client


@pytest.fixture
def imagem_fake(tmp_path):
    """JPEG mínimo válido em disco (para _claude_visao abrir e base64-encodar)."""
    p = tmp_path / "foto.jpg"
    p.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00")
    return str(p)


# ---------------------------------------------------------------------------
# chamar_visao — fluxos principais
# ---------------------------------------------------------------------------

def test_visao_inteligente_sem_confianca_vai_para_ollama(
    set_modo, com_api, mock_ollama_ok, mock_claude_ok, imagem_fake
):
    set_modo("inteligente")
    resp, modelo = chamar_visao("descreva", imagem_fake, confianca_anterior=None)
    assert resp == "resposta ollama"
    assert modelo == "ollama"
    mock_ollama_ok.assert_called_once()
    mock_claude_ok.messages.create.assert_not_called()


def test_visao_modo_claude_chama_anthropic_direto(
    set_modo, com_api, mock_ollama_ok, mock_claude_ok, imagem_fake
):
    set_modo("claude")
    resp, modelo = chamar_visao("descreva", imagem_fake)
    assert resp == "resposta claude"
    assert modelo == "claude_api"
    mock_claude_ok.messages.create.assert_called_once()
    mock_ollama_ok.assert_not_called()


def test_visao_claude_falha_cai_para_ollama(
    set_modo, com_api, mock_ollama_ok, mock_claude_falha, imagem_fake
):
    """Modo CLAUDE: se Claude falha, sistema tenta Ollama antes de desistir."""
    set_modo("claude")
    resp, modelo = chamar_visao("descreva", imagem_fake)
    assert resp == "resposta ollama"
    assert modelo == "ollama"


def test_visao_ollama_falha_fallback_para_claude(
    set_modo, com_api, mock_ollama_falha, mock_claude_ok, imagem_fake
):
    """INTELIGENTE primeira tentativa → roteador escolhe Ollama; se Ollama
    falhar e houver API + modo != offline, sobe pro Claude automaticamente."""
    set_modo("inteligente")
    resp, modelo = chamar_visao("descreva", imagem_fake)
    assert resp == "resposta claude"
    assert modelo == "claude_api"


def test_visao_offline_nao_faz_fallback_para_claude(
    set_modo, com_api, mock_ollama_falha, mock_claude_ok, imagem_fake
):
    set_modo("offline")
    with pytest.raises(RuntimeError, match="Visão indisponível"):
        chamar_visao("descreva", imagem_fake)
    mock_claude_ok.messages.create.assert_not_called()


def test_visao_sem_api_e_ollama_falho_levanta(
    set_modo, sem_api, mock_ollama_falha, imagem_fake
):
    set_modo("inteligente")
    with pytest.raises(RuntimeError, match="Visão indisponível"):
        chamar_visao("descreva", imagem_fake)


def test_visao_confianca_baixa_pede_segunda_opiniao_claude(
    set_modo, com_api, mock_ollama_ok, mock_claude_ok, imagem_fake
):
    """INTELIGENTE + confianca < limiar → vai direto pro Claude.
    Não tenta Ollama de novo (a tentativa local que gerou a confiança baixa
    já aconteceu fora desta chamada)."""
    set_modo("inteligente")
    _, modelo = chamar_visao("descreva", imagem_fake, confianca_anterior=0.3)
    assert modelo == "claude_api"
    mock_claude_ok.messages.create.assert_called_once()
    mock_ollama_ok.assert_not_called()


# ---------------------------------------------------------------------------
# chamar_texto
# ---------------------------------------------------------------------------

def test_texto_classificacao_sempre_local(
    set_modo, com_api, mock_ollama_ok, mock_claude_ok
):
    set_modo("inteligente")
    _, modelo = chamar_texto("classifique X", tarefa=TarefaTexto.CLASSIFICACAO)
    assert modelo == "ollama"
    mock_claude_ok.messages.create.assert_not_called()


def test_texto_text_to_sql_em_inteligente_vai_para_claude(
    set_modo, com_api, mock_ollama_ok, mock_claude_ok
):
    set_modo("inteligente")
    _, modelo = chamar_texto("converta", tarefa=TarefaTexto.TEXT_TO_SQL)
    assert modelo == "claude_api"
    mock_ollama_ok.assert_not_called()


def test_texto_local_primeiro_nao_chama_claude_proativamente(
    set_modo, com_api, mock_ollama_ok, mock_claude_ok
):
    """LOCAL_PRIMEIRO: roteador retorna False mesmo em tarefa preferencial.
    Só sobe pra Claude se Ollama falhar."""
    set_modo("local_primeiro")
    _, modelo = chamar_texto("converta", tarefa=TarefaTexto.TEXT_TO_SQL)
    assert modelo == "ollama"
    mock_claude_ok.messages.create.assert_not_called()


def test_texto_local_primeiro_ollama_falha_fallback_claude(
    set_modo, com_api, mock_ollama_falha, mock_claude_ok
):
    set_modo("local_primeiro")
    _, modelo = chamar_texto("converta", tarefa=TarefaTexto.TEXT_TO_SQL)
    assert modelo == "claude_api"


def test_texto_offline_ollama_falha_levanta(
    set_modo, com_api, mock_ollama_falha, mock_claude_ok
):
    set_modo("offline")
    with pytest.raises(RuntimeError, match="Texto indisponível"):
        chamar_texto("oi", tarefa=TarefaTexto.RESPOSTA_CHAT)
    mock_claude_ok.messages.create.assert_not_called()


def test_texto_passa_max_tokens_para_anthropic(
    set_modo, com_api, mock_ollama_ok, mock_claude_ok
):
    """max_tokens deve ser repassado para anthropic.messages.create."""
    set_modo("claude")
    chamar_texto("oi", tarefa=TarefaTexto.RESPOSTA_CHAT, max_tokens=2048)
    kwargs = mock_claude_ok.messages.create.call_args.kwargs
    assert kwargs["max_tokens"] == 2048


# ---------------------------------------------------------------------------
# Wrapper de Ollama — timeout e exceção
# ---------------------------------------------------------------------------

def test_ollama_timeout_levanta_indisponivel(monkeypatch, mocker, imagem_fake):
    """Se a thread de ollama.chat excede OLLAMA_TIMEOUT_S, deve levantar."""
    monkeypatch.setattr("core.llm.OLLAMA_TIMEOUT_S", 0.05)

    def lento(*args, **kwargs):
        time.sleep(0.3)
        return _ollama_resposta("tarde demais")
    mocker.patch("core.llm.ollama.chat", side_effect=lento)

    with pytest.raises(OllamaIndisponivel, match="Timeout"):
        _ollama_visao("p", imagem_fake)


def test_ollama_excecao_vira_indisponivel(mocker, imagem_fake):
    """Exceção interna do cliente Ollama deve ser convertida para OllamaIndisponivel."""
    mocker.patch("core.llm.ollama.chat", side_effect=ConnectionError("conexão recusada"))
    with pytest.raises(OllamaIndisponivel, match="conexão recusada"):
        _ollama_visao("p", imagem_fake)


# ---------------------------------------------------------------------------
# extrair_json — robustez do parser de saída de LLM
# ---------------------------------------------------------------------------

class TestExtrairJson:
    def test_objeto_limpo(self):
        assert extrair_json('{"a": 1, "b": "x"}') == {"a": 1, "b": "x"}

    def test_array_limpo(self):
        assert extrair_json('[1, 2, 3]') == [1, 2, 3]

    def test_remove_marcadores_de_bloco_json(self):
        s = 'json\n{"a": 1}\n'
        assert extrair_json(s) == {"a": 1}

    def test_remove_marcadores_de_bloco_python(self):
        s = 'python\n[1, 2]\n'
        assert extrair_json(s) == [1, 2]

    def test_extrai_de_meio_de_texto(self):
        s = 'claro, aqui vai: {"a": 1, "b": 2} — espero ter ajudado'
        assert extrair_json(s) == {"a": 1, "b": 2}

    def test_invalido_levanta_value_error(self):
        with pytest.raises(ValueError):
            extrair_json("não tem json aqui")

    def test_aninhado(self):
        s = '{"x": {"y": [1, 2]}}'
        assert extrair_json(s) == {"x": {"y": [1, 2]}}
