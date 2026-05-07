"""
Execução das chamadas aos modelos de linguagem.
A DECISÃO de qual chamar está em core/roteador.py.
Este módulo apenas executa, com timeout e tratamento de erros.
"""
import json
import logging
import re
import threading
from pathlib import Path
from typing import Optional
import ollama
from core.config import (
    OLLAMA_TIMEOUT_S,
    ANTHROPIC_API_KEY, ANTHROPIC_MODEL,
)
from core.runtime import get_vision_model, get_text_model
from core.roteador import TarefaTexto, deve_usar_claude_visao, deve_usar_claude_texto

_log = logging.getLogger("casaiq.llm")


class OllamaIndisponivel(Exception):
    pass


# --- Ollama ------------------------------------------------------------------

def _ollama_visao(prompt: str, imagem_path: str) -> str:
    """
    Chamada de visão via Ollama.
    images= recebe o CAMINHO DO ARQUIVO (string) — a lib encoda internamente.
    """
    resultado, excecao = [None], [None]
    def _run():
        try:
            resp = ollama.chat(
                model=get_vision_model(),
                messages=[{"role": "user", "content": prompt, "images": [imagem_path]}],
            )
            resultado[0] = resp["message"]["content"]
        except Exception as e:
            excecao[0] = e
    t = threading.Thread(target=_run)
    t.start(); t.join(timeout=OLLAMA_TIMEOUT_S)
    if t.is_alive():
        raise OllamaIndisponivel(f"Timeout {OLLAMA_TIMEOUT_S}s")
    if excecao[0]:
        raise OllamaIndisponivel(str(excecao[0]))
    return resultado[0]


def _ollama_texto(prompt: str) -> str:
    resultado, excecao = [None], [None]
    def _run():
        try:
            resp = ollama.chat(
                model=get_text_model(),
                messages=[{"role": "user", "content": prompt}],
            )
            resultado[0] = resp["message"]["content"]
        except Exception as e:
            excecao[0] = e
    t = threading.Thread(target=_run)
    t.start(); t.join(timeout=OLLAMA_TIMEOUT_S)
    if t.is_alive():
        raise OllamaIndisponivel(f"Timeout {OLLAMA_TIMEOUT_S}s")
    if excecao[0]:
        raise OllamaIndisponivel(str(excecao[0]))
    return resultado[0]


# --- Claude API --------------------------------------------------------------

def _claude_visao(prompt: str, imagem_path: str) -> str:
    import anthropic, base64
    ext = Path(imagem_path).suffix.lower()
    mt  = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
           ".png": "image/png",  ".webp": "image/webp"}.get(ext, "image/jpeg")
    with open(imagem_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode()
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=ANTHROPIC_MODEL, max_tokens=1024,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": mt, "data": data}},
            {"type": "text",  "text": prompt},
        ]}],
    )
    _log.info("claude_visao_chamada", extra={"modelo": ANTHROPIC_MODEL})
    return resp.content[0].text


def _claude_texto(prompt: str, max_tokens: int = 512) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=ANTHROPIC_MODEL, max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    _log.info("claude_texto_chamada", extra={
        "modelo": ANTHROPIC_MODEL, "max_tokens": max_tokens,
    })
    return resp.content[0].text


# --- Funções públicas (usadas pelos agentes) ---------------------------------

def chamar_visao(prompt: str, imagem_path: str,
                 confianca_anterior: Optional[float] = None) -> tuple[str, str]:
    """
    Retorna (resposta, modelo_usado).
    Consulta o roteador para decidir local vs Claude.
    Se Claude for indicado, tenta Claude primeiro (não espera o local falhar).
    Se local for indicado mas falhar, tenta Claude como fallback (exceto modo offline).
    """
    from core.roteador import Modo, _modo
    usar_claude = deve_usar_claude_visao(confianca_anterior)

    if usar_claude:
        try:
            return _claude_visao(prompt, imagem_path), "claude_api"
        except Exception as e:
            _log.warning("claude_visao_falhou_fallback_ollama", extra={"erro": str(e)})
    try:
        return _ollama_visao(prompt, imagem_path), "ollama"
    except OllamaIndisponivel as e:
        if ANTHROPIC_API_KEY and _modo() != Modo.OFFLINE and not usar_claude:
            _log.warning("ollama_visao_indisponivel_fallback_claude", extra={"erro": str(e)})
            return _claude_visao(prompt, imagem_path), "claude_api"
        raise RuntimeError(f"Visão indisponível: Ollama falhou ({e}) e sem Claude API.") from e


def chamar_texto(prompt: str, tarefa: TarefaTexto = TarefaTexto.RESPOSTA_CHAT,
                 max_tokens: int = 512) -> tuple[str, str]:
    """
    Retorna (resposta, modelo_usado).
    Roteador decide por tipo de tarefa — tarefas de alta complexidade vão para Claude.
    """
    from core.roteador import Modo, _modo
    usar_claude = deve_usar_claude_texto(tarefa)

    if usar_claude:
        try:
            return _claude_texto(prompt, max_tokens), "claude_api"
        except Exception as e:
            _log.warning("claude_texto_falhou_fallback_ollama",
                         extra={"erro": str(e), "tarefa": tarefa.value})
    try:
        return _ollama_texto(prompt), "ollama"
    except OllamaIndisponivel as e:
        if ANTHROPIC_API_KEY and _modo() != Modo.OFFLINE and not usar_claude:
            _log.warning("ollama_texto_indisponivel_fallback_claude",
                         extra={"erro": str(e), "tarefa": tarefa.value})
            return _claude_texto(prompt, max_tokens), "claude_api"
        raise RuntimeError(f"Texto indisponível: Ollama falhou ({e}) e sem Claude API.") from e


def extrair_json(texto: str):
    """
    Extrai JSON de resposta de modelo que pode conter texto extra.
    Não usa backtick no código para não quebrar blocos Markdown.
    """
    texto = texto.strip()
    linhas = [l for l in texto.splitlines()
              if l.strip().lower() not in ("json", "python", "")]
    texto = "\n".join(linhas).strip()
    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        match = re.search(r'(\{.*\}|\[.*\])', texto, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise ValueError(f"JSON não encontrado em: {texto[:200]}")
