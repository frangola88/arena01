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
    think=False: desabilita o modo de raciocínio do qwen3-vl (que consome todos os
    tokens na fase de thinking e deixa message.content vazio).
    """
    resultado, excecao = [None], [None]
    def _run():
        try:
            resp = ollama.chat(
                model=get_vision_model(),
                messages=[{"role": "user", "content": prompt, "images": [imagem_path]}],
                think=False,
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
                think=False,
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
    """Chama Claude API via httpx direto — bypass do SDK 0.97+ que tem bug de connection."""
    import base64, json, httpx
    ext = Path(imagem_path).suffix.lower()
    mt  = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
           ".png": "image/png",  ".webp": "image/webp"}.get(ext, "image/jpeg")
    with open(imagem_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode()
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 16384,
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": mt, "data": data}},
            {"type": "text",  "text": prompt},
        ]}],
    }
    r = httpx.post("https://api.anthropic.com/v1/messages",
                   headers=headers, json=body, timeout=240)
    if r.status_code != 200:
        raise RuntimeError(f"Claude API {r.status_code}: {r.text[:300]}")
    resp = r.json()
    _log.info("claude_visao_chamada", extra={"modelo": ANTHROPIC_MODEL})
    return resp["content"][0]["text"]


def _claude_texto(prompt: str, max_tokens: int = 512) -> str:
    """Chama Claude API via httpx direto — bypass do SDK 0.97+ que tem bug de connection."""
    import json, httpx
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    r = httpx.post("https://api.anthropic.com/v1/messages",
                   headers=headers, json=body, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Claude API {r.status_code}: {r.text[:300]}")
    resp = r.json()
    _log.info("claude_texto_chamada", extra={"modelo": ANTHROPIC_MODEL, "max_tokens": max_tokens})
    return resp["content"][0]["text"]


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
    Remove blocos <think>...</think> do qwen3 antes de parsear.
    Não usa backtick no código para não quebrar blocos Markdown.
    """
    if not texto or not texto.strip():
        raise ValueError("Resposta vazia do modelo")
    # Remove thinking do qwen3/qwen2.5 que pode vazar para o content
    texto = re.sub(r'<think>.*?</think>', '', texto, flags=re.DOTALL)
    texto = texto.strip()
    # Remove fences markdown (```json ... ``` ou ``` ... ```)
    texto = re.sub(r'^```[a-z]*\s*\n?', '', texto)
    if texto.endswith('```'):
        texto = texto[:-3].rstrip()
    texto = texto.strip()
    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        match = re.search(r'(\{.*\}|\[.*\])', texto, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise ValueError(f"JSON não encontrado em: {texto[:200]}")
