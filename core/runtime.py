"""
Estado de runtime mutável para os modelos LLM.
Inicializa a partir de core/config.py mas permite troca em tempo de execução
via POST /api/modelos. Thread-safe (lock + dict).
"""
import threading
from core.config import OLLAMA_VISION_MODEL, OLLAMA_TEXT_MODEL

_lock = threading.Lock()
_estado = {
    "vision_model": OLLAMA_VISION_MODEL,
    "text_model":   OLLAMA_TEXT_MODEL,
}


def get_vision_model() -> str:
    with _lock:
        return _estado["vision_model"]


def get_text_model() -> str:
    with _lock:
        return _estado["text_model"]


def set_models(vision_model: str | None = None, text_model: str | None = None) -> dict:
    with _lock:
        if vision_model:
            _estado["vision_model"] = vision_model
        if text_model:
            _estado["text_model"] = text_model
        return dict(_estado)
