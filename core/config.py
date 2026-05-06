"""Configurações globais do CasaIQ."""
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR             = Path(__file__).parent.parent.resolve()
load_dotenv(BASE_DIR / ".env")

STORAGE_DIR          = BASE_DIR / "storage"
FOTOS_ORIGINAIS_DIR  = STORAGE_DIR / "fotos_originais"
RECORTES_DIR         = STORAGE_DIR / "recortes"
ICONES_DIR           = STORAGE_DIR / "icones"
VIDEOS_ORIGINAIS_DIR = STORAGE_DIR / "videos_originais"
KEYFRAMES_DIR        = STORAGE_DIR / "keyframes"
DB_PATH              = BASE_DIR / "casaiq.db"

for _d in [FOTOS_ORIGINAIS_DIR, RECORTES_DIR, ICONES_DIR,
           VIDEOS_ORIGINAIS_DIR, KEYFRAMES_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# Ollama
OLLAMA_VISION_MODEL  = os.getenv("CASAIQ_VISION_MODEL",  "qwen2.5vl:7b")
OLLAMA_TEXT_MODEL    = os.getenv("CASAIQ_TEXT_MODEL",    "llama3.2:3b")
OLLAMA_BASE_URL      = os.getenv("OLLAMA_BASE_URL",      "http://localhost:11434")
OLLAMA_TIMEOUT_S     = int(os.getenv("OLLAMA_TIMEOUT_S", "120"))

# Claude API (Anthropic) — opcional
ANTHROPIC_API_KEY    = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL      = "claude-sonnet-4-20250514"

# Modo de operação do roteador inteligente
# Valores válidos: offline | local_primeiro | hibrido | claude | inteligente
CASAIQ_MODO          = os.getenv("CASAIQ_MODO", "inteligente")

# Limiar de confiança: abaixo disso, roteador considera pedir segunda opinião
LIMIAR_CONFIANCA     = float(os.getenv("CASAIQ_LIMIAR_CONFIANCA", "0.65"))

# Servidor HTTP
API_HOST             = os.getenv("CASAIQ_HOST", "0.0.0.0")
API_PORT             = int(os.getenv("CASAIQ_PORT", "8000"))

# Vídeo — extração de keyframes
CASAIQ_VIDEO_FPS         = float(os.getenv("CASAIQ_VIDEO_FPS", "0.5"))
CASAIQ_VIDEO_MAX_FRAMES  = int(os.getenv("CASAIQ_VIDEO_MAX_FRAMES", "10"))
