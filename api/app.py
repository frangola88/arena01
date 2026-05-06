"""
API FastAPI. CRÍTICO: StaticFiles montado APÓS todos os routers /api.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from core.database import init_db
from core.roteador import descricao_modo
from api.routes.localizacoes import router as r_loc
from api.routes.fotos        import router as r_fotos
from api.routes.videos       import router as r_videos
from api.routes.objetos      import router as r_obj
from api.routes.chat         import router as r_chat
from api.routes.estatisticas import router as r_stats
from api.routes.modelos      import router as r_modelos

app = FastAPI(title="CasaIQ", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # uso local — não usar wildcard de subdomínio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    init_db()
    print(f"[CasaIQ] Modo de operação: {descricao_modo()}")

# Endpoint de status do roteador (útil para a interface mostrar o modo ativo)
@app.get("/api/modo")
def get_modo():
    from core.roteador import descricao_modo, _modo, _tem_api
    from core.config import (OLLAMA_TIMEOUT_S, CASAIQ_VIDEO_FPS,
                             CASAIQ_VIDEO_MAX_FRAMES)
    from core.runtime import get_vision_model, get_text_model
    return {
        "modo": str(_modo()),
        "descricao": descricao_modo(),
        "tem_api_key": _tem_api(),
        "vision_model": get_vision_model(),
        "text_model":   get_text_model(),
        "timeout_s":    OLLAMA_TIMEOUT_S,
        "video_fps":    CASAIQ_VIDEO_FPS,
        "video_max_frames": CASAIQ_VIDEO_MAX_FRAMES,
    }

# Favicon vazio — silencia 404 cosmético do navegador
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    from fastapi.responses import Response
    return Response(status_code=204)

# Routers /api — ANTES do StaticFiles
app.include_router(r_loc,     prefix="/api")
app.include_router(r_fotos,   prefix="/api")
app.include_router(r_videos,  prefix="/api")
app.include_router(r_obj,     prefix="/api")
app.include_router(r_chat,    prefix="/api")
app.include_router(r_stats,   prefix="/api")
app.include_router(r_modelos, prefix="/api")

BASE_DIR = Path(__file__).parent.parent
# Servir imagens de storage (recortes, ícones)
app.mount("/storage", StaticFiles(directory=str(BASE_DIR / "storage")), name="storage")
# StaticFiles da interface — SEMPRE por último
app.mount("/", StaticFiles(directory=str(BASE_DIR / "web"), html=True), name="web")
