"""
API FastAPI. CRÍTICO: StaticFiles montado APÓS todos os routers /api.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from core.database import init_db
from core.logging_config import setup_logging
from core.roteador import descricao_modo
from api.routes.localizacoes    import router as r_loc
from api.routes.fotos           import router as r_fotos
from api.routes.videos          import router as r_videos
from api.routes.objetos         import router as r_obj
from api.routes.chat            import router as r_chat
from api.routes.estatisticas    import router as r_stats
from api.routes.modelos         import router as r_modelos
from api.routes.observabilidade import router as r_observabilidade
from api.routes.admin           import router as r_admin


_log = logging.getLogger("casaiq.app")


def _reconciliar_jobs_orfaos() -> None:
    """
    Reconciliação no boot: se o processo anterior foi finalizado/reiniciado
    enquanto fotos ou vídeos estavam sendo processados, marca-os como erro
    para destravar a interface e permitir reprocessamento.
    """
    try:
        from core.database import get_db
        conn = get_db()
        try:
            cursor_fotos = conn.execute("""
                UPDATE fotos_processadas
                SET status = 'erro',
                    erro_mensagem = 'Processamento interrompido por reinício do servidor',
                    concluido_em = CURRENT_TIMESTAMP
                WHERE status IN ('pendente', 'processando')
            """)
            cursor_videos = conn.execute("""
                UPDATE videos_processados
                SET status = 'erro',
                    erro_mensagem = 'Processamento interrompido por reinício do servidor',
                    concluido_em = CURRENT_TIMESTAMP
                WHERE status IN ('pendente', 'processando')
            """)
            conn.commit()
            total_reconciliados = cursor_fotos.rowcount + cursor_videos.rowcount
            if total_reconciliados > 0:
                _log.warning("jobs_orfaos_reconciliados", extra={
                    "fotos": cursor_fotos.rowcount,
                    "videos": cursor_videos.rowcount,
                })
        finally:
            conn.close()
    except Exception as e:
        _log.error("erro_reconciliacao_jobs_orfaos", extra={"erro": str(e)})


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    init_db()
    _reconciliar_jobs_orfaos()
    _log.info("startup", extra={"descricao_modo": descricao_modo()})
    yield


app = FastAPI(title="CasaIQ", version="3.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # uso local — não usar wildcard de subdomínio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
app.include_router(r_loc,              prefix="/api")
app.include_router(r_fotos,            prefix="/api")
app.include_router(r_videos,           prefix="/api")
app.include_router(r_obj,              prefix="/api")
app.include_router(r_chat,             prefix="/api")
app.include_router(r_stats,            prefix="/api")
app.include_router(r_modelos,          prefix="/api")
app.include_router(r_observabilidade,  prefix="/api")
app.include_router(r_admin,            prefix="/api")

BASE_DIR = Path(__file__).parent.parent
# Servir imagens de storage (recortes, ícones)
app.mount("/storage", StaticFiles(directory=str(BASE_DIR / "storage")), name="storage")
# StaticFiles da interface — SEMPRE por último
app.mount("/", StaticFiles(directory=str(BASE_DIR / "web"), html=True), name="web")
