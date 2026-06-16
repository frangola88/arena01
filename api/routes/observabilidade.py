"""
Observabilidade: métricas da pipeline e logs estruturados.

Endpoints:
  GET /api/observabilidade/saude       — health check + métricas básicas
  GET /api/observabilidade/logs        — últimos logs estruturados
  GET /api/observabilidade/metricas    — pipeline metrics (fotos, vídeos, objetos)
"""
import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Query

from core.database import get_db
from core.roteador import descricao_modo, _modo

_log = logging.getLogger("casaiq.observabilidade")

router = APIRouter(tags=["observabilidade"])

# Store de logs estruturados (em memória, últimos N)
_LOGS_BUFFER = []
_MAX_LOGS = 100


class LogHandler(logging.Handler):
    """Handler que captura logs estruturados para /api/observabilidade/logs."""
    
    def emit(self, record: logging.LogRecord):
        try:
            if hasattr(record, "extra"):
                extra = record.extra
            else:
                extra = {k: v for k, v in record.__dict__.items()
                        if k not in ("name", "msg", "args", "created", "levelname", "levelno")}
            
            log_entry = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "logger": record.name,
                "level": record.levelname,
                "message": record.getMessage(),
                "extra": extra
            }
            
            _LOGS_BUFFER.append(log_entry)
            # Manter apenas últimos N logs
            if len(_LOGS_BUFFER) > _MAX_LOGS:
                _LOGS_BUFFER.pop(0)
        except Exception as e:
            # Não lançar erro em logger
            pass


# Instalar handler nos loggers principais
for logger_name in ["casaiq.app", "casaiq.migrations", "casaiq.agent_1", 
                    "casaiq.agent_2", "casaiq.agent_3", "casaiq.agent_4"]:
    logger = logging.getLogger(logger_name)
    if not any(isinstance(h, LogHandler) for h in logger.handlers):
        logger.addHandler(LogHandler())


@router.get("/observabilidade/saude")
def health_check():
    """
    Health check com status básico.
    
    Retorna:
      - modo: Current operational mode
      - status: "ok" se tudo funcionando
      - timestamp: Server time
    """
    return {
        "status": "ok",
        "modo": str(_modo()),
        "descricao": descricao_modo(),
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/observabilidade/metricas")
def metricas_pipeline():
    """
    Métricas da pipeline.
    
    Retorna contadores de:
      - Fotos processadas (sucesso, erro, pendente)
      - Vídeos processados (sucesso, erro, pendente)
      - Objetos detectados total
      - Categorias disponíveis
    """
    conn = get_db()
    try:
        # Fotos por status
        fotos = conn.execute(
            "SELECT status, COUNT(*) as count FROM fotos_processadas GROUP BY status"
        ).fetchall()
        fotos_dict = {row[0]: row[1] for row in fotos}
        
        # Vídeos por status
        videos = conn.execute(
            "SELECT status, COUNT(*) as count FROM videos_processados GROUP BY status"
        ).fetchall()
        videos_dict = {row[0]: row[1] for row in videos}
        
        # Total de objetos
        total_objetos = conn.execute(
            "SELECT COUNT(*) FROM objetos"
        ).fetchone()[0]
        
        # Objetos por categoria
        por_categoria = conn.execute("""
            SELECT c.nome, COUNT(o.id) as count
            FROM categorias c
            LEFT JOIN objetos o ON o.categoria_id = c.id
            GROUP BY c.id, c.nome
            ORDER BY count DESC
        """).fetchall()
        
        # Localizações
        total_localizacoes = conn.execute(
            "SELECT COUNT(*) FROM localizacoes"
        ).fetchone()[0]
        
        return {
            "fotos_processadas": {
                "total": sum(fotos_dict.values()),
                "por_status": fotos_dict
            },
            "videos_processados": {
                "total": sum(videos_dict.values()),
                "por_status": videos_dict
            },
            "objetos": {
                "total": total_objetos,
                "por_categoria": [
                    {"categoria": row[0], "count": row[1]} for row in por_categoria
                ]
            },
            "localizacoes": {
                "total": total_localizacoes
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    finally:
        conn.close()


@router.get("/observabilidade/logs")
def obter_logs(
    ultimos: int = Query(50, ge=1, le=100),
    nivel: Optional[str] = Query(None, pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"),
    logger: Optional[str] = None
):
    """
    Últimos logs estruturados.
    
    Query params:
      - ultimos: Número de últimos logs (1-100, padrão 50)
      - nivel: Filtrar por nível (DEBUG|INFO|WARNING|ERROR|CRITICAL)
      - logger: Filtrar por nome do logger
    
    Retorna:
      - logs: Lista de entries com timestamp, logger, level, message, extra
      - total: Total de logs capturados
    """
    filtrados = _LOGS_BUFFER
    
    # Filtrar por nível
    if nivel:
        filtrados = [l for l in filtrados if l["level"] == nivel]
    
    # Filtrar por logger
    if logger:
        filtrados = [l for l in filtrados if logger.lower() in l["logger"].lower()]
    
    # Retornar últimos N
    resultado = filtrados[-ultimos:] if len(filtrados) > ultimos else filtrados
    
    return {
        "logs": resultado,
        "total_capturado": len(_LOGS_BUFFER),
        "total_filtrado": len(filtrados),
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/observabilidade/resumo")
def resumo_operacional():
    """
    Resumo operacional completo.
    
    Combina saúde + metricas + últimos logs para um dashboard unificado.
    """
    return {
        "saude": health_check(),
        "metricas": metricas_pipeline(),
        "logs_recentes": obter_logs(ultimos=20),
        "timestamp": datetime.utcnow().isoformat()
    }
