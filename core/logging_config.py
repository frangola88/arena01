"""Configuração de logging estruturado para o CasaIQ.

Saída: uma linha JSON por evento, em stdout (uvicorn captura). Permite
`grep`, `jq`, ou ingestão em qualquer pipeline de logs sem parser custom.

Uso:
    from core.logging_config import setup_logging
    setup_logging()  # idempotente; chamar uma vez no startup do processo

Para emitir eventos estruturados:
    log = logging.getLogger("casaiq.roteador")
    log.info("decisao_visao", extra={"modo": "offline", "decisao": False})
"""
import json
import logging
import sys
from typing import Any


# Atributos padrão do LogRecord — não duplicar no payload JSON.
_RESERVADOS = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "taskName",
})


class JSONFormatter(logging.Formatter):
    """Serializa cada LogRecord como uma linha JSON. Atributos extras
    passados via `logger.info(msg, extra={...})` aparecem no topo do payload.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "evento": record.getMessage(),
        }
        for k, v in record.__dict__.items():
            if k in _RESERVADOS or k.startswith("_"):
                continue
            payload[k] = v
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(level: str = "INFO") -> None:
    """Configura o root logger com handler JSON em stdout. Idempotente:
    chamar múltiplas vezes não duplica handlers."""
    root = logging.getLogger()
    if any(getattr(h, "_casaiq_marker", False) for h in root.handlers):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    handler._casaiq_marker = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
