"""Testes de core.logging_config — formatter JSON + setup idempotente.

A configuração não é exercitada em testes que usam caplog (caplog instala seu
próprio handler), mas o módulo precisa funcionar quando o app inicia. Aqui
checamos diretamente o JSONFormatter e o setup_logging.
"""
from __future__ import annotations

import json
import logging

from core.logging_config import JSONFormatter, setup_logging


# ---------------------------------------------------------------------------
# JSONFormatter
# ---------------------------------------------------------------------------

def _record(msg: str, **extras) -> logging.LogRecord:
    """Cria um LogRecord como o logging.Logger.makeRecord faria."""
    rec = logging.LogRecord(
        name="casaiq.teste", level=logging.INFO, pathname=__file__,
        lineno=1, msg=msg, args=(), exc_info=None,
    )
    for k, v in extras.items():
        setattr(rec, k, v)
    return rec


def test_formatter_emite_json_valido_com_campos_padrao():
    out = JSONFormatter().format(_record("evento_x"))
    payload = json.loads(out)
    for chave in ("ts", "level", "logger", "evento"):
        assert chave in payload
    assert payload["level"] == "INFO"
    assert payload["logger"] == "casaiq.teste"
    assert payload["evento"] == "evento_x"


def test_formatter_inclui_campos_extras():
    rec = _record("decisao_visao", modo="offline", decisao=False, motivo="x")
    payload = json.loads(JSONFormatter().format(rec))
    assert payload["modo"] == "offline"
    assert payload["decisao"] is False
    assert payload["motivo"] == "x"


def test_formatter_nao_duplica_campos_reservados_do_logrecord():
    rec = _record("e", modo="claude")
    payload = json.loads(JSONFormatter().format(rec))
    # Atributos do LogRecord (filename, lineno, etc.) NÃO devem vazar:
    for proibido in ("filename", "lineno", "pathname", "msecs", "thread"):
        assert proibido not in payload


def test_formatter_serializa_objetos_nao_jsonaveis_via_str():
    """`default=str` deve garantir que tipos exóticos não quebrem o formatter."""
    class Coisa:
        def __repr__(self): return "<Coisa>"
    rec = _record("e", obj=Coisa())
    payload = json.loads(JSONFormatter().format(rec))
    assert payload["obj"] == "<Coisa>"


def test_formatter_inclui_traceback_em_exc_info():
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        import sys
        rec = logging.LogRecord(
            name="casaiq.teste", level=logging.ERROR, pathname=__file__,
            lineno=1, msg="erro", args=(), exc_info=sys.exc_info(),
        )
    payload = json.loads(JSONFormatter().format(rec))
    assert "exc" in payload
    assert "RuntimeError" in payload["exc"]
    assert "boom" in payload["exc"]


# ---------------------------------------------------------------------------
# setup_logging — idempotente
# ---------------------------------------------------------------------------

def test_setup_logging_e_idempotente():
    """Múltiplas chamadas não devem empilhar handlers."""
    root = logging.getLogger()
    pre = list(root.handlers)
    setup_logging()
    setup_logging()
    setup_logging()
    pos = list(root.handlers)
    casaiq_handlers = [h for h in pos if getattr(h, "_casaiq_marker", False)]
    assert len(casaiq_handlers) == 1
    # Limpeza pra não vazar handler entre arquivos de teste
    for h in casaiq_handlers:
        if h not in pre:
            root.removeHandler(h)
