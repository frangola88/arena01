"""Fixtures compartilhadas dos testes do CasaIQ.

Os módulos `core/*` fazem `from core.config import X` no topo, o que liga `X`
ao namespace do módulo importador. Para sobrescrever `CASAIQ_MODO`,
`ANTHROPIC_API_KEY` ou `LIMIAR_CONFIANCA` em testes é preciso patchar o nome
no módulo consumidor (ex.: `core.roteador.CASAIQ_MODO`), não em `core.config`.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Configuração do roteador
# ---------------------------------------------------------------------------

@pytest.fixture
def set_modo(monkeypatch):
    """Fixa CASAIQ_MODO no namespace do roteador. Uso: set_modo("offline")."""
    def _apply(modo: str):
        monkeypatch.setattr("core.roteador.CASAIQ_MODO", modo)
    return _apply


@pytest.fixture
def set_api(monkeypatch):
    """Fixa ANTHROPIC_API_KEY nos namespaces que a importam como nome local.

    `core.llm` também faz `from core.config import ANTHROPIC_API_KEY` no topo,
    então a fallback chain de chamar_visao/chamar_texto consulta a cópia dela —
    é preciso patchar nos dois lugares.
    """
    def _apply(key: str):
        monkeypatch.setattr("core.roteador.ANTHROPIC_API_KEY", key)
        try:
            import core.llm  # noqa: F401
            monkeypatch.setattr("core.llm.ANTHROPIC_API_KEY", key)
        except ImportError:
            pass
    return _apply


@pytest.fixture
def set_limiar(monkeypatch):
    """Fixa LIMIAR_CONFIANCA no namespace do roteador."""
    def _apply(valor: float):
        monkeypatch.setattr("core.roteador.LIMIAR_CONFIANCA", valor)
    return _apply


@pytest.fixture
def com_api(set_api):
    """Atalho: ANTHROPIC_API_KEY definida."""
    set_api("sk-test-key")


@pytest.fixture
def sem_api(set_api):
    """Atalho: ANTHROPIC_API_KEY vazia."""
    set_api("")


# ---------------------------------------------------------------------------
# Banco de dados isolado (usado nas fases 3 e 4)
# ---------------------------------------------------------------------------

@pytest.fixture
def db_temp(tmp_path, monkeypatch) -> Path:
    """Aponta DB_PATH para um arquivo SQLite temporário e roda init_db()."""
    db_path = tmp_path / "casaiq_test.db"
    # Patcha em todos os módulos que importaram DB_PATH como nome local
    monkeypatch.setattr("core.config.DB_PATH", db_path)
    monkeypatch.setattr("core.database.DB_PATH", db_path)

    from core.database import init_db
    init_db()
    yield db_path

    # SQLite WAL pode deixar arquivos auxiliares — tmp_path cuida da limpeza
