"""
Testes para migrations_runner.

Valida:
  - Criação de tabela migrations
  - Aplicação sequencial de migrações
  - Idempotência (aplicar 2x mesmo resultado)
  - Detecção de migrações pendentes
"""
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest

from core.migrations_runner import (
    run_migrations,
    get_migration_status,
    _ensure_migrations_table,
    _get_applied_migrations,
)


@pytest.fixture
def temp_db():
    """Database temporário em memória."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


# ============================================================================
# Testes: Tabela de migrações
# ============================================================================


def test_ensure_migrations_table_cria_tabela(temp_db):
    """Verifica que _ensure_migrations_table cria a tabela."""
    _ensure_migrations_table(temp_db)

    cursor = temp_db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='migrations'"
    )
    assert cursor.fetchone() is not None


def test_get_applied_migrations_vazio(temp_db):
    """Sem migrações aplicadas, retorna set vazio."""
    _ensure_migrations_table(temp_db)
    applied = _get_applied_migrations(temp_db)
    assert applied == set()


def test_get_applied_migrations_com_dados(temp_db):
    """Retorna versões de migrações já aplicadas."""
    _ensure_migrations_table(temp_db)
    temp_db.execute("INSERT INTO migrations (version, name) VALUES (1, 'first')")
    temp_db.execute("INSERT INTO migrations (version, name) VALUES (2, 'second')")
    temp_db.commit()

    applied = _get_applied_migrations(temp_db)
    assert applied == {1, 2}


# ============================================================================
# Testes: Status de migrações
# ============================================================================


def test_get_migration_status_nenhuma_aplicada(temp_db):
    """Status quando nenhuma migração foi aplicada."""
    _ensure_migrations_table(temp_db)

    # Mock para retornar migrações fictícias
    with patch("core.migrations_runner._get_migration_files") as mock_files:
        mock_files.return_value = [
            (1, "001_first", Path("001_first.sql")),
            (2, "002_second", Path("002_second.sql")),
        ]

        status = get_migration_status(temp_db)
        assert status["applied"] == []
        assert status["pending"] == [(1, "001_first"), (2, "002_second")]


def test_get_migration_status_algumas_aplicadas(temp_db):
    """Status com migrações parcialmente aplicadas."""
    _ensure_migrations_table(temp_db)
    temp_db.execute("INSERT INTO migrations (version, name) VALUES (1, '001_first')")
    temp_db.commit()

    with patch("core.migrations_runner._get_migration_files") as mock_files:
        mock_files.return_value = [
            (1, "001_first", Path("001_first.sql")),
            (2, "002_second", Path("002_second.sql")),
        ]

        status = get_migration_status(temp_db)
        assert status["applied"] == [(1, "001_first")]
        assert status["pending"] == [(2, "002_second")]


# ============================================================================
# Testes: Execução de migrações
# ============================================================================


def test_run_migrations_sem_files(temp_db):
    """Sem arquivos de migração, retorna resultado vazio."""
    with patch("core.migrations_runner._get_migration_files") as mock_files:
        mock_files.return_value = []

        result = run_migrations(temp_db)
        assert result["applied"] == []
        assert result["skipped"] == []
        assert result["errors"] == {}


def test_run_migrations_uma_migracao_valida(temp_db):
    """Executa uma migração SQL válida com sucesso."""
    with patch("core.migrations_runner._get_migration_files") as mock_files:
        # Criar arquivo SQL temporário
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".sql", delete=False
        ) as f:
            f.write("CREATE TABLE test_table (id INTEGER PRIMARY KEY);")
            f.flush()

            mock_files.return_value = [(1, "001_test", Path(f.name))]

            result = run_migrations(temp_db)

            assert result["applied"] == [(1, "001_test")]
            assert result["skipped"] == []
            assert result["errors"] == {}

            # Verificar que tabela foi criada
            cursor = temp_db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='test_table'"
            )
            assert cursor.fetchone() is not None


def test_run_migrations_idempotente(temp_db):
    """Aplicar migrações 2x produz mesmo resultado."""
    with patch("core.migrations_runner._get_migration_files") as mock_files:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".sql", delete=False
        ) as f:
            f.write("CREATE TABLE IF NOT EXISTS test (id INTEGER);")
            f.flush()

            mock_files.return_value = [(1, "001_test", Path(f.name))]

            # Primeira execução
            result1 = run_migrations(temp_db)
            assert result1["applied"] == [(1, "001_test")]

            # Segunda execução
            result2 = run_migrations(temp_db)
            assert result2["applied"] == []
            assert result2["skipped"] == [(1, "001_test")]


def test_run_migrations_sql_invalido_faz_rollback(temp_db):
    """Se migração falhar, DB fica em estado anterior (rollback)."""
    with patch("core.migrations_runner._get_migration_files") as mock_files:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".sql", delete=False
        ) as f:
            f.write("CREATE TABLE test (id INTEGER); INVALID SQL HERE;")
            f.flush()

            mock_files.return_value = [(1, "001_invalid", Path(f.name))]

            with pytest.raises(sqlite3.Error):
                run_migrations(temp_db)

            # Verificar que tabela não foi criada (rollback funciona)
            applied = _get_applied_migrations(temp_db)
            assert 1 not in applied


def test_run_migrations_multiplas_sequencial(temp_db):
    """Aplica múltiplas migrações em ordem."""
    with patch("core.migrations_runner._get_migration_files") as mock_files:
        files = []
        for i in range(1, 4):
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".sql", delete=False
            ) as f:
                f.write(f"CREATE TABLE IF NOT EXISTS tab{i} (id INTEGER);")
                f.flush()
                files.append((i, f"00{i}_test", Path(f.name)))

        mock_files.return_value = files

        result = run_migrations(temp_db)

        assert len(result["applied"]) == 3
        assert all(v in [r[0] for r in result["applied"]] for v in [1, 2, 3])

        # Verificar que todas as 3 tabelas foram criadas
        for i in range(1, 4):
            cursor = temp_db.execute(
                f"SELECT name FROM sqlite_master WHERE type='table' AND name='tab{i}'"
            )
            assert cursor.fetchone() is not None
