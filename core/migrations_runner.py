"""
Migration runner para CasaIQ.

Aplicar migrações SQL versionadas de forma segura:
  1. Tabela `migrations` rastreia quais migrações já foram aplicadas
  2. Cada arquivo SQL em core/migrations/NNN_*.sql é uma migração
  3. Runner aplica apenas as que ainda não foram (idempotente)
  4. Falha segura: se uma migração falhar, DB fica no último estado conhecido

Uso:
  from core.migrations_runner import run_migrations
  run_migrations(conn)  # Aplicar todas as pendentes

Estrutura:
  core/migrations/
    001_initial_schema.sql
    002_add_indices.sql
    003_future_change.sql
    ...
"""
import logging
import sqlite3
from pathlib import Path
from typing import Optional

_log = logging.getLogger("casaiq.migrations")


def _ensure_migrations_table(conn: sqlite3.Connection) -> None:
    """Cria tabela `migrations` se não existir."""
    conn.execute("""
    CREATE TABLE IF NOT EXISTS migrations (
        version INTEGER PRIMARY KEY,
        name TEXT NOT NULL UNIQUE,
        applied_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()


def _get_applied_migrations(conn: sqlite3.Connection) -> set[int]:
    """Retorna set de versões já aplicadas."""
    _ensure_migrations_table(conn)
    cursor = conn.execute("SELECT version FROM migrations ORDER BY version")
    return {row[0] for row in cursor.fetchall()}


def _get_migration_files() -> list[tuple[int, str, Path]]:
    """
    Retorna lista de (version, name, path) de todas as migrações.
    Ordem: crescente por versão.
    """
    migrations_dir = Path(__file__).parent / "migrations"
    if not migrations_dir.exists():
        return []

    migrations = []
    for file in sorted(migrations_dir.glob("*.sql")):
        # Esperado: NNN_name.sql (e.g., 001_initial_schema.sql)
        parts = file.stem.split("_", 1)
        if len(parts) == 2 and parts[0].isdigit():
            version = int(parts[0])
            name = file.stem
            migrations.append((version, name, file))

    return migrations


def run_migrations(conn: sqlite3.Connection) -> dict:
    """
    Aplica todas as migrações pendentes.

    Args:
        conn: sqlite3.Connection ativa

    Returns:
        dict com chaves:
        - applied: lista de (version, name) aplicadas nesta sessão
        - skipped: lista de (version, name) já aplicadas antes
        - errors: dict {version: error_message} se houve falhas

    Raises:
        sqlite3.Error se alguma migração falhar
    """
    _ensure_migrations_table(conn)
    applied_versions = _get_applied_migrations(conn)
    migration_files = _get_migration_files()

    result = {"applied": [], "skipped": [], "errors": {}}

    for version, name, filepath in migration_files:
        if version in applied_versions:
            result["skipped"].append((version, name))
            _log.debug(f"Migração {version} ({name}) já foi aplicada")
            continue

        # Ler e aplicar migração
        try:
            sql_content = filepath.read_text(encoding="utf-8")
            _log.info(f"Aplicando migração {version} ({name})...")
            conn.executescript(sql_content)

            # Registrar aplicação
            conn.execute(
                "INSERT INTO migrations (version, name) VALUES (?, ?)",
                (version, name),
            )
            conn.commit()

            result["applied"].append((version, name))
            _log.info(f"✓ Migração {version} ({name}) aplicada com sucesso")

        except Exception as e:
            conn.rollback()
            error_msg = f"{type(e).__name__}: {str(e)}"
            result["errors"][version] = error_msg
            _log.error(f"✗ Migração {version} ({name}) falhou: {error_msg}")
            raise sqlite3.Error(
                f"Falha ao aplicar migração {version} ({name}): {error_msg}"
            ) from e

    return result


def get_migration_status(conn: sqlite3.Connection) -> dict:
    """
    Retorna status das migrações sem aplicar nada.

    Returns:
        dict com chaves:
        - applied: lista de (version, name) já aplicadas
        - pending: lista de (version, name) pendentes
    """
    _ensure_migrations_table(conn)
    applied_versions = _get_applied_migrations(conn)
    migration_files = _get_migration_files()

    applied_list = [
        (v, n) for v, n, _ in migration_files if v in applied_versions
    ]
    pending_list = [
        (v, n) for v, n, _ in migration_files if v not in applied_versions
    ]

    return {"applied": applied_list, "pending": pending_list}
