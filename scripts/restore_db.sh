#!/bin/bash

# CasaIQ Database Restore Script
# Restaura casaiq.db a partir de um backup.
# Uso: ./restore_db.sh <caminho_do_backup>
# Exemplo: ./restore_db.sh .backups/casaiq_2026-05-13_191216.db

set -euo pipefail

CASAIQ_DIR="/home/cuco/projetos/casaiq"
DB_FILE="$CASAIQ_DIR/casaiq.db"
BACKUP_SOURCE="${1:-}"

# Validações
if [[ -z "$BACKUP_SOURCE" ]]; then
    echo "Uso: $0 <caminho_do_backup>"
    echo "Exemplo: $0 .backups/casaiq_2026-05-13_191216.db"
    exit 1
fi

# Resolver caminho absoluto se for relativo
if [[ ! "$BACKUP_SOURCE" = /* ]]; then
    BACKUP_SOURCE="$CASAIQ_DIR/$BACKUP_SOURCE"
fi

if [[ ! -f "$BACKUP_SOURCE" ]]; then
    echo "ERRO: Backup não encontrado: $BACKUP_SOURCE" >&2
    exit 1
fi

if [[ ! -f "$DB_FILE" ]]; then
    echo "ERRO: Banco atual não encontrado: $DB_FILE" >&2
    exit 1
fi

# Backup de segurança do DB atual
SAFETY_BACKUP="$DB_FILE.restore_safety_$(date +'%Y%m%d_%H%M%S').bak"
cp "$DB_FILE" "$SAFETY_BACKUP"
echo "Backup de segurança criado: $SAFETY_BACKUP"

# Restaurar (sqlite3 sobrescreve o destino)
sqlite3 "$BACKUP_SOURCE" ".backup '$DB_FILE'" 2>/dev/null || {
    echo "ERRO ao restaurar. Revertendo para: $SAFETY_BACKUP" >&2
    cp "$SAFETY_BACKUP" "$DB_FILE"
    exit 1
}

echo "✓ Restaurado com sucesso de: $BACKUP_SOURCE"
echo "  Tamanho: $(du -h "$DB_FILE" | cut -f1)"
echo "  Backup de segurança preservado em: $SAFETY_BACKUP"
