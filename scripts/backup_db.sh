#!/bin/bash

# CasaIQ Database Backup Script
# Faz backup automático de casaiq.db com retenção inteligente.
# Uso: ./backup_db.sh [--dry-run]
# Cron: 0 2 * * * /home/cuco/projetos/casaiq/scripts/backup_db.sh >> /tmp/casaiq_backup.log 2>&1

set -euo pipefail

# Configuração
CASAIQ_DIR="/home/cuco/projetos/casaiq"
DB_FILE="$CASAIQ_DIR/casaiq.db"
BACKUP_DIR="$CASAIQ_DIR/.backups"
RETENTION_DAYS=30  # manter backups dos últimos 30 dias
DRY_RUN="${1:-}"

# Verificações
if [[ ! -f "$DB_FILE" ]]; then
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERRO: $DB_FILE não encontrado" >&2
    exit 1
fi

if [[ ! -d "$BACKUP_DIR" ]]; then
    mkdir -p "$BACKUP_DIR"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Criado diretório de backups: $BACKUP_DIR"
fi

# Gerar nome do backup: casaiq_YYYY-MM-DD_HHmmss.db
TIMESTAMP=$(date +'%Y-%m-%d_%H%M%S')
BACKUP_FILE="$BACKUP_DIR/casaiq_${TIMESTAMP}.db"

# Fazer backup usando sqlite3 (lida com WAL automaticamente)
if [[ "$DRY_RUN" == "--dry-run" ]]; then
    echo "[DRY-RUN] Seria criado: $BACKUP_FILE"
else
    sqlite3 "$DB_FILE" ".backup '$BACKUP_FILE'" 2>/dev/null || {
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERRO ao fazer backup" >&2
        exit 1
    }
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Backup criado: $BACKUP_FILE ($(du -h "$BACKUP_FILE" | cut -f1))"
fi

# Limpar backups antigos (> RETENTION_DAYS dias)
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Limpando backups antigos (retendo últimos $RETENTION_DAYS dias)..."
find "$BACKUP_DIR" -name "casaiq_*.db" -type f -mtime "+$RETENTION_DAYS" | while read -r old_backup; do
    if [[ "$DRY_RUN" == "--dry-run" ]]; then
        echo "[DRY-RUN] Seria deletado: $old_backup"
    else
        rm -f "$old_backup"
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] Deletado: $old_backup"
    fi
done

# Resumo
BACKUP_COUNT=$(find "$BACKUP_DIR" -name "casaiq_*.db" -type f | wc -l)
BACKUP_SIZE=$(du -sh "$BACKUP_DIR" | cut -f1)
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Status: $BACKUP_COUNT backups no total, $BACKUP_SIZE de espaço"
