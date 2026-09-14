#!/usr/bin/env bash
# CasaIQ — Script de Inicialização Rápida & Verificação de Saúde
# Uso: ./casaiq_start.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🏠 ========================================================="
echo "🏠 Iniciando CasaIQ Pro v4.0 — Inventário Doméstico"
echo "🏠 ========================================================="

# 1. Verificar Ollama
if curl -s http://localhost:11434 >/dev/null 2>&1; then
    echo "🧠 [OK] Ollama ativo em http://localhost:11434"
else
    echo "⚠️  [AVISO] Ollama não está rodando. O modo offline/local precisará do Ollama."
    echo "    Inicie com: sudo systemctl start ollama"
fi

# 2. Verificar Ambiente Python
PYTHON_BIN="/home/cuco/miniconda/envs/casaiq/bin/python"
UVICORN_BIN="/home/cuco/miniconda/envs/casaiq/bin/uvicorn"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "❌ [ERRO] Ambiente conda 'casaiq' não encontrado em $PYTHON_BIN"
    exit 1
fi
echo "🐍 [OK] Python: $($PYTHON_BIN --version)"

# 3. Integridade do Banco SQLite
if [[ -f "casaiq.db" ]]; then
    INTEGRIDADE=$("$PYTHON_BIN" -c "
import sqlite3
conn = sqlite3.connect('casaiq.db')
res = conn.execute('PRAGMA integrity_check;').fetchone()[0]
print(res)
conn.close()
")
    if [[ "$INTEGRIDADE" == "ok" ]]; then
        echo "💾 [OK] SQLite WAL integridade: OK"
    else
        echo "⚠️  [ALERTA] Integridade do banco retornou: $INTEGRIDADE"
    fi
fi

# 4. Iniciar Servidor
echo "🚀 [OK] Servidor iniciando em http://0.0.0.0:8000"
echo "   Pressione Ctrl+C para encerrar."
echo "========================================================="

exec "$UVICORN_BIN" api.app:app --host 0.0.0.0 --port 8000 --reload
