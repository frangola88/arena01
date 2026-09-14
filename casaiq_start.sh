#!/usr/bin/env bash
# CasaIQ Pro v4.0 — Script de Inicialização Rápida & Verificação de Saúde
# Uso: ./casaiq_start.sh [--no-inbox]

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

# 4. Estrutura do Inbox Watcher (Zero-Click Ingest)
mkdir -p storage/inbox/processados storage/inbox/erros storage/fotos_originais storage/videos_originais
echo "📥 [OK] Pasta Inbox monitorada: storage/inbox/"

# Iniciar Inbox Watcher em background (se não desabilitado)
WATCHER_PID=""
if [[ "${1:-}" != "--no-inbox" ]]; then
    "$PYTHON_BIN" scripts/inbox_watcher.py --daemon >/dev/null 2>&1 &
    WATCHER_PID=$!
    echo "👁️  [OK] Inbox Watcher ativo em segundo plano (PID: $WATCHER_PID)"
    trap 'echo -e "\n🛑 Encerrando CasaIQ e Inbox Watcher..."; kill $WATCHER_PID 2>/dev/null || true; exit 0' INT TERM EXIT
fi

# 5. Detecção de IP LAN para Acesso Móvel PWA
LAN_IP=$(ip route get 1 2>/dev/null | awk '{print $7}')
echo "🌐 ========================================================="
echo "🚀 [OK] Servidor Local:   http://localhost:8000"
if [[ -n "$LAN_IP" ]]; then
    echo "📱 [OK] Acesso Celular:  http://$LAN_IP:8000"
    echo "   (Abra no navegador do celular e adicione à Tela Inicial como PWA)"
fi
echo "========================================================="
echo "   Pressione Ctrl+C para encerrar."

exec "$UVICORN_BIN" api.app:app --host 0.0.0.0 --port 8000 --reload
