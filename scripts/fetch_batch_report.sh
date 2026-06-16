#!/bin/bash
# Puxar relatório final do batch

REPORT_DIR="~/casaiq_batch_reports"
mkdir -p "$REPORT_DIR"

echo "📥 Puxando relatórios do Oracle VM..."

rsync -avz oracle-casaiq:~/casaiq_scraper/batch_reports/ "$REPORT_DIR/" 2>/dev/null

echo ""
echo "✅ Relatórios salvos em: $(cd $REPORT_DIR && pwd)"
echo ""
echo "📋 Relatórios disponíveis:"
ls -lh "$REPORT_DIR"/*.json 2>/dev/null | tail -5 || echo "Nenhum relatório encontrado"

echo ""
echo "📊 Último relatório:"
LATEST=$(ls -t "$REPORT_DIR"/*.json 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
  echo "Arquivo: $LATEST"
  echo ""
  echo "Resumo rápido:"
  python3 << 'PY'
import json, sys
try:
  with open(sys.argv[1]) as f:
    r = json.load(f)
  print(f"  Batch ID:   {r['batch_id']}")
  print(f"  Total:      {r['total_spiders']} spiders")
  print(f"  ✅ Sucesso: {r['successful']}")
  print(f"  ⚠️  Parcial: {r['partial']}")
  print(f"  ❌ Erros:   {r['errors']}")
  print(f"  📦 Itens:   {r['total_items']:,}")
  print(f"  ⏱️  Tempo:   {r['duration_seconds']//3600}h {(r['duration_seconds']%3600)//60}m")
except Exception as e:
  print(f"Erro ao ler relatório: {e}")
PY
  "$LATEST"
fi
