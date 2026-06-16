#!/bin/bash
# Monitor batch scraping em tempo real

echo "📊 Batch Scraping Monitor"
echo "========================="
echo ""

# Ver status do processo
PID_FILE="/home/ubuntu/casaiq_scraper/.batch_orchestrator.pid"

if ssh oracle-casaiq "[ -f $PID_FILE ]" 2>/dev/null; then
  PID=$(ssh oracle-casaiq "cat $PID_FILE")
  
  if ssh oracle-casaiq "kill -0 $PID 2>/dev/null"; then
    echo "✅ Status: RODANDO (PID: $PID)"
  else
    echo "⛔ Status: FINALIZADO"
  fi
else
  echo "⛔ Status: NÃO INICIADO"
fi

echo ""
echo "📋 Últimas linhas do log:"
echo "------------------------"
ssh oracle-casaiq "tail -20 ~/casaiq_scraper/logs/batch_current.log 2>/dev/null" || echo "(Log não disponível)"

echo ""
echo "📊 Relatórios gerados:"
echo "---------------------"
ssh oracle-casaiq "ls -lht ~/casaiq_scraper/batch_reports/ 2>/dev/null | head -5" || echo "(Nenhum relatório ainda)"
