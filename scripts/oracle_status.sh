#!/bin/bash
# Monitora o progresso do scraping no Oracle VM
# Uso: ./scripts/oracle_status.sh

echo "=== Oracle CasaIQ Scraper — $(date '+%d/%m %H:%M') ==="
echo

ssh oracle-casaiq "
echo '--- SPIDERS ATIVOS ---'
/usr/bin/tmux ls 2>/dev/null || echo 'nenhum spider rodando'

echo
echo '--- ITENS COLETADOS ---'
for f in ~/casaiq_scraper/data/export/*.jsonl; do
    [ -f \"\$f\" ] && printf '  %-25s %d itens\n' \$(basename \$f) \$(wc -l < \"\$f\")
done

echo
echo '--- IMAGENS ---'
printf '  Ferramentas:  %d imagens\n' \$(ls ~/casaiq_scraper/data/raw/images/ 2>/dev/null | wc -l)
printf '  Eletrônica:   %d imagens\n' \$(ls ~/casaiq_scraper/data/raw/electronics/ 2>/dev/null | wc -l)

echo
echo '--- DISCO USADO ---'
du -sh ~/casaiq_scraper/data/ 2>/dev/null

echo
echo '--- ÚLTIMAS ATIVIDADES ---'
for log in ~/casaiq_scraper/logs/*.log; do
    [ -f \"\$log\" ] && echo \"  \$(basename \$log): \$(tail -1 \$log 2>/dev/null | cut -c1-80)\"
done
"
