#!/usr/bin/env bash
# Painel de controle único — rodar de qualquer máquina pra ver estado completo.
# Uso: painel.sh        (do Fedora ou direto na Oracle)

HOST=${CASAIQ_HOST:-oracle-casaiq}

ssh "$HOST" 'bash -s' <<'REMOTE'
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  CasaIQ — Painel de Controle Oracle ($(date '+%Y-%m-%d %H:%M:%S'))  ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"

echo
echo "▶ tmux windows ativas:"
tmux list-windows -t main 2>/dev/null | awk '{print "  " $0}'

echo
echo "▶ Coleta atual (itens reais nos jsonls):"
printf '  %-25s %s\n' 'Frente' 'Itens'
printf '  %-25s %s\n' '------' '-----'
for s in superpro delupo ferimport screwfix; do
  if [ "$s" = screwfix ]; then path=~/casaiq_scraper/data/export/screwfix.jsonl
  else path=~/br_vtex/data/export/$s.jsonl; fi
  [ -f "$path" ] && printf '  %-25s %s\n' "$s (Oracle)" "$(wc -l < $path)"
done
for s in kennedy minas; do
  path=~/br_vtex_fedora_migrate/data/export/$s.jsonl
  [ -f "$path" ] && printf '  %-25s %s\n' "$s (ex-Fedora)" "$(wc -l < $path)"
done
n=$(wc -l < ~/ali_local/data/export/aliexpress.jsonl 2>/dev/null || echo 0)
nkw=$(wc -l < ~/ali_local/data/aliexpress_keywords_done.txt 2>/dev/null || echo 0)
printf '  %-25s %s (%s kw)\n' "ali (Oracle)" "$n" "$nkw"

echo
echo "▶ Coleta BR legada (completa):"
for f in eletrogate filipeflop palacio_ferramentas baudaeletronica; do
  n=$(wc -l < ~/casaiq_scraper/data/export/$f.jsonl 2>/dev/null || echo 0)
  printf '  %-25s %s\n' "$f" "$n"
done

echo
echo "▶ Batches:"
for batch_dir in ~/casaiq/batches/*/; do
  [ -d "$batch_dir" ] || continue
  name=$(basename "$batch_dir")
  status=$batch_dir/status.json
  if [ -f "$status" ]; then
    done=$(python3 -c "import json; d=json.load(open('$status')); print(f\"{d.get('done',0)}/{d.get('total',0)} ok={d.get('ok',0)} err={d.get('err',0)} rate={d.get('rate_per_s',0)}/s ETA={d.get('eta_min',0)}min\")" 2>/dev/null)
    echo "  $name: $done"
  else
    # contagem direta de output
    n=$(ls "${batch_dir}output" 2>/dev/null | wc -l)
    echo "  $name: $n arquivos no output"
  fi
done

echo
echo "▶ Orquestrador / Watchdog:"
[ -f ~/casaiq/orchestrator/state.json ] && cat ~/casaiq/orchestrator/state.json | sed 's/^/  /'
[ -f ~/casaiq/orchestrator/healthcheck.json ] && {
  echo
  echo "  healthcheck:"
  cat ~/casaiq/orchestrator/healthcheck.json | sed 's/^/    /'
}

echo
echo "▶ Recursos:"
mem=$(free -m | awk 'NR==2 {printf "%.1f/%.1f GB", $3/1024, $2/1024}')
load=$(uptime | grep -oE 'load average.*' | head -1)
disk=$(df -h ~ | tail -1 | awk '{print $3 " usados / " $2 " total (" $5 ")"}')
echo "  RAM:  $mem"
echo "  Load: $load"
echo "  Disco home: $disk"
REMOTE
