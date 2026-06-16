#!/usr/bin/env bash
# Watchdog v2 — escreve healthcheck E dispara alerta WhatsApp em NOVAS mortes.

HEALTH=/home/ubuntu/casaiq/orchestrator/healthcheck.json
LAST=/home/ubuntu/casaiq/orchestrator/last_dead_alert.txt
NOTIFY=/home/ubuntu/casaiq/orchestrator/notify.py
mkdir -p $(dirname "$HEALTH")
NOW=$(date '+%Y-%m-%d %H:%M:%S')

EXPECTED="ali_loop|screwfix_extract|vtex_superpro|vtex_delupo|vtex_ferimport|vtex_kennedy|vtex_minas|batch_matting|auto_pipeline"

alive=()
dead=()
for w in $(echo "$EXPECTED" | tr '|' ' '); do
  if tmux list-windows -t main 2>/dev/null | awk '{print $2}' | tr -d ':*-' | grep -qw "$w"; then
    alive+=("$w")
  else
    dead+=("$w")
  fi
done

python3 - <<PY > "$HEALTH"
import json
print(json.dumps({
    'updated': '$NOW',
    'alive': "${alive[*]}".split(),
    'dead':  "${dead[*]}".split(),
    'n_alive': len("${alive[*]}".split()),
    'n_dead':  len("${dead[*]}".split()),
}, indent=2))
PY

# Compara com último alerta — só notifica NOVAS mortes (evita spam)
current_dead=$(echo "${dead[*]}" | tr ' ' '\n' | sort | tr '\n' ',' | sed 's/,$//')
last_dead=$(cat "$LAST" 2>/dev/null || echo '')

if [ "$current_dead" != "$last_dead" ] && [ -n "$current_dead" ]; then
  # Há mortes diferentes do alerta anterior — pode ser NOVA morte
  # Detecta o que está em current mas não em last
  new_dead=$(comm -23 <(echo "$current_dead" | tr ',' '\n' | sort) <(echo "$last_dead" | tr ',' '\n' | sort) 2>/dev/null | grep -v '^$' | tr '\n' ',' | sed 's/,$//')
  if [ -n "$new_dead" ]; then
    "$NOTIFY" "⚠️ *CasaIQ alerta*\nWindows mortas: $new_dead\nHora: $NOW" 2>/dev/null
  fi
  echo "$current_dead" > "$LAST"
elif [ -z "$current_dead" ] && [ -n "$last_dead" ]; then
  # Tudo voltou — também alerta
  "$NOTIFY" "✅ *CasaIQ — todas as windows OK*\nHora: $NOW" 2>/dev/null
  echo "" > "$LAST"
fi
