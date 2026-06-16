#!/usr/bin/env bash
# Resumo diário 07h — comparação com snapshot da véspera.
SNAP_DIR=/home/ubuntu/casaiq/orchestrator/snapshots
mkdir -p "$SNAP_DIR"

TODAY=$(date +%Y-%m-%d)
YESTERDAY=$(date -d 'yesterday' +%Y-%m-%d)
TODAY_FILE=$SNAP_DIR/snap_$TODAY.txt
YESTER_FILE=$SNAP_DIR/snap_$YESTERDAY.txt

# Capta números atuais (chaves: frente=valor)
{
  echo "ts=$(date '+%Y-%m-%d %H:%M:%S')"
  for s in superpro delupo ferimport screwfix kennedy minas; do
    if [ "$s" = screwfix ]; then path=/home/ubuntu/casaiq_scraper/data/export/screwfix.jsonl
    elif [ "$s" = kennedy ] || [ "$s" = minas ]; then path=/home/ubuntu/br_vtex_fedora_migrate/data/export/$s.jsonl
    else path=/home/ubuntu/br_vtex/data/export/$s.jsonl; fi
    n=$(wc -l < $path 2>/dev/null || echo 0)
    echo "$s=$n"
  done
  echo "ali=$(wc -l < /home/ubuntu/ali_local/data/export/aliexpress.jsonl 2>/dev/null || echo 0)"
  echo "matting=$(ls /home/ubuntu/casaiq/batches/01_matting/output/*.png 2>/dev/null | wc -l)"
  echo "dinov2=$(ls /home/ubuntu/casaiq/batches/02_dinov2/output 2>/dev/null | wc -l)"
} > "$TODAY_FILE"

# Monta mensagem com diff
msg="☀️ *CasaIQ — Bom dia $(date '+%Y-%m-%d')*"$'\n\n'

if [ -f "$YESTER_FILE" ]; then
  msg+="*Crescimento desde ontem:*"$'\n'
  while IFS='=' read -r key val_today; do
    [ "$key" = "ts" ] && continue
    val_yest=$(grep -E "^$key=" "$YESTER_FILE" 2>/dev/null | cut -d= -f2)
    val_yest=${val_yest:-0}
    diff=$((val_today - val_yest))
    sign=""
    [ "$diff" -gt 0 ] && sign="+"
    msg+="• $key: $val_today (${sign}${diff})"$'\n'
  done < "$TODAY_FILE"
else
  msg+="(primeiro snapshot, sem comparação)"$'\n\n*Valores atuais:*'$'\n'
  while IFS='=' read -r key val; do
    [ "$key" = "ts" ] && continue
    msg+="• $key: $val"$'\n'
  done < "$TODAY_FILE"
fi

# Soma total
total=0
for s in superpro delupo ferimport screwfix kennedy minas ali; do
  v=$(grep -E "^$s=" "$TODAY_FILE" | cut -d= -f2)
  total=$((total + ${v:-0}))
done
total=$((total + 23238))  # BR legados
msg+=$'\n*Total coletado:* '"$total"' itens'

# Health
if [ -f /home/ubuntu/casaiq/orchestrator/healthcheck.json ]; then
  dead=$(python3 -c "import json; print(','.join(json.load(open('/home/ubuntu/casaiq/orchestrator/healthcheck.json'))['dead']))" 2>/dev/null)
  [ -n "$dead" ] && msg+=$'\n\n⚠️ *Mortas:* '"$dead"
fi

python3 /home/ubuntu/casaiq/orchestrator/notify.py "$msg"
