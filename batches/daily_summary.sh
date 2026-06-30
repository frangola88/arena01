#!/usr/bin/env bash
# daily_summary.sh — resumo diário com diff desde ontem.
SNAP_DIR=/home/ubuntu/casaiq/orchestrator/snapshots
NOTIFY=/home/ubuntu/casaiq/orchestrator/notify.py
BATCHES=/home/ubuntu/casaiq/batches
mkdir -p "$SNAP_DIR"

TODAY=$(date +%Y-%m-%d)
YEST=$(date  -d 'yesterday' +%Y-%m-%d)

bat_int() {
    python3 -c "import json; d=json.load(open('$BATCHES/$1/status.json')); print(int(float(d.get('$2',0))))" 2>/dev/null || echo 0
}

matting_done=$(bat_int 01_matting done)
matting_tot=$(bat_int  01_matting total)
matting_eta=$(bat_int  01_matting eta_min)
dino_done=$(bat_int    02_dinov2  done)
dino_tot=$(bat_int     02_dinov2  total)
n_emb=$(wc -l < "$BATCHES/02_dinov2/embeddings_index.jsonl" 2>/dev/null || echo 0)

coleta=0
for f in \
    /home/ubuntu/br_vtex/data/export/superpro.jsonl \
    /home/ubuntu/br_vtex/data/export/delupo.jsonl \
    /home/ubuntu/br_vtex/data/export/ferimport.jsonl \
    /home/ubuntu/casaiq_scraper/data/export/screwfix.jsonl \
    /home/ubuntu/br_vtex_fedora_migrate/data/export/kennedy.jsonl \
    /home/ubuntu/br_vtex_fedora_migrate/data/export/minas.jsonl \
    /home/ubuntu/ali_local/data/export/aliexpress.jsonl; do
    [ -f "$f" ] && coleta=$(( coleta + $(wc -l < "$f") ))
done

echo "matting=$matting_done dino=$dino_done emb=$n_emb coleta=$coleta" > "$SNAP_DIR/snap_$TODAY.txt"

YSNAP=$SNAP_DIR/snap_$YEST.txt
diff_val() { v=$(grep -o "$1=[0-9]*" "$YSNAP" 2>/dev/null | cut -d= -f2); echo $(( ${2:-0} - ${v:-0} )); }
d_matt=$(diff_val matting "$matting_done")
d_dino=$(diff_val dino    "$dino_done")
d_emb=$(diff_val  emb     "$n_emb")
d_col=$(diff_val  coleta  "$coleta")

sign() { [ "$1" -gt 0 ] && echo "+$1" || echo "$1"; }

msg="☀️ *CasaIQ — $(date '+%d/%m %H:%M')*"$'\n\n'

if [ "$matting_tot" -gt 0 ] && [ "$matting_done" -ge "$matting_tot" ]; then
    msg+="✅ *Matting:* ${matting_done}/${matting_tot} concluído"$'\n'
elif [ "$matting_tot" -gt 0 ]; then
    pct=$(( matting_done * 100 / matting_tot ))
    msg+="⏳ *Matting:* ${matting_done}/${matting_tot} (${pct}%) $(sign $d_matt)/dia"$'\n'
    if [ "$matting_eta" -gt 0 ]; then
        eta_h=$(( matting_eta / 60 ))
        eta_m=$(( matting_eta % 60 ))
        msg+="   → ETA ~${eta_h}h${eta_m}min"$'\n'
    fi
else
    msg+="🎨 *Matting:* aguardando"$'\n'
fi

if [ "$dino_tot" -gt 0 ] && [ "$dino_done" -ge "$dino_tot" ]; then
    msg+="✅ *DINOv2:* ${dino_done}/${dino_tot} concluído"$'\n'
elif [ "$dino_tot" -gt 0 ]; then
    pct=$(( dino_done * 100 / dino_tot ))
    msg+="⏳ *DINOv2:* ${dino_done}/${dino_tot} (${pct}%) $(sign $d_dino)/dia"$'\n'
else
    msg+="🧬 *DINOv2:* aguardando matting"$'\n'
fi

msg+=$'\n'"📦 *Gazetteer Oracle:* ${n_emb} emb $(sign $d_emb)/dia"$'\n'
msg+="🛒 *Coleta:* ${coleta} itens $(sign $d_col)/dia"$'\n'

python3 "$NOTIFY" "$msg"
