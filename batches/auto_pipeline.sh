#!/usr/bin/env bash
# Auto-pipeline v3 — encadeia batches E notifica eventos críticos via WhatsApp.

set -e
ORCH_DIR=/home/ubuntu/casaiq/orchestrator
NOTIFY=$ORCH_DIR/notify.py
mkdir -p "$ORCH_DIR"
LOG=$ORCH_DIR/auto_pipeline.log
STATE=$ORCH_DIR/state.json
SLEEP_CHECK=300

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
notify() { "$NOTIFY" "$1" 2>/dev/null || true; }
write_state() { echo "{\"last_check\": \"$(date '+%Y-%m-%d %H:%M:%S')\", \"current_batch\": \"$1\", \"status\": \"$2\"}" > "$STATE"; }

batch_is_done() {
  local b=$1
  local sf=/home/ubuntu/casaiq/batches/$b/status.json
  [ -f "$sf" ] || return 1
  local d t
  d=$(python3 -c "import json; print(json.load(open('$sf'))['done'])" 2>/dev/null || echo 0)
  t=$(python3 -c "import json; print(json.load(open('$sf'))['total'])" 2>/dev/null || echo 1)
  [ "$d" -ge "$t" ] && [ "$t" -gt 0 ]
}

batch_is_running() {
  tmux list-windows -t main 2>/dev/null | grep -q "batch_$1"
}

launch_batch_window() {
  local dir=$1 short=$2 pre=$3
  log "lançando batch_${short}..."
  [ -n "$pre" ] && eval "$pre"
  tmux new-window -t main -n batch_${short} \
    "/home/ubuntu/miniconda3/bin/conda run -n deep_learning python -u /home/ubuntu/casaiq/batches/${dir}/run.py 2>&1 | tee /home/ubuntu/casaiq/batches/${dir}/run.log"
  notify "▶️ *CasaIQ: batch_${short} iniciou*"
}

wait_for_done() {
  local b=$1
  while true; do
    batch_is_done "$b" && return 0
    write_state "$b" "rodando"
    sleep $SLEEP_CHECK
  done
}

main() {
  log "===== auto-pipeline v3 iniciado ====="
  notify "🚀 *CasaIQ auto-pipeline ON*"$'\n'"Ordem: matting → dinov2 → faiss → sqlite_db"

  wait_for_done matting
  log "✅ batch_01_matting concluído"
  n=$(ls /home/ubuntu/casaiq/batches/01_matting/output/*.png 2>/dev/null | wc -l)
  notify "✅ *CasaIQ: matting concluído*"$'\n'"PNGs gerados: $n"

  if ! batch_is_running dinov2 && ! batch_is_done dinov2; then
    launch_batch_window 02_dinov2 dinov2 \
      'ls /home/ubuntu/casaiq/batches/01_matting/output/*.png > /home/ubuntu/casaiq/batches/02_dinov2/input_list.txt'
  fi
  wait_for_done dinov2
  log "✅ batch_02_dinov2 concluído"
  notify "✅ *CasaIQ: dinov2 encoding concluído*"$'\n'"Embeddings prontos."

  if ! batch_is_running faiss && ! batch_is_done faiss; then
    launch_batch_window 03_faiss faiss ''
  fi
  wait_for_done faiss
  log "✅ batch_03_faiss concluído"
  notify "✅ *CasaIQ: gazetteer.faiss pronto*"

  if ! batch_is_running sqlite_db && ! batch_is_done sqlite_db; then
    launch_batch_window 04_sqlite_db sqlite_db ''
  fi
  wait_for_done sqlite_db
  log "✅ batch_04_sqlite_db concluído"
  db=/home/ubuntu/casaiq/batches/04_sqlite_db/gazetteer.db
  size=$(du -h "$db" 2>/dev/null | cut -f1)
  notify "🎉 *CasaIQ PIPELINE COMPLETA*"$'\n'"gazetteer.db: $size"$'\n'"Pronto pra puxar pro Fedora."

  log "===== TUDO concluído ====="
  write_state "all" "done"
}

main
