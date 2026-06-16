#!/usr/bin/env bash
# Sincronização Fedora ↔ Oracle para trabalho em 2 frentes.
#
# Layouts espelhados:
#   Fedora: ~/projetos/casaiq/{pipeline,poc_dinov2}/
#   Oracle: ~/casaiq/{pipeline,poc_dinov2}/
#
# Uso:
#   sync_oracle.sh push-code        # Fedora → Oracle (código)
#   sync_oracle.sh pull-data        # Oracle → Fedora (jsonl de coleta)
#   sync_oracle.sh push-data        # Fedora → Oracle (jsonl + parquet do Fedora)
#   sync_oracle.sh pull-imgs        # Oracle → Fedora (imagens baixadas pela Oracle)
#   sync_oracle.sh pull-results     # Oracle → Fedora (output do POC ou pipeline rodando lá)
#   sync_oracle.sh status           # mostra discrepâncias entre as duas
#
# Excludes padrão: __pycache__, *.pyc, .git, .DS_Store

set -e
HOST=oracle-casaiq
FEDORA=$HOME/projetos/casaiq
ORACLE=casaiq
RSYNC_OPTS="-avz --exclude=__pycache__ --exclude=*.pyc --exclude=.DS_Store --exclude=.git"

cmd=${1:-status}

case $cmd in
push-code)
  echo "[push-code] Fedora → Oracle (pipeline + poc_dinov2 + scripts)"
  for sub in pipeline poc_dinov2 scripts; do
    [ -d "$FEDORA/$sub" ] || continue
    echo "  → $sub/"
    rsync $RSYNC_OPTS --exclude='data/' --exclude='canonical/' --exclude='output/' \
      "$FEDORA/$sub/" "$HOST:$ORACLE/$sub/"
  done
  ;;

pull-data)
  echo "[pull-data] Oracle → Fedora (jsonl de coleta)"
  mkdir -p "$FEDORA/data/oracle_pull"
  for path in \
      "/home/ubuntu/br_vtex/data/export/" \
      "/home/ubuntu/casaiq_scraper/data/export/" \
      "/home/ubuntu/ali_local/data/export/"; do
    name=$(basename $(dirname $(dirname "$path")))
    dest="$FEDORA/data/oracle_pull/$name/"
    mkdir -p "$dest"
    echo "  ← $path"
    rsync $RSYNC_OPTS --include='*.jsonl' --include='*/' --exclude='*' \
      "$HOST:$path" "$dest"
  done
  ;;

push-data)
  echo "[push-data] Fedora → Oracle (jsonl + parquet local)"
  rsync $RSYNC_OPTS \
    "$FEDORA/data/br_vtex/data/export/" "$HOST:$ORACLE/data/br_vtex/data/export/"
  if [ -f "$FEDORA/pipeline/stage1_consolidate/dataset_v0.parquet" ]; then
    rsync $RSYNC_OPTS "$FEDORA/pipeline/stage1_consolidate/dataset_v0.parquet" \
      "$HOST:$ORACLE/pipeline/stage1_consolidate/"
  fi
  ;;

pull-imgs)
  echo "[pull-imgs] Oracle → Fedora (imagens baixadas na Oracle)"
  mkdir -p "$HOME/dataset_imgs"
  rsync $RSYNC_OPTS --include='*.jpg' --include='*/' --exclude='*' \
    "$HOST:dataset_imgs/" "$HOME/dataset_imgs/"
  ;;

pull-results)
  echo "[pull-results] Oracle → Fedora (output de POC/pipeline)"
  for sub in poc_dinov2/output pipeline/stage1_consolidate pipeline/stage2_imgs pipeline/stage3_enrich pipeline/stage4_pack; do
    src="$HOST:$ORACLE/$sub/"
    dest="$FEDORA/${sub}_from_oracle/"
    echo "  ← $sub"
    rsync $RSYNC_OPTS "$src" "$dest" 2>/dev/null || true
  done
  ;;

status)
  echo "=== status Fedora ==="
  for sub in pipeline poc_dinov2 data/br_vtex/data/export data/ali_local/data/export; do
    [ -d "$FEDORA/$sub" ] && echo "  Fedora $sub: $(find "$FEDORA/$sub" -type f 2>/dev/null | wc -l) arquivos"
  done
  echo
  echo "=== status Oracle ==="
  ssh $HOST "for sub in casaiq/pipeline casaiq/poc_dinov2 br_vtex/data/export casaiq_scraper/data/export ali_local/data/export dataset_imgs; do
    [ -d ~/\$sub ] && echo \"  Oracle \$sub: \$(find ~/\$sub -type f 2>/dev/null | wc -l) arquivos\"
  done"
  ;;

*)
  echo "uso: sync_oracle.sh {push-code|pull-data|push-data|pull-imgs|pull-results|status}"
  exit 1
  ;;
esac

echo "[done]"
