#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# expand_gazetteer.sh — expansão automática do gazetteer após o download stage2.
#
# Cadeia: (espera o download terminar) → backup → encode_all → merge_and_rebuild.
# Dispara sozinho: detecta o fim do download pela ausência do processo
# download_imgs.py. Idempotente/restart-safe: cada etapa Python retoma de onde
# parou e o merge deduplica shas já presentes no gazetteer.
#
# Acompanhar:
#   cat  ~/projetos/casaiq/pipeline/expand_gazetteer.status
#   tail -f ~/projetos/casaiq/pipeline/expand_gazetteer.log
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

BASE="$HOME/projetos/casaiq"
S5="$BASE/pipeline/stage5_encode"
ENVN="ai-dl-rl"
LOG="$BASE/pipeline/expand_gazetteer.log"
STATUS="$BASE/pipeline/expand_gazetteer.status"
DL_PATTERN="stage2_imgs/download_imgs.py"

ts(){ date '+%Y-%m-%d %H:%M:%S'; }
log(){ echo "[$(ts)] $*" >> "$LOG"; }
st(){ echo "$(ts) | $1" > "$STATUS"; log "STATUS=$1"; }

log "================ expand_gazetteer START (pid $$) ================"

# 1) Espera o download do stage2 terminar (processo download_imgs.py some)
st "aguardando_download"
sleep 5   # garante que o processo do download já existe antes de checar
while pgrep -f "$DL_PATTERN" >/dev/null 2>&1; do
    sleep 120
done
log "download finalizado (processo ausente)"
dl=$(ls -t "$BASE"/pipeline/stage2_imgs/stage2_fedora_*.log 2>/dev/null | head -1)
[ -n "${dl:-}" ] && log "ultima linha download: $(tail -1 "$dl" 2>/dev/null)"
log "imgs no disco: $(ls "$HOME/dataset_imgs"/*.jpg 2>/dev/null | wc -l)"

# 2) Backup reversível do gazetteer atual (antes de qualquer rebuild)
st "backup"
BK="$BASE/data/gazetteer/_archive/pre_expand_$(date +%Y%m%d_%H%M)"
mkdir -p "$BK"
for f in "$BASE/embeddings.npy" \
         "$BASE/data/gazetteer/index_mapping.jsonl" \
         "$BASE/data/gazetteer/gazetteer.faiss" \
         "$BASE/data/gazetteer/text_index.json"; do
    cp -p "$f" "$BK/" 2>/dev/null && log "backup: $(basename "$f")"
done
log "backup em $BK"

# 3) encode_all — DINOv2 de TODAS as dataset_imgs (resume os já feitos)
st "encode_all"
conda run --no-capture-output -n "$ENVN" python "$S5/encode_all.py" >> "$LOG" 2>&1
rc=$?
if [ $rc -ne 0 ]; then st "ERRO_encode_all(rc=$rc)"; log "abortado no encode_all"; exit 1; fi
log "encode_all OK"

# 4) merge_and_rebuild — 47k + novos (dedup) → FAISS + map + embeddings + text_index
st "merge_rebuild"
conda run --no-capture-output -n "$ENVN" python "$S5/merge_and_rebuild.py" >> "$LOG" 2>&1
rc=$?
if [ $rc -ne 0 ]; then st "ERRO_merge(rc=$rc)"; log "abortado no merge"; exit 1; fi
log "merge_and_rebuild OK"

# 5) Verificação final de consistência
st "verificando"
conda run -n "$ENVN" python - <<'PY' >> "$LOG" 2>&1
import numpy as np
from pathlib import Path
B = Path.home() / "projetos/casaiq"
emb = np.load(B / "embeddings.npy", mmap_mode="r")
n_map = sum(1 for _ in open(B / "data/gazetteer/index_mapping.jsonl"))
ok = emb.shape[0] == n_map
print(f"[verif] embeddings={emb.shape}  index_mapping={n_map}  consistente={ok}")
PY

st "CONCLUIDO"
log "================ expand_gazetteer DONE ================"
