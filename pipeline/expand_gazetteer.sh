#!/usr/bin/env bash
# expand_gazetteer.sh — encode imagens novas + merge no gazetteer.
#
# Uso: bash expand_gazetteer.sh
# Acompanhar:
#   cat  ~/projetos/casaiq/pipeline/expand_gazetteer.status
#   tail -f ~/projetos/casaiq/pipeline/expand_gazetteer.log
set -eo pipefail

BASE="$HOME/projetos/casaiq"
S5="$BASE/pipeline/stage5_encode"
ENVN="casaiq"
LOG="$BASE/pipeline/expand_gazetteer.log"
STATUS="$BASE/pipeline/expand_gazetteer.status"

ts(){ date '+%Y-%m-%d %H:%M:%S'; }
log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
st(){ echo "$(ts) | $1" > "$STATUS"; log "STATUS=$1"; }

log "================ expand_gazetteer START (pid $$) ================"

# 1) Verifica se há imagens novas para encodar
N_IMGS=$(ls "$HOME/dataset_imgs"/*.jpg 2>/dev/null | wc -l)
N_ENC=$(wc -l < "$BASE/embeddings_index.jsonl" 2>/dev/null || echo 0)
log "imgs no disco: $N_IMGS  já encodadas: $N_ENC"
if [ "$N_IMGS" -le "$N_ENC" ]; then
    log "Nada novo — gazetteer já atualizado. Saindo."
    st "CONCLUIDO_SEM_MUDANCA"
    exit 0
fi

# 2) Backup reversível do gazetteer atual
st "backup"
BK="$BASE/data/gazetteer/_archive/pre_expand_$(date +%Y%m%d_%H%M)"
mkdir -p "$BK"
for f in "$BASE/embeddings.npy" "$BASE/embeddings_index.jsonl"; do
    [ -f "$f" ] && cp -p "$f" "$BK/" && log "backup: $(basename "$f")"
done

# 3) Encode — DINOv2 das imagens novas (resume-safe)
st "encode_all"
conda run --no-capture-output -n "$ENVN" python "$S5/encode_all.py" >> "$LOG" 2>&1
log "encode_all OK"

# 4) Merge + dedup → gazetteer atualizado
st "merge_rebuild"
conda run --no-capture-output -n "$ENVN" python "$S5/merge_and_rebuild.py" >> "$LOG" 2>&1
log "merge_and_rebuild OK"

# 5) Verificação final
st "verificando"
conda run -n "$ENVN" python3 - >> "$LOG" 2>&1 <<'PY'
import numpy as np
from pathlib import Path
B = Path.home() / "projetos/casaiq"
emb = np.load(B / "embeddings.npy", mmap_mode="r")
n_map = sum(1 for _ in open(B / "embeddings_index.jsonl"))
print(f"[verif] embeddings={emb.shape}  index={n_map}  ok={emb.shape[0]==n_map}")
PY

st "CONCLUIDO"
log "================ expand_gazetteer DONE ================"
