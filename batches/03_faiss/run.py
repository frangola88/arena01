"""Batch 03 — index FAISS dos embeddings DINOv2.

Lê embeddings.npy + embeddings_index.jsonl do batch 02, constrói índice FAISS
IndexFlatIP (inner product, equivalente a cosine porque embeddings são L2-normalizados),
salva como gazetteer.faiss.

Trivialmente rápido: ~10 min pra 100k embeddings.
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import faiss

BATCH_DIR = Path(os.environ.get('CASAIQ_BATCH_DIR', str(Path.home() / 'casaiq/batches/03_faiss')))
DINOV2_DIR = Path(os.environ.get('CASAIQ_DINOV2_DIR', str(Path.home() / 'casaiq/batches/02_dinov2')))
BATCH_DIR.mkdir(parents=True, exist_ok=True)

def main():
    t0 = time.time()
    npy = DINOV2_DIR / 'embeddings.npy'
    idx_file = DINOV2_DIR / 'embeddings_index.jsonl'
    if not npy.exists():
        sys.exit(f'ERRO: {npy} não existe. Batch 02 precisa terminar antes.')

    print(f'[03_faiss] carregando {npy}')
    embeddings = np.load(npy)
    print(f'  shape={embeddings.shape} dtype={embeddings.dtype}')

    # IndexFlatIP é cosine pois embeddings já estão L2-normalizados pelo batch_02
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    print(f'  index ntotal={index.ntotal}')

    out_idx = BATCH_DIR / 'gazetteer.faiss'
    faiss.write_index(index, str(out_idx))
    print(f'  escrito {out_idx} ({out_idx.stat().st_size/1e6:.1f} MB)')

    # Copia o mapeamento idx→key pro mesmo dir
    if idx_file.exists():
        (BATCH_DIR / 'index_mapping.jsonl').write_text(idx_file.read_text())

    # Smoke test: top-1 do primeiro embedding deve ser ele mesmo (cosine = 1.0)
    q = embeddings[0:1]
    D, I = index.search(q.astype(np.float32), 5)
    print(f'  smoke: top-5 do item 0 → ids={I[0].tolist()} scores={D[0].round(3).tolist()}')

    # Status final
    status = {
        'batch': '03_faiss',
        'total': int(embeddings.shape[0]),
        'done': int(embeddings.shape[0]),
        'ok': int(embeddings.shape[0]),
        'err': 0,
        'rate_per_s': 0,
        'eta_min': 0,
        'elapsed_min': round((time.time() - t0) / 60, 2),
        'current': '',
        'updated': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    (BATCH_DIR / 'status.json').write_text(json.dumps(status, indent=2))
    print(f'[03_faiss] FIM em {time.time()-t0:.1f}s')

if __name__ == '__main__':
    main()
