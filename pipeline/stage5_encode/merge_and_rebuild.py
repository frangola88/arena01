"""
Merge embeddings novos (stage2) com gazetteer existente e reconstrói índices.
Uso: conda run -n casaiq python pipeline/stage5_encode/merge_and_rebuild.py
"""
import json, subprocess, sys, time
from pathlib import Path
import faiss
import numpy as np

BASE    = Path.home() / "projetos/casaiq"
GAZ     = BASE / "data/gazetteer"
STAGE5  = BASE / "pipeline/stage5_encode"

EMB_OLD  = BASE / "embeddings.npy"
MAP_OLD  = GAZ  / "index_mapping.jsonl"
EMB_NEW  = STAGE5 / "embeddings_new.npy"
IDX_NEW  = STAGE5 / "embeddings_index_new.jsonl"
OUT_FAISS = GAZ / "gazetteer.faiss"
OUT_MAP   = GAZ / "index_mapping.jsonl"
OUT_EMB   = BASE / "embeddings.npy"

def wait_for_encode(interval=60, max_wait=7200):
    deadline = time.time() + max_wait
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        ok = EMB_NEW.exists() and EMB_NEW.stat().st_size > 0
        size = EMB_NEW.stat().st_size if EMB_NEW.exists() else 0
        print(f"[{attempt}] embeddings_new.npy: {'PRONTO' if ok else 'aguardando'} ({size/1e6:.1f} MB)", flush=True)
        if ok: return True
        time.sleep(interval)
    return False

def main():
    print("=== merge_and_rebuild ===", flush=True)
    if not (EMB_NEW.exists() and EMB_NEW.stat().st_size > 0):
        print("Aguardando encode_all.py...", flush=True)
        if not wait_for_encode(): sys.exit(1)

    print("[1] Carregando antigos...", flush=True)
    emb_old = np.load(EMB_OLD).astype("float32")
    map_old = [json.loads(l) for l in open(MAP_OLD)]
    print(f"    {emb_old.shape}  map={len(map_old)}", flush=True)

    print("[2] Carregando novos...", flush=True)
    emb_new = np.load(EMB_NEW).astype("float32")
    idx_new = [json.loads(l) for l in open(IDX_NEW)]
    print(f"    {emb_new.shape}  map={len(idx_new)}", flush=True)

    emb_all = np.concatenate([emb_old, emb_new], axis=0)
    print(f"[3] Total: {emb_all.shape}", flush=True)

    print("[4] index_mapping...", flush=True)
    new_map = [{"idx": i, "key": e["key"]} for i, e in enumerate(map_old)]
    offset  = len(map_old)
    new_map += [{"idx": offset+i, "key": e["sha"]} for i, e in enumerate(idx_new)]

    print("[5] Salvando embeddings...", flush=True)
    np.save(OUT_EMB, emb_all)

    print("[6] Rebuild FAISS...", flush=True)
    idx = faiss.IndexFlatIP(emb_all.shape[1])
    idx.add(emb_all)
    faiss.write_index(idx, str(OUT_FAISS))
    print(f"    {idx.ntotal} vetores", flush=True)

    print("[7] Salvando index_mapping...", flush=True)
    with open(OUT_MAP, "w") as f:
        for e in new_map: f.write(json.dumps(e) + "\n")

    print("[8] Rebuild text_index...", flush=True)
    r = subprocess.run(
        ["conda", "run", "-n", "ai-dl-rl",
         "python", str(BASE / "pipeline/stage4_pack/build_text_index.py")],
        capture_output=True, text=True, cwd=str(BASE))
    print("    ok" if r.returncode == 0 else f"    AVISO: {r.stderr[-200:]}", flush=True)

    print("[9] Smoke test...", flush=True)
    D, I = idx.search(emb_all[:1], 3)
    print(f"    top3 dists: {D[0]}", flush=True)
    assert D[0][0] > 0.999

    print(f"\nDONE  total={emb_all.shape[0]}  antigos={len(map_old)}  novos={len(idx_new)}", flush=True)

if __name__ == "__main__":
    main()
