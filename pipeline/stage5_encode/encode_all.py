"""Stage 5 — DINOv2 encode de TODAS as imagens novas (stage2).

Saída:
  pipeline/stage5_encode/embeddings_new.npy        — (N, 384) float32
  pipeline/stage5_encode/embeddings_index_new.jsonl — {"idx": i, "sha": sha256}

Restart-safe: retoma de onde parou.
"""
import json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

IMGS_DIR    = Path.home() / "dataset_imgs"
OUT_DIR     = Path(__file__).parent
OUT_NPY     = OUT_DIR / "embeddings_new.npy"
OUT_JSONL   = OUT_DIR / "embeddings_index_new.jsonl"
BATCH       = 64
REPORT_EACH = 500
MODEL       = "facebook/dinov2-small"


def main():
    t0 = time.time()
    print(f"[encode_all] carregando {MODEL}...", flush=True)
    proc  = AutoImageProcessor.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).eval()
    print(f"[encode_all] dim={model.config.hidden_size}  batch={BATCH}", flush=True)

    # restart-safe: carrega shas já feitos
    done, done_vecs = {}, []
    if OUT_JSONL.exists() and OUT_NPY.exists():
        old_emb = np.load(OUT_NPY)
        for line in open(OUT_JSONL):
            d = json.loads(line)
            done[d["sha"]] = len(done)
        if len(done) == old_emb.shape[0]:
            done_vecs = list(old_emb)
            print(f"[encode_all] retomando: {len(done)} já feitos", flush=True)
        else:
            done, done_vecs = {}, []   # inconsistente, refaz tudo
            print("[encode_all] jsonl/npy inconsistente — refazendo tudo", flush=True)

    all_imgs = sorted(IMGS_DIR.glob("*.jpg"))
    pending  = [p for p in all_imgs if p.stem not in done]
    print(f"[encode_all] total={len(all_imgs)}  pendentes={len(pending)}", flush=True)

    new_vecs, new_meta, errors = [], [], 0

    @torch.no_grad()
    def encode_batch(paths):
        imgs, valid = [], []
        for p in paths:
            try:
                imgs.append(Image.open(p).convert("RGB"))
                valid.append(p)
            except Exception:
                pass
        if not imgs:
            return None, []
        inp = proc(images=imgs, return_tensors="pt")
        out = model(**inp)
        cls = F.normalize(out.last_hidden_state[:, 0, :], dim=-1)
        return cls.numpy().astype("float32"), valid

    for i in range(0, len(pending), BATCH):
        batch = pending[i:i+BATCH]
        vecs, valid = encode_batch(batch)
        if vecs is None:
            errors += len(batch)
            continue
        errors += len(batch) - len(valid)
        for j, p in enumerate(valid):
            new_vecs.append(vecs[j])
            new_meta.append(p.stem)

        done_so_far = len(done_vecs) + len(new_vecs)
        if done_so_far % REPORT_EACH < BATCH:
            elapsed = time.time() - t0
            rate = done_so_far / elapsed if elapsed > 0 else 0
            rem  = (len(pending) - i - BATCH) / rate / 60 if rate > 0 else 0
            print(f"  {done_so_far}/{len(all_imgs)}  {rate:.1f} img/s  ETA {rem:.0f}min  erros={errors}", flush=True)

    # salva
    all_vecs = done_vecs + new_vecs
    all_meta = list(done.keys()) + new_meta
    mat = np.stack(all_vecs).astype("float32")
    np.save(OUT_NPY, mat)
    with open(OUT_JSONL, "w") as f:
        for i, sha in enumerate(all_meta):
            f.write(json.dumps({"idx": i, "sha": sha}) + "\n")

    elapsed = time.time() - t0
    print(f"\n{'='*50}", flush=True)
    print(f"DONE  shape={mat.shape}  erros={errors}  tempo={elapsed/60:.1f}min", flush=True)
    print(f"  → {OUT_NPY}", flush=True)
    print(f"  → {OUT_JSONL}", flush=True)


if __name__ == "__main__":
    main()
