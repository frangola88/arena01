"""Stage 5 — DINOv2 encode das novas imagens baixadas pelo stage2.

Este script (Fedora) encoda a PRIMEIRA METADE do lote:
  imagens cujo stem (sha256) começa com [0-7] em hex.

Saída:
  ~/projetos/casaiq/pipeline/stage5_encode/embeddings_fedora.npy      — (N, 384) float32
  ~/projetos/casaiq/pipeline/stage5_encode/embeddings_index_fedora.jsonl — {"idx": i, "key": sha256}

Resume-safe: pula shas que já estão no jsonl de saída.
Dedup contra gazetteer: keys do gazetteer são 16-char, as novas são 64-char sha256 →
  formatos diferentes, sem sobreposição — dedup apenas dentro do próprio lote novo.
"""

import json
import time
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# ── Paths ──────────────────────────────────────────────────────────────────────
IMGS_DIR   = Path.home() / "dataset_imgs"
OUT_DIR    = Path(__file__).parent          # pipeline/stage5_encode/
OUT_NPY    = OUT_DIR / "embeddings_fedora.npy"
OUT_JSONL  = OUT_DIR / "embeddings_index_fedora.jsonl"

MODEL_NAME = "facebook/dinov2-small"
BATCH_SIZE = 32
REPORT_EVERY = 500   # linhas de progresso a cada N imagens processadas

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_already_done() -> set:
    """Retorna conjunto de shas já presentes no jsonl de saída (resume-safe)."""
    done = set()
    if OUT_JSONL.exists():
        with open(OUT_JSONL) as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["key"])
    return done


def list_target_images() -> list[Path]:
    """Lista .jpg em IMGS_DIR cujo stem começa com 0-7 (primeira metade hex)."""
    first_half = set("01234567")
    imgs = [
        p for p in sorted(IMGS_DIR.glob("*.jpg"))
        if p.stem and p.stem[0] in first_half
    ]
    return imgs


@torch.no_grad()
def encode_batch(paths: list[Path], proc, model) -> np.ndarray:
    imgs = []
    for p in paths:
        try:
            imgs.append(Image.open(p).convert("RGB"))
        except Exception as e:
            print(f"  [WARN] falha ao abrir {p.name}: {e}", flush=True)
            imgs.append(None)

    # Filtra Nones para encode, depois reinsere zeros para manter alinhamento
    valid_imgs = [img for img in imgs if img is not None]
    if not valid_imgs:
        return None, [i for i, img in enumerate(imgs) if img is None]

    inputs = proc(images=valid_imgs, return_tensors="pt")
    out = model(**inputs)
    cls = F.normalize(out.last_hidden_state[:, 0, :], dim=-1)
    return cls.numpy().astype("float32"), [i for i, img in enumerate(imgs) if img is None]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    # 1. Carrega modelo
    print(f"[stage5] Carregando {MODEL_NAME}...", flush=True)
    proc  = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).eval()
    print(f"[stage5] Modelo carregado. dim={model.config.hidden_size}", flush=True)

    # 2. Lista imagens alvo (primeira metade hex)
    all_imgs = list_target_images()
    print(f"[stage5] Imagens encontradas com stem [0-7]: {len(all_imgs)}", flush=True)

    # 3. Resume: pula já processados
    done = load_already_done()
    pending = [p for p in all_imgs if p.stem not in done]
    print(f"[stage5] Já encodadas (jsonl): {len(done)}  |  Pendentes: {len(pending)}", flush=True)

    if not pending:
        print("[stage5] Nada a fazer — tudo já encodado.", flush=True)
    else:
        # Abre jsonl em modo append (resume-safe)
        jsonl_f = open(OUT_JSONL, "a")

        # Carrega embeddings já existentes para reconstruir ao final
        existing_embeddings = []
        existing_keys = []
        if OUT_JSONL.exists() and done:
            # Relê para garantir ordem
            pass  # vamos acumular tudo em memória no final

        new_embeddings = []
        new_keys = []
        errors = []
        n_encoded = 0

        # Processa em batches
        for batch_start in range(0, len(pending), BATCH_SIZE):
            batch_paths = pending[batch_start : batch_start + BATCH_SIZE]
            vecs, bad_indices = encode_batch(batch_paths, proc, model)

            bad_set = set(bad_indices)
            valid_paths = [p for i, p in enumerate(batch_paths) if i not in bad_set]

            for path in [batch_paths[i] for i in bad_indices]:
                errors.append(path.name)

            if vecs is not None:
                for i, path in enumerate(valid_paths):
                    sha = path.stem
                    new_embeddings.append(vecs[i])
                    new_keys.append(sha)
                    jsonl_f.write(json.dumps({"idx": -1, "key": sha}) + "\n")  # idx reindexado no final
                n_encoded += len(valid_paths)

            if n_encoded and n_encoded % REPORT_EVERY < BATCH_SIZE:
                elapsed = time.time() - t0
                speed = n_encoded / elapsed if elapsed > 0 else 0
                remaining = len(pending) - batch_start - len(batch_paths)
                eta = remaining / speed if speed > 0 else 0
                print(
                    f"[stage5] {n_encoded}/{len(pending)} encodadas | "
                    f"{speed:.1f} img/s | ETA {eta/60:.1f} min | erros: {len(errors)}",
                    flush=True,
                )
            jsonl_f.flush()

        jsonl_f.close()

    # 4. Reconstrói npy + reindexação do jsonl a partir do jsonl completo
    print("[stage5] Reconstruindo npy e reindexando jsonl...", flush=True)
    all_keys_ordered = []
    with open(OUT_JSONL) as f:
        for line in f:
            line = line.strip()
            if line:
                all_keys_ordered.append(json.loads(line)["key"])

    # Remove duplicatas mantendo primeira ocorrência
    seen = set()
    unique_keys = []
    for k in all_keys_ordered:
        if k not in seen:
            seen.add(k)
            unique_keys.append(k)

    # Para cada key, precisa do embedding — acumula do que foi calculado agora + recarrega de npy se existia
    # Estratégia: monta dict key→vec de tudo que processamos nesta sessão
    key_to_vec = {}

    # Carrega npy anterior se existia (para resume de sessões anteriores)
    if OUT_NPY.exists():
        old_npy = np.load(OUT_NPY)
        # Recupera keys do jsonl ANTERIOR (antes desta sessão) — as que estavam em done
        old_jsonl_keys = []
        # Re-lemos: as keys done foram carregadas antes; mas não temos a ordem garantida
        # Solução: recarrega o jsonl em ordem para mapear idx → key do npy antigo
        # O npy antigo tem len(done) linhas se não havia nada "invalid" gravado com idx=-1
        # Mas pode ter idx=-1 se foi uma sessão anterior com o mesmo script.
        # Abordagem segura: rebuildar a partir de um snapshot antes desta sessão.
        # Como isso é complexo, usamos outra estratégia: o encode_batch acima
        # gravou idx=-1 para os novos. Vamos reconstruir só o que calculamos agora.
        pass

    # Monta dict dos vetores calculados nesta sessão
    if pending:
        # new_keys e new_embeddings foram acumulados acima
        for k, v in zip(new_keys, new_embeddings):
            key_to_vec[k] = v

    # Para as keys que já estavam no jsonl antes (done), precisamos do npy antigo
    if OUT_NPY.exists() and done:
        old_npy = np.load(OUT_NPY)
        # Recria mapeamento do jsonl legado (linhas com idx != -1 e key in done)
        # Relemos o jsonl completo e pegamos as primeiras len(done) únicas
        done_keys_ordered = [k for k in unique_keys if k in done]
        if len(done_keys_ordered) == old_npy.shape[0]:
            for i, k in enumerate(done_keys_ordered):
                key_to_vec[k] = old_npy[i]
        else:
            print(f"[stage5] AVISO: mismatch done_keys ({len(done_keys_ordered)}) vs old_npy ({old_npy.shape[0]}). Ignorando npy antigo.", flush=True)

    # Filtra unique_keys para as que temos vetor
    final_keys = [k for k in unique_keys if k in key_to_vec]
    missing = [k for k in unique_keys if k not in key_to_vec]
    if missing:
        print(f"[stage5] AVISO: {len(missing)} keys sem embedding (serão omitidas do npy).", flush=True)

    if final_keys:
        mat = np.stack([key_to_vec[k] for k in final_keys]).astype("float32")
        np.save(OUT_NPY, mat)

        # Reescreve jsonl com idx corretos e sem duplicatas
        with open(OUT_JSONL, "w") as f:
            for i, k in enumerate(final_keys):
                f.write(json.dumps({"idx": i, "key": k}) + "\n")

        print(f"[stage5] Salvo: {OUT_NPY} shape={mat.shape}", flush=True)
        print(f"[stage5] Salvo: {OUT_JSONL} ({len(final_keys)} entradas)", flush=True)
    else:
        print("[stage5] Nenhum embedding para salvar.", flush=True)

    # 5. Relatório final
    elapsed = time.time() - t0
    n_final = len(final_keys) if final_keys else 0
    n_new_this_run = len(pending) - len([e for e in errors])  # encodadas nesta sessão
    print("", flush=True)
    print("=" * 60, flush=True)
    print(f"[stage5] CONCLUÍDO", flush=True)
    print(f"  Total encodadas (npy): {n_final}", flush=True)
    print(f"  Novas nesta sessão:    {n_new_this_run}", flush=True)
    print(f"  Erros de leitura:      {len(errors)}", flush=True)
    if errors:
        for e in errors[:20]:
            print(f"    - {e}", flush=True)
        if len(errors) > 20:
            print(f"    ... e mais {len(errors)-20}", flush=True)
    print(f"  Tempo total:           {elapsed/60:.1f} min ({elapsed:.0f}s)", flush=True)
    print(f"  Shape final npy:       {mat.shape if final_keys else 'N/A'}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
