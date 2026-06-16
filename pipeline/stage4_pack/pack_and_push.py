"""Etapa 4 — empacotamento final + push HF Hub.

Junta dataset_v0 + pass1 + pass2 + img_index num único dataset HF.
Filtra util_casaiq=true. Split estratificado por categoria.
Push privado pro HuggingFace Hub.

Uso:
    python pack_and_push.py build                   # gera dataset local
    python pack_and_push.py push --repo cbpsoares/casaiq-tools-v1
    python pack_and_push.py stats
"""
import argparse, json, sys
from pathlib import Path
import pandas as pd

PIPE = Path.home() / 'projetos' / 'casaiq' / 'pipeline'
PARQUET_V0 = PIPE / 'stage1_consolidate' / 'dataset_v0.parquet'
PASS1 = PIPE / 'stage3_enrich' / 'pass1_classified.jsonl'
PASS2 = PIPE / 'stage3_enrich' / 'pass2_normalized.jsonl'
IMG_INDEX = PIPE / 'stage2_imgs' / 'img_index.jsonl'
DATASET_DIR = PIPE / 'stage4_pack' / 'dataset_v1'
TRAIN_RATIO, VAL_RATIO = 0.90, 0.05  # 90/5/5

def load_jsonl(p: Path) -> pd.DataFrame:
    if not p.exists(): return pd.DataFrame()
    return pd.read_json(p, lines=True)

def build():
    df = pd.read_parquet(PARQUET_V0)
    df['_id'] = df.index.astype(str)
    p1, p2, idx = load_jsonl(PASS1), load_jsonl(PASS2), load_jsonl(IMG_INDEX)
    print(f'[4] v0={len(df):,} pass1={len(p1):,} pass2={len(p2):,} imgs={len(idx):,}')

    # merge step-by-step
    df = df.merge(p1, on='_id', how='left')
    if not p2.empty:
        df = df.merge(p2, on='_id', how='left')
    if not idx.empty:
        df = df.merge(idx[['_id', 'img_path']], on='_id', how='left')

    # filtros essenciais
    df = df[df['util_casaiq'].fillna(False)]
    df = df[df['img_path'].notna() & df['img_path'].str.len().gt(5)]
    print(f'[4] após filtro util+img: {len(df):,}')

    # coluna nome final (prefere nome_limpo, fallback nome_bruto)
    df['nome_final'] = df['nome_limpo'].fillna(df['nome_bruto'])
    df['brand_final'] = df['brand_canonical'].fillna(df['brand'])

    # split estratificado por categoria_inferida
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    splits = []
    for cat, grp in df.groupby('categoria_inferida'):
        n = len(grp)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        grp = grp.copy()
        grp['_split'] = 'test'
        grp.iloc[:n_train, grp.columns.get_loc('_split')] = 'train'
        grp.iloc[n_train:n_train + n_val, grp.columns.get_loc('_split')] = 'val'
        splits.append(grp)
    df = pd.concat(splits, ignore_index=True)
    print(f'[4] splits: {df["_split"].value_counts().to_dict()}')

    # colunas finais
    out_cols = ['_id', 'nome_final', 'brand_final', 'categoria_inferida',
                'fonte', 'idioma', 'dominio_tipo', 'img_path', 'img_url',
                'product_url', '_split']
    out = df[[c for c in out_cols if c in df.columns]]

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    for split in ['train', 'val', 'test']:
        sub = out[out['_split'] == split]
        path = DATASET_DIR / f'{split}.parquet'
        sub.to_parquet(path, index=False)
        print(f'  {split:>5}: {len(sub):,} → {path}')

    # README mínimo
    readme = f"""---
license: apache-2.0
language: [pt, en]
size_categories: [10K<n<100K]
task_categories: [image-classification, image-to-text]
---

# CasaIQ Tools Dataset v1

Dataset privado de fotos de ferramentas/equipamentos pra fine-tune Qwen2-VL no agente
de visão do CasaIQ (inventário doméstico inteligente).

## Stats
- Total: {len(out):,} itens com imagem válida
- Train/Val/Test: {out['_split'].value_counts().to_dict()}

## Fontes
{out['fonte'].value_counts().to_string()}

## Categorias
{out['categoria_inferida'].value_counts().to_string()}
"""
    (DATASET_DIR / 'README.md').write_text(readme)
    print(f'\n[4] dataset_v1 pronto em {DATASET_DIR}')

def push(repo: str):
    """Faz upload via huggingface_hub CLI (precisa `huggingface-cli login` antes)."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("pip install huggingface_hub")
    if not DATASET_DIR.exists():
        sys.exit('rode `pack_and_push.py build` antes')
    api = HfApi()
    print(f'[push] criando/atualizando dataset {repo} (privado)')
    api.create_repo(repo_id=repo, repo_type='dataset', private=True, exist_ok=True)
    api.upload_folder(folder_path=str(DATASET_DIR), repo_id=repo,
                      repo_type='dataset', commit_message='Initial v1 push')
    print('[push] OK')

def stats():
    if DATASET_DIR.exists():
        for split in ['train', 'val', 'test']:
            p = DATASET_DIR / f'{split}.parquet'
            if p.exists():
                df = pd.read_parquet(p)
                print(f'{split:>5}: {len(df):,}')
                if split == 'train':
                    print(f'  top categorias: {df["categoria_inferida"].value_counts().head(5).to_dict()}')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['build', 'push', 'stats'])
    ap.add_argument('--repo', default='cbpsoares/casaiq-tools-v1')
    args = ap.parse_args()
    if args.mode == 'build': build()
    elif args.mode == 'push': push(args.repo)
    elif args.mode == 'stats': stats()

if __name__ == '__main__':
    main()
