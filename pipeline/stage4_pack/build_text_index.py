"""Constrói text_index.json para lookup texto→embedding no GazetteerMatcher.

Fonte de verdade: dataset_v0.parquet (98k itens; 18k têm img_local preenchido).
  key = sha256(img_local)[:16]  →  faiss_idx  (via index_mapping.jsonl)
  slug = slug_norm(nome_bruto + " " + brand)  e  slug_norm(nome_bruto) sozinho

Saída: ~/projetos/casaiq/data/gazetteer/text_index.json
Formato: {"slug": [faiss_idx, ...], ...}

Uso:
    python build_text_index.py
    python build_text_index.py --stats
    python build_text_index.py --gazetteer-dir /caminho/alternativo
"""
import argparse, hashlib, json, re, sys, unicodedata
from pathlib import Path
from collections import defaultdict

try:
    import pandas as pd
except ImportError:
    sys.exit("pip install pandas pyarrow")

PIPE    = Path.home() / "projetos/casaiq/pipeline"
GAZ_DIR = Path.home() / "projetos/casaiq/data/gazetteer"

PARQUET  = PIPE / "stage1_consolidate/dataset_v0.parquet"
MAP_FILE = GAZ_DIR / "index_mapping.jsonl"

TEXT_INDEX = GAZ_DIR / "text_index.json"


def slug_norm(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-zA-Z0-9 ]", " ", s).lower()
    return re.sub(r"\s+", " ", s).strip()


def img_key(img_local: str) -> str:
    """Replica a lógica do batch_01: sha256(img_local)[:16]."""
    return hashlib.sha256(img_local.encode()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true", help="Só mostra stats do índice existente")
    ap.add_argument("--gazetteer-dir", default=str(GAZ_DIR))
    args = ap.parse_args()

    gaz_dir = Path(args.gazetteer_dir)

    if args.stats:
        idx_path = gaz_dir / "text_index.json"
        if not idx_path.exists():
            sys.exit("text_index.json não encontrado")
        idx = json.loads(idx_path.read_text())
        print(f"slugs únicos:  {len(idx):,}")
        total_hits = sum(len(v) for v in idx.values())
        print(f"total hits:    {total_hits:,}")
        print(f"hits/slug:     {total_hits/len(idx):.1f}")
        print("\nTop 10 slugs com mais hits:")
        for slug, idxs in sorted(idx.items(), key=lambda x: -len(x[1]))[:10]:
            print(f"  {slug[:60]:60s} {len(idxs)} hits")
        return

    # Verifica arquivos necessários
    for p, label in [(PARQUET, "dataset_v0.parquet"),
                     (gaz_dir / "index_mapping.jsonl", "index_mapping.jsonl")]:
        if not p.exists():
            sys.exit(f"ERRO: {label} não encontrado em {p}")

    map_file = gaz_dir / "index_mapping.jsonl"

    print("[1] carregando index_mapping.jsonl...")
    key_to_faiss: dict[str, int] = {}
    with open(map_file) as f:
        for ln in f:
            d = json.loads(ln)
            key_to_faiss[d["key"]] = d["idx"]
    print(f"    {len(key_to_faiss):,} entradas no mapa faiss")

    print("[2] carregando dataset_v0.parquet...")
    df = pd.read_parquet(PARQUET, columns=["nome_bruto", "brand", "img_local"])
    # Mantém só linhas com img_local preenchido (as que foram matted)
    df = df[df["img_local"].notna()].copy()
    print(f"    {len(df):,} itens com img_local")

    print("[3] construindo text_index...")
    text_idx: dict[str, list[int]] = defaultdict(list)
    hits = miss_key = 0

    for _, row in df.iterrows():
        img_local = str(row["img_local"])
        mkey = img_key(img_local)
        faiss_idx = key_to_faiss.get(mkey)
        if faiss_idx is None:
            miss_key += 1
            continue

        nome  = str(row.get("nome_bruto") or "")
        brand = str(row.get("brand") or "")

        # Slug nome+brand (específico)
        slug_full = slug_norm(f"{nome} {brand}")
        if slug_full:
            text_idx[slug_full].append(faiss_idx)

        # Slug só nome (genérico) — só adiciona se diferente do full
        slug_nome = slug_norm(nome)
        if slug_nome and slug_nome != slug_full:
            text_idx[slug_nome].append(faiss_idx)

        hits += 1

    print(f"    hits: {hits:,} | sem faiss_idx: {miss_key:,}")
    print(f"    slugs únicos: {len(text_idx):,}")

    gaz_dir.mkdir(parents=True, exist_ok=True)
    out_path = gaz_dir / "text_index.json"
    out_path.write_text(json.dumps(dict(text_idx), ensure_ascii=False))
    print(f"[4] escrito: {out_path} ({out_path.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
