"""Etapa 1 — consolidação e dedupe.

Lê todos os JSONL espalhados (Fedora + Oracle via rsync), normaliza colunas,
deduplica em 3 passos (img_url exata, sku-por-fonte, nome-normalizado), gera
`dataset_v0.parquet`.

Uso:
    python consolidate.py            # roda tudo
    python consolidate.py --no-rsync # pula rsync (debug)
    python consolidate.py --stats    # só lê o parquet existente e mostra stats
"""
import argparse, json, hashlib, re, subprocess, sys, unicodedata
from pathlib import Path
from collections import Counter

try:
    import pandas as pd
except ImportError:
    sys.exit("pip install pandas pyarrow (no env do casaiq)")

BASE = Path.home() / 'projetos' / 'casaiq'
PIPE = BASE / 'pipeline'
RAW_LOCAL = BASE / 'data' / 'br_vtex' / 'data' / 'export'   # Fedora VTEX (kennedy, minas, corebral, simeao)
ALI_FEDORA = BASE / 'data' / 'ali_local' / 'data' / 'export' / 'aliexpress.jsonl'
ORACLE_PULL = BASE / 'data' / 'oracle_pull'                  # destino do rsync
DATASET_OUT = PIPE / 'stage1_consolidate' / 'dataset_v0.parquet'
REPORT_OUT  = PIPE / 'stage1_consolidate' / 'dedupe_report.txt'

ORACLE_PATHS = [
    # (label, caminho remoto, sub-pasta local)
    ('br_vtex',       '/home/ubuntu/br_vtex/data/export/',         'br_vtex'),
    ('casaiq_scraper','/home/ubuntu/casaiq_scraper/data/export/',  'casaiq_scraper'),
    ('ali_local',     '/home/ubuntu/ali_local/data/export/',       'ali_local'),
]

def slug_norm(s: str) -> str:
    """Normalização agressiva pra dedupe: lower, sem acento, só [a-z0-9 ]."""
    if not s: return ''
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')
    s = re.sub(r'[^a-zA-Z0-9 ]', ' ', s).lower()
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def rsync_oracle():
    """Puxa todos os JSONLs da Oracle pro Fedora."""
    ORACLE_PULL.mkdir(parents=True, exist_ok=True)
    for label, remote, sub in ORACLE_PATHS:
        local = ORACLE_PULL / sub
        local.mkdir(parents=True, exist_ok=True)
        cmd = ['rsync', '-avz', '--include=*.jsonl', '--include=*/',
               '--exclude=*', f'oracle-casaiq:{remote}', str(local) + '/']
        print(f'[rsync] {label} ← {remote}')
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f'[rsync] ERRO em {label}: {r.stderr[:300]}', file=sys.stderr)
        else:
            # contagem
            n = sum(1 for _ in local.rglob('*.jsonl'))
            print(f'        → {n} arquivo(s) jsonl')

def load_jsonl(path: Path, source_label: str) -> list[dict]:
    """Lê 1 JSONL e marca cada linha com 'fonte_local'."""
    out = []
    try:
        with open(path) as f:
            for ln in f:
                ln = ln.strip()
                if not ln: continue
                try:
                    d = json.loads(ln)
                    d['_source_file'] = source_label
                    out.append(d)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return []
    return out

def collect_all() -> pd.DataFrame:
    """Junta todos os JSONLs num só DataFrame."""
    rows = []

    # 1) Fedora VTEX
    for jsonl in RAW_LOCAL.glob('*.jsonl'):
        rows.extend(load_jsonl(jsonl, f'fedora/{jsonl.stem}'))

    # 2) Ali Fedora
    rows.extend(load_jsonl(ALI_FEDORA, 'fedora/ali'))

    # 3) Oracle (depois do rsync)
    for jsonl in ORACLE_PULL.rglob('*.jsonl'):
        rel = jsonl.relative_to(ORACLE_PULL).as_posix().replace('.jsonl', '')
        rows.extend(load_jsonl(jsonl, f'oracle/{rel}'))

    df = pd.DataFrame(rows)
    return df

def normalize_img_url(x):
    """Normaliza img_url — scrapers diferentes armazenam como str, dict ou list."""
    if isinstance(x, str): return x
    if isinstance(x, dict): return x.get('url') or x.get('src') or next(iter(x.values()), None)
    if isinstance(x, list): return x[0] if x else None
    return None

def dedupe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """3 passos: (1) img_url exata, (2) sku dentro da mesma fonte, (3) nome+brand normalizado."""
    stats = {'inicial': len(df)}

    # Normaliza img_url para str (scrapers diferentes usam dict/list/str)
    df['img_url'] = df['img_url'].apply(normalize_img_url)

    # PASSO 1 — dropa quem não tem imagem ou nome
    df = df[df['img_url'].notna() & df['img_url'].str.len().gt(5)]
    df = df[df['nome_bruto'].notna() & df['nome_bruto'].str.len().gt(2)]
    stats['após_filtros_básicos'] = len(df)

    # PASSO 2 — dedupe img_url exato (preferir fonte mais rica = a que tem mais campos preenchidos)
    df['_fill_score'] = df[['brand', 'sku', 'categoria']].notna().sum(axis=1)
    df = df.sort_values('_fill_score', ascending=False)
    df = df.drop_duplicates(subset=['img_url'], keep='first')
    stats['após_dedupe_img_url'] = len(df)

    # PASSO 3 — dedupe (sku, fonte) — mesmo SKU dentro do mesmo seller é a mesma coisa
    # FIX: astype(str) converte NaN/None para 'nan'/'None' (len>0), marcando items sem SKU
    # como has_sku=True e colapsando todos num único (fonte, 'nan'). Usar notna() primeiro.
    has_sku = df['sku'].notna() & df['sku'].astype(str).str.strip().str.len().gt(0)
    df_sku = df[has_sku].drop_duplicates(subset=['fonte', 'sku'], keep='first')
    df_nosku = df[~has_sku]
    df = pd.concat([df_sku, df_nosku], ignore_index=True)
    stats['após_dedupe_sku+fonte'] = len(df)

    # PASSO 4 — dedupe nome+brand normalizado (cross-source)
    df['_name_key'] = df['nome_bruto'].apply(slug_norm)
    df['_brand_key'] = df['brand'].fillna('').apply(slug_norm)
    df['_full_key'] = df['_name_key'] + '|' + df['_brand_key']
    df = df.drop_duplicates(subset=['_full_key'], keep='first')
    stats['após_dedupe_nome+brand'] = len(df)

    # Limpa colunas auxiliares
    df = df.drop(columns=['_fill_score', '_name_key', '_brand_key', '_full_key'])

    return df.reset_index(drop=True), stats

def write_report(df: pd.DataFrame, stats: dict) -> str:
    lines = []
    lines.append('=== Etapa 1 — Consolidação & Dedupe ===\n')
    lines.append(f'Saída: {DATASET_OUT}\n')
    lines.append(f'Total final: {len(df):,}\n\n')
    lines.append('Passos de redução:')
    for k, v in stats.items():
        lines.append(f'  {k:30s} {v:>8,}')
    lines.append('')

    lines.append('--- distribuição por fonte ---')
    for f, n in df['fonte'].value_counts().items():
        lines.append(f'  {f:30s} {n:>8,}')
    lines.append('')

    lines.append('--- distribuição por dominio_tipo ---')
    for f, n in df['dominio_tipo'].value_counts().items():
        lines.append(f'  {f:30s} {n:>8,}')
    lines.append('')

    lines.append('--- distribuição por idioma ---')
    for f, n in df['idioma'].value_counts().items():
        lines.append(f'  {f:30s} {n:>8,}')
    lines.append('')

    lines.append('--- top 20 brands ---')
    for f, n in df['brand'].value_counts().head(20).items():
        lines.append(f'  {str(f)[:30]:30s} {n:>8,}')
    lines.append('')

    lines.append('--- itens sem brand: {}/{} ({}%) ---'.format(
        df['brand'].isna().sum(), len(df), int(100*df['brand'].isna().sum()/len(df))
    ))
    lines.append('--- itens sem categoria: {}/{} ({}%) ---'.format(
        df['categoria'].fillna('').str.len().eq(0).sum(), len(df),
        int(100*df['categoria'].fillna('').str.len().eq(0).sum()/len(df))
    ))
    return '\n'.join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-rsync', action='store_true')
    ap.add_argument('--stats', action='store_true', help='só lê o parquet e mostra stats')
    args = ap.parse_args()

    if args.stats:
        df = pd.read_parquet(DATASET_OUT)
        print(write_report(df, {'lido_de_parquet': len(df)}))
        return

    if not args.no_rsync:
        rsync_oracle()

    print('\n[merge] carregando JSONLs...')
    df = collect_all()
    print(f'  → {len(df):,} linhas brutas')

    print('\n[dedupe] aplicando 4 passos...')
    df, stats = dedupe(df)

    print(f'\n[write] {DATASET_OUT}')
    DATASET_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATASET_OUT, index=False)

    report = write_report(df, stats)
    REPORT_OUT.write_text(report)
    print(report)

if __name__ == '__main__':
    main()
