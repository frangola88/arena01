"""Etapa 3 — enrichment com Ollama local. Custo R$0.

Two-pass:
  Pass 1 (rápido): util_casaiq + categoria — para TODOS os itens.
  Pass 2 (mais lento): nome_limpo + brand_canonical — só para util_casaiq=true.

Restart-safe: lê jsonl incremental, pula items já processados.
Cache de brand: marcas já normalizadas são reusadas (evita re-rodar LLM).

Uso:
    python enrich_ollama.py pass1                 # classifica todos
    python enrich_ollama.py pass2                 # normaliza só util=true
    python enrich_ollama.py pass1 --model qwen2.5:7b
    python enrich_ollama.py stats                 # contagens
"""
import argparse, json, os, sys, time, signal
from pathlib import Path
import requests
import pandas as pd

PIPE = Path.home() / 'projetos' / 'casaiq' / 'pipeline'
PARQUET_V0 = PIPE / 'stage1_consolidate' / 'dataset_v0.parquet'
STAGE3_DIR = PIPE / 'stage3_enrich'
PASS1_OUT  = STAGE3_DIR / 'pass1_classified.jsonl'
PASS2_OUT  = STAGE3_DIR / 'pass2_normalized.jsonl'
BRAND_CACHE = STAGE3_DIR / 'brand_cache.json'

OLLAMA = os.environ.get('OLLAMA_URL', 'http://localhost:11434')
DEFAULT_MODEL = os.environ.get('OLLAMA_MODEL', 'llama3.2:3b')

CATEGORIAS = [
    'ferramenta_manual', 'ferramenta_eletrica', 'fixador', 'medicao',
    'abrasivos', 'solda', 'jardim', 'automotivo', 'eletronico',
    'epi', 'construcao', 'outro'
]

PROMPT_PASS1 = """Você classifica produtos para um inventário doméstico/oficina.

REGRAS:
- Responda SOMENTE JSON válido: {"util": true|false, "cat": "..."}
- util=true se for ferramenta, equipamento, acessório ou item que fica em casa/oficina/garagem
- util=false se for: insumo a granel grande (sacos de cimento, areia), EPI ultra-específico industrial, equipamento médico, alimento, item B2B sem uso doméstico
- cat tem que ser UMA das: ferramenta_manual, ferramenta_eletrica, fixador, medicao, abrasivos, solda, jardim, automotivo, eletronico, epi, construcao, outro

EXEMPLOS:
"Furadeira Bosch 750W" → {"util": true, "cat": "ferramenta_eletrica"}
"Chave de Fenda Philips 6mm" → {"util": true, "cat": "ferramenta_manual"}
"Parafuso M6 100un" → {"util": true, "cat": "fixador"}
"Cimento Portland 50kg" → {"util": false, "cat": "construcao"}
"Capacete EPI Industrial Soldagem NR-6" → {"util": false, "cat": "epi"}
"Multímetro Digital Hikari" → {"util": true, "cat": "medicao"}

PRODUTO: {nome}
JSON:"""

PROMPT_PASS2 = """Você normaliza nomes de produtos.

REGRAS:
- Responda SOMENTE JSON válido: {"nome": "...", "brand": "..."}
- nome = nome curto e limpo em português, sem stopwords desnecessárias, sem códigos numéricos longos
- brand = marca canônica (DEWALT, BOSCH, MAKITA, STANLEY, IRWIN, VONDER, TRAMONTINA, WORKER, KENNEDY, etc.) ou null se não identificar
- Preserve modelo principal e potência/tamanho se relevante

EXEMPLOS:
input: "Furadeira de Impacto Bosch GSB 550 RE 550W"
output: {"nome": "Furadeira de Impacto 550W", "brand": "BOSCH"}

input: "WORKER 12 Pcs Wood Carving Hand Chisel Tool Set"
output: {"nome": "Jogo de Formões para Madeira 12 peças", "brand": "WORKER"}

input: "STANLEY 77082 Heavy Duty Utility Knife Blades 10 Pack"
output: {"nome": "Lâminas de Estilete 10 unidades", "brand": "STANLEY"}

PRODUTO: {nome}
JSON:"""

def ollama_call(prompt: str, model: str, max_retries: int = 2) -> dict | None:
    """Chama Ollama com format=json. Retorna dict ou None se falhar."""
    for attempt in range(max_retries + 1):
        try:
            r = requests.post(f'{OLLAMA}/api/generate', json={
                'model': model,
                'prompt': prompt,
                'format': 'json',
                'stream': False,
                'options': {'temperature': 0.0, 'num_predict': 200},
            }, timeout=60)
            if r.status_code != 200:
                continue
            text = r.json().get('response', '').strip()
            return json.loads(text)
        except (requests.RequestException, json.JSONDecodeError):
            if attempt < max_retries:
                time.sleep(1)
                continue
    return None

def load_done_keys(path: Path, key: str = '_id') -> set:
    if not path.exists(): return set()
    done = set()
    with open(path) as f:
        for ln in f:
            try:
                done.add(json.loads(ln)[key])
            except (json.JSONDecodeError, KeyError):
                continue
    return done

def append_json(path: Path, obj: dict):
    with open(path, 'a') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')

class GracefulExit:
    def __init__(self):
        self.stop = False
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)
    def _handler(self, *args):
        print('\n[INT] vou terminar após o item atual...', flush=True)
        self.stop = True

def pass1(model: str, limit: int | None = None):
    """Classifica util_casaiq + categoria em todos os itens."""
    if not PARQUET_V0.exists():
        sys.exit(f'ERRO: {PARQUET_V0} não existe. Rode a Etapa 1 antes.')
    df = pd.read_parquet(PARQUET_V0)
    df['_id'] = df.index.astype(str)
    done = load_done_keys(PASS1_OUT)
    pending = df[~df['_id'].isin(done)]
    if limit:
        pending = pending.head(limit)
    total = len(pending)
    print(f'[PASS1] modelo={model} | {len(done)} já feitos | {total} pendentes')

    t0 = time.time()
    exit_ctl = GracefulExit()
    ok = err = 0
    for i, (_, row) in enumerate(pending.iterrows(), 1):
        if exit_ctl.stop: break
        nome = str(row.get('nome_bruto', ''))[:300]
        if not nome:
            continue
        prompt = PROMPT_PASS1.replace('{nome}', nome)
        res = ollama_call(prompt, model)
        if res and isinstance(res, dict) and 'util' in res:
            out = {
                '_id': row['_id'],
                'util_casaiq': bool(res.get('util')),
                'categoria_inferida': str(res.get('cat', 'outro')),
            }
            append_json(PASS1_OUT, out)
            ok += 1
        else:
            err += 1
        if i % 50 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta_min = (total - i) / rate / 60 if rate > 0 else 0
            print(f'[PASS1 {i}/{total}] ok={ok} err={err} ({rate:.1f}/s) ETA={eta_min:.0f}min')
    print(f'[PASS1] FIM: ok={ok} err={err} em {(time.time()-t0)/60:.1f}min')

def pass2(model: str, limit: int | None = None):
    """Normaliza nome+brand para itens util_casaiq=true."""
    df = pd.read_parquet(PARQUET_V0)
    df['_id'] = df.index.astype(str)
    pass1 = pd.read_json(PASS1_OUT, lines=True)
    util = pass1[pass1['util_casaiq']]['_id'].tolist()
    df_util = df[df['_id'].isin(util)]
    done = load_done_keys(PASS2_OUT)
    pending = df_util[~df_util['_id'].isin(done)]
    if limit:
        pending = pending.head(limit)
    total = len(pending)
    print(f'[PASS2] modelo={model} | {len(done)} já feitos | {total} pendentes')

    brand_cache = json.loads(BRAND_CACHE.read_text()) if BRAND_CACHE.exists() else {}
    t0 = time.time()
    exit_ctl = GracefulExit()
    ok = err = cache_hits = 0
    for i, (_, row) in enumerate(pending.iterrows(), 1):
        if exit_ctl.stop: break
        nome = str(row.get('nome_bruto', ''))[:300]
        if not nome: continue

        # Cache hit: brand inicial bate com uma já normalizada
        cached_brand = None
        original_brand = str(row.get('brand') or '').strip().upper()
        if original_brand and original_brand in brand_cache:
            cached_brand = brand_cache[original_brand]
            cache_hits += 1

        prompt = PROMPT_PASS2.replace('{nome}', nome)
        res = ollama_call(prompt, model)
        if res and isinstance(res, dict) and 'nome' in res:
            new_brand = res.get('brand') or cached_brand
            if new_brand and original_brand and original_brand not in brand_cache:
                brand_cache[original_brand] = new_brand
            out = {
                '_id': row['_id'],
                'nome_limpo': str(res.get('nome', ''))[:200],
                'brand_canonical': new_brand,
            }
            append_json(PASS2_OUT, out)
            ok += 1
        else:
            err += 1
        if i % 50 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta_min = (total - i) / rate / 60 if rate > 0 else 0
            print(f'[PASS2 {i}/{total}] ok={ok} err={err} cache={cache_hits} ({rate:.1f}/s) ETA={eta_min:.0f}min')
            BRAND_CACHE.write_text(json.dumps(brand_cache, ensure_ascii=False, indent=2))
    BRAND_CACHE.write_text(json.dumps(brand_cache, ensure_ascii=False, indent=2))
    print(f'[PASS2] FIM: ok={ok} err={err} cache_hits={cache_hits} em {(time.time()-t0)/60:.1f}min')

def stats():
    """Resumo do que tá feito."""
    if PARQUET_V0.exists():
        df = pd.read_parquet(PARQUET_V0)
        print(f'dataset_v0:    {len(df):,} itens')
    if PASS1_OUT.exists():
        p1 = pd.read_json(PASS1_OUT, lines=True)
        print(f'pass1 done:    {len(p1):,}')
        print(f'  util=true:   {p1["util_casaiq"].sum():,}')
        print(f'  util=false:  {(~p1["util_casaiq"]).sum():,}')
        print(f'  categorias:')
        for c, n in p1['categoria_inferida'].value_counts().items():
            print(f'    {c:25s} {n:>8,}')
    if PASS2_OUT.exists():
        p2 = pd.read_json(PASS2_OUT, lines=True)
        print(f'pass2 done:    {len(p2):,}')
        print(f'  brands únicos (top10):')
        for b, n in p2['brand_canonical'].value_counts().head(10).items():
            print(f'    {str(b)[:25]:25s} {n:>8,}')
    if BRAND_CACHE.exists():
        bc = json.loads(BRAND_CACHE.read_text())
        print(f'brand_cache:   {len(bc)} entradas')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=['pass1', 'pass2', 'stats'])
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()
    STAGE3_DIR.mkdir(parents=True, exist_ok=True)
    if args.phase == 'pass1': pass1(args.model, args.limit)
    elif args.phase == 'pass2': pass2(args.model, args.limit)
    elif args.phase == 'stats': stats()

if __name__ == '__main__':
    main()
