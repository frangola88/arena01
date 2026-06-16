"""Batch 04 v2 — gazetteer.db final (SQLite + sqlite-vec).

Schema fechado com cuco em 2026-05-31. 4 flags de pipeline separados:
matted | enriched_p1 | enriched_p2 | encoded.

Produz: gazetteer.db autocontido (~120-150 MB pra 18k itens)
  - tabela `items`     : metadata + thumbnail BLOB (128x128) + flags de estado
  - virtual `vec_items`: embeddings 384-dim (sqlite-vec)

Restart-safe via INSERT...ON CONFLICT(fonte, sku) DO UPDATE.
"""
import hashlib, io, json, os, sqlite3, sys, time
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import sqlite_vec
except ImportError:
    sys.exit('pip install sqlite-vec  (no env deep_learning)')

BATCH_DIR = Path(os.environ.get('CASAIQ_BATCH_DIR', str(Path.home() / 'casaiq/batches/04_sqlite_db')))
DINOV2_DIR = Path(os.environ.get('CASAIQ_DINOV2_DIR', str(Path.home() / 'casaiq/batches/02_dinov2')))
MATTED_DIR = Path(os.environ.get('CASAIQ_MATTED_DIR', str(Path.home() / 'casaiq/batches/01_matting/output')))

# Pode opcionalmente trazer dados de Pass 1/Pass 2 se já tiver rodado
PASS1 = Path(os.environ.get('CASAIQ_PASS1', str(Path.home() / 'casaiq/pipeline/stage3_enrich/pass1_classified.jsonl')))
PASS2 = Path(os.environ.get('CASAIQ_PASS2', str(Path.home() / 'casaiq/pipeline/stage3_enrich/pass2_normalized.jsonl')))

DB_PATH = BATCH_DIR / 'gazetteer.db'
THUMB_SIZE = 128

JSONL_SOURCES = [
    Path.home() / 'casaiq_scraper/data/export',
    Path.home() / 'br_vtex/data/export',
    Path.home() / 'br_vtex_fedora_migrate/data/export',
    Path.home() / 'ali_local/data/export',
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    -- identificação
    fonte TEXT NOT NULL,
    sku TEXT,
    product_url TEXT,
    -- texto
    nome_bruto TEXT NOT NULL,
    nome_limpo TEXT,
    brand TEXT,
    brand_canonical TEXT,
    -- categorização
    categoria_origem TEXT,
    categoria_inferida TEXT,
    util_casaiq INTEGER,
    -- origem
    idioma TEXT,
    pais_origem TEXT,
    dominio_tipo TEXT,
    -- imagens
    img_url TEXT,
    img_local TEXT,
    img_matted_sha TEXT,
    img_matted_path TEXT,
    thumbnail BLOB,
    -- estado do pipeline (4 flags separados)
    matted INTEGER DEFAULT 0,
    enriched_p1 INTEGER DEFAULT 0,
    enriched_p2 INTEGER DEFAULT 0,
    encoded INTEGER DEFAULT 0,
    -- provenance
    inserted_at TEXT,
    last_updated_at TEXT,
    UNIQUE(fonte, sku)
);
CREATE INDEX IF NOT EXISTS idx_brand_canonical ON items(brand_canonical);
CREATE INDEX IF NOT EXISTS idx_categoria       ON items(categoria_inferida);
CREATE INDEX IF NOT EXISTS idx_fonte           ON items(fonte);
CREATE INDEX IF NOT EXISTS idx_util_encoded    ON items(util_casaiq, encoded);
CREATE INDEX IF NOT EXISTS idx_matted_sha      ON items(img_matted_sha);
CREATE INDEX IF NOT EXISTS idx_pipeline_state  ON items(matted, enriched_p1, encoded);
CREATE INDEX IF NOT EXISTS idx_pais            ON items(pais_origem);
"""

# Mapa de fonte → país. Default = unknown.
PAIS_MAP = {
    'eletrogate.com': 'BR', 'filipeflop.com': 'BR', 'palaciodasferramentas.com.br': 'BR',
    'baudaeletronica.com.br': 'BR', 'ferramentaskennedy.com.br': 'BR',
    'minasferramentas.com.br': 'BR', 'superproatacado.com.br': 'BR',
    'corebral.com.br': 'BR', 'simeaorj.com': 'BR', 'ferimport.com.br': 'BR',
    'delupo.com.br': 'BR', 'anhangueraferramentas.com.br': 'BR',
    'lojadomecanico.com.br': 'BR',
    'screwfix.com': 'UK',
    'mouser.com': 'USA', 'digikey.com': 'USA',
    'aliexpress.com': 'CN', 'aliexpress.us': 'CN',
}
IDIOMA_DEFAULT = {'BR': 'pt', 'UK': 'en', 'USA': 'en', 'CN': 'pt'}  # Ali nosso é em pt

def infer_pais(fonte: str) -> str:
    if not fonte: return 'unknown'
    f = fonte.lower().strip()
    if f in PAIS_MAP: return PAIS_MAP[f]
    if f.endswith('.com.br') or '.br' in f: return 'BR'
    if f.endswith('.co.uk') or '.uk' in f: return 'UK'
    return 'unknown'

def sha_of_path(path: str) -> str:
    return hashlib.sha256(path.encode()).hexdigest()[:16]

def make_thumbnail(matted_path: Path) -> bytes | None:
    try:
        img = Image.open(matted_path).convert('RGBA')
        img.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
        canvas = Image.new('RGB', (THUMB_SIZE, THUMB_SIZE), (255, 255, 255))
        offset = ((THUMB_SIZE - img.size[0]) // 2, (THUMB_SIZE - img.size[1]) // 2)
        canvas.paste(img.convert('RGB'), offset, img.split()[-1])
        buf = io.BytesIO()
        canvas.save(buf, 'JPEG', quality=85, optimize=True)
        return buf.getvalue()
    except Exception:
        return None

def open_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.executescript(SCHEMA)
    conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS vec_items USING vec0(embedding FLOAT[384])")
    return conn

def load_embeddings() -> dict[str, np.ndarray]:
    npy = DINOV2_DIR / 'embeddings.npy'
    idx = DINOV2_DIR / 'embeddings_index.jsonl'
    if not (npy.exists() and idx.exists()): return {}
    mat = np.load(npy)
    lookup = {}
    with open(idx) as f:
        for ln in f:
            d = json.loads(ln)
            lookup[d['key']] = mat[d['idx']]
    return lookup

def load_pass1() -> dict:
    """Retorna {_id: {util_casaiq, categoria_inferida}}. Vazio se não rodou."""
    if not PASS1.exists(): return {}
    out = {}
    with open(PASS1) as f:
        for ln in f:
            try:
                d = json.loads(ln)
                out[d.get('_id') or d.get('key')] = d
            except json.JSONDecodeError:
                continue
    return out

def load_pass2() -> dict:
    if not PASS2.exists(): return {}
    out = {}
    with open(PASS2) as f:
        for ln in f:
            try:
                d = json.loads(ln)
                out[d.get('_id') or d.get('key')] = d
            except json.JSONDecodeError:
                continue
    return out

def main():
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    NOW = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f'[04_sqlite_db v2] {DB_PATH}', flush=True)

    embeddings = load_embeddings()
    pass1 = load_pass1()
    pass2 = load_pass2()
    print(f'  embeddings: {len(embeddings):,} | pass1: {len(pass1):,} | pass2: {len(pass2):,}', flush=True)

    conn = open_db(DB_PATH)
    cur = conn.cursor()
    inserted = with_thumb = with_vec = with_p1 = with_p2 = 0

    for src_dir in JSONL_SOURCES:
        if not src_dir.exists(): continue
        for jsonl in src_dir.glob('*.jsonl'):
            print(f'  → {jsonl.name}', flush=True)
            with open(jsonl) as f:
                for ln in f:
                    try:
                        d = json.loads(ln)
                    except json.JSONDecodeError:
                        continue

                    fonte = (d.get('fonte') or '').strip().lower()
                    sku = str(d.get('sku') or '')[:80]
                    nome_bruto = d.get('nome_bruto')
                    if not (fonte and nome_bruto):
                        continue

                    img_url = d.get('img_url') if isinstance(d.get('img_url'), str) else None
                    img_local = d.get('img_local') or ''
                    sha = sha_of_path(img_local) if img_local else None

                    matted_png = (MATTED_DIR / f'{sha}.png') if sha else None
                    matted_path = str(matted_png) if matted_png and matted_png.exists() else None
                    matted_flag = 1 if matted_path else 0

                    thumb = make_thumbnail(matted_png) if matted_path else None
                    if thumb: with_thumb += 1

                    # Pass 1 / Pass 2 lookups (keyed por _id ou fonte+sku)
                    key_p = f'{fonte}::{sku}'
                    p1 = pass1.get(key_p) or {}
                    p2 = pass2.get(key_p) or {}
                    has_p1 = 1 if p1 else 0
                    has_p2 = 1 if p2 else 0
                    if has_p1: with_p1 += 1
                    if has_p2: with_p2 += 1

                    nome_limpo = p2.get('nome_limpo') or d.get('nome_limpo')
                    brand_canonical = p2.get('brand_canonical') or d.get('brand_canonical')
                    util_casaiq = p1.get('util_casaiq')
                    util_casaiq = int(util_casaiq) if util_casaiq is not None else None
                    categoria_inferida = p1.get('categoria_inferida') or d.get('categoria_inferida')

                    pais = infer_pais(fonte)
                    idioma = d.get('idioma') or IDIOMA_DEFAULT.get(pais)
                    encoded_flag = 1 if (sha and sha in embeddings) else 0

                    cur.execute("""
                        INSERT INTO items (
                            fonte, sku, product_url,
                            nome_bruto, nome_limpo, brand, brand_canonical,
                            categoria_origem, categoria_inferida, util_casaiq,
                            idioma, pais_origem, dominio_tipo,
                            img_url, img_local, img_matted_sha, img_matted_path, thumbnail,
                            matted, enriched_p1, enriched_p2, encoded,
                            inserted_at, last_updated_at
                        ) VALUES (?,?,?, ?,?,?,?, ?,?,?, ?,?,?, ?,?,?,?,?, ?,?,?,?, ?,?)
                        ON CONFLICT(fonte, sku) DO UPDATE SET
                            nome_bruto=excluded.nome_bruto,
                            nome_limpo=COALESCE(excluded.nome_limpo, items.nome_limpo),
                            brand_canonical=COALESCE(excluded.brand_canonical, items.brand_canonical),
                            util_casaiq=COALESCE(excluded.util_casaiq, items.util_casaiq),
                            categoria_inferida=COALESCE(excluded.categoria_inferida, items.categoria_inferida),
                            img_matted_sha=COALESCE(excluded.img_matted_sha, items.img_matted_sha),
                            img_matted_path=COALESCE(excluded.img_matted_path, items.img_matted_path),
                            thumbnail=COALESCE(excluded.thumbnail, items.thumbnail),
                            matted=excluded.matted,
                            enriched_p1=excluded.enriched_p1,
                            enriched_p2=excluded.enriched_p2,
                            encoded=excluded.encoded,
                            last_updated_at=excluded.last_updated_at
                    """, (
                        fonte, sku, d.get('product_url'),
                        nome_bruto, nome_limpo, d.get('brand'), brand_canonical,
                        d.get('categoria'), categoria_inferida, util_casaiq,
                        idioma, pais, d.get('dominio_tipo'),
                        img_url, img_local, sha, matted_path, thumb,
                        matted_flag, has_p1, has_p2, encoded_flag,
                        NOW, NOW
                    ))
                    item_id = cur.lastrowid or cur.execute(
                        "SELECT id FROM items WHERE fonte=? AND sku=?", (fonte, sku)).fetchone()[0]

                    if encoded_flag:
                        vec = embeddings[sha].astype(np.float32).tobytes()
                        cur.execute("INSERT OR REPLACE INTO vec_items(rowid, embedding) VALUES (?, ?)",
                                    (item_id, vec))
                        with_vec += 1

                    inserted += 1
                    if inserted % 1000 == 0:
                        conn.commit()
                        print(f'    [{inserted:,}] thumb={with_thumb} vec={with_vec} p1={with_p1} p2={with_p2}', flush=True)
    conn.commit()

    # Stats finais
    n_items = cur.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    n_thumb = cur.execute("SELECT COUNT(*) FROM items WHERE thumbnail IS NOT NULL").fetchone()[0]
    n_vec = cur.execute("SELECT COUNT(*) FROM vec_items").fetchone()[0]
    n_util = cur.execute("SELECT COUNT(*) FROM items WHERE util_casaiq=1").fetchone()[0]
    n_p1 = cur.execute("SELECT COUNT(*) FROM items WHERE enriched_p1=1").fetchone()[0]
    n_p2 = cur.execute("SELECT COUNT(*) FROM items WHERE enriched_p2=1").fetchone()[0]
    by_pais = dict(cur.execute("SELECT pais_origem, COUNT(*) FROM items GROUP BY pais_origem").fetchall())
    by_fonte = dict(cur.execute("SELECT fonte, COUNT(*) FROM items GROUP BY fonte ORDER BY 2 DESC LIMIT 15").fetchall())
    db_mb = DB_PATH.stat().st_size / 1e6
    elapsed = time.time() - t0

    status = {
        'batch': '04_sqlite_db',
        'total': n_items, 'done': n_items, 'ok': n_items, 'err': 0,
        'rate_per_s': round(n_items / elapsed, 2),
        'eta_min': 0, 'elapsed_min': round(elapsed / 60, 2),
        'current': '', 'updated': NOW,
        'extras': {
            'db_path': str(DB_PATH), 'db_size_mb': round(db_mb, 1),
            'items_total': n_items,
            'with_thumbnail': n_thumb, 'with_embedding': n_vec,
            'with_pass1': n_p1, 'with_pass2': n_p2,
            'util_casaiq_true': n_util,
            'by_pais': by_pais, 'by_fonte_top': by_fonte,
        }
    }
    (BATCH_DIR / 'status.json').write_text(json.dumps(status, indent=2, ensure_ascii=False))

    print(f'\n[04_sqlite_db] FIM em {elapsed:.1f}s')
    print(f'  {n_items:,} items | thumb {n_thumb:,} | vec {n_vec:,} | p1 {n_p1:,} | p2 {n_p2:,}')
    print(f'  util_casaiq=1: {n_util:,}')
    print(f'  DB: {db_mb:.1f} MB')
    print(f'  por país: {by_pais}')

    # Smoke test só se houver embeddings
    if n_vec:
        row = cur.execute(
            "SELECT i.id, i.nome_bruto, i.brand_canonical FROM items i JOIN vec_items v ON v.rowid=i.id LIMIT 1"
        ).fetchone()
        if row:
            qid = row[0]
            qvec = cur.execute("SELECT embedding FROM vec_items WHERE rowid=?", (qid,)).fetchone()[0]
            print(f'\n  smoke: vizinhos de [{qid}] {row[1][:60]} ({row[2]})')
            for r in cur.execute("""
                SELECT i.id, i.nome_bruto, i.brand_canonical,
                       vec_distance_cosine(v.embedding, ?) AS dist
                FROM vec_items v JOIN items i ON i.id=v.rowid
                ORDER BY dist LIMIT 5
            """, (qvec,)):
                print(f'    [{r[0]}] d={r[3]:.3f}  {r[1][:60]} ({r[2]})')

    conn.close()

if __name__ == '__main__':
    main()
