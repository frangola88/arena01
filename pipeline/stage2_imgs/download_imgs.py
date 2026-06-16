"""Etapa 2 — download de imagens com rate-limit por domínio.

Lê o dataset_v0.parquet, pra cada item baixa img_url, redimensiona pra 448x448
(tile padrão Qwen2-VL), salva como JPEG Q85 em ~/dataset_imgs/<sha256>.jpg.

Rate-limit: 1.5s entre requests no mesmo domínio (configurável).
Multi-thread bucketed por domínio (não compete contra si mesmo num único site).
Restart-safe: pula hashes já presentes em disco.

Pode rodar do Fedora ou Oracle. Imagens ficam onde rodar.

Uso:
    python download_imgs.py                          # tudo
    python download_imgs.py --only-util              # só util_casaiq=true (precisa stage3 pass1)
    python download_imgs.py --workers 12 --sleep 1.0
    python download_imgs.py stats                    # contagens
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from queue import Queue
from threading import Thread, Lock
import requests

try:
    from PIL import Image
    import pandas as pd
except ImportError:
    sys.exit("pip install pillow pandas pyarrow")

PIPE = Path.home() / 'projetos' / 'casaiq' / 'pipeline'
PARQUET_V0 = PIPE / 'stage1_consolidate' / 'dataset_v0.parquet'
PASS1 = PIPE / 'stage3_enrich' / 'pass1_classified.jsonl'
IMG_DIR = Path(os.environ.get('CASAIQ_IMG_DIR', str(Path.home() / 'dataset_imgs')))
IMG_INDEX = PIPE / 'stage2_imgs' / 'img_index.jsonl'
ERROR_LOG = PIPE / 'stage2_imgs' / 'errors.log'

UA = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36'
TARGET_SIZE = 448

def domain_of(url: str) -> str:
    try: return urlparse(url).netloc.lower()
    except Exception: return 'unknown'

def hash_url(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()

def fetch_and_save(url: str, sleep_before: float) -> tuple[str, str | None]:
    """Return (status, img_path). status in {ok, exists, http_err, decode_err, save_err}."""
    h = hash_url(url)
    dest = IMG_DIR / f'{h}.jpg'
    if dest.exists():
        return ('exists', str(dest))
    if sleep_before > 0:
        time.sleep(sleep_before)
    try:
        r = requests.get(url, headers={'User-Agent': UA}, timeout=20, stream=True)
        if r.status_code != 200 or not r.headers.get('Content-Type', '').startswith('image'):
            return (f'http_{r.status_code}', None)
        from io import BytesIO
        buf = BytesIO(r.content)
        img = Image.open(buf).convert('RGB')
        # resize mantendo aspecto, depois pad
        img.thumbnail((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
        bg = Image.new('RGB', (TARGET_SIZE, TARGET_SIZE), (255, 255, 255))
        offset = ((TARGET_SIZE - img.size[0]) // 2, (TARGET_SIZE - img.size[1]) // 2)
        bg.paste(img, offset)
        bg.save(dest, 'JPEG', quality=85, optimize=True)
        return ('ok', str(dest))
    except Image.UnidentifiedImageError:
        return ('decode_err', None)
    except Exception as e:
        return (f'err:{type(e).__name__}', None)

def log_err(url: str, status: str):
    with open(ERROR_LOG, 'a') as f:
        f.write(f'[{time.strftime("%H:%M:%S")}] {status:20s} {url}\n')

class DomainScheduler:
    """1 worker por domínio. Cada worker dorme sleep_per_req entre requests."""
    def __init__(self, sleep_per_req: float, on_result):
        self.sleep_per_req = sleep_per_req
        self.queues: dict[str, Queue] = {}
        self.threads: list[Thread] = []
        self.on_result = on_result
        self.lock = Lock()

    def submit(self, idx: int, url: str):
        d = domain_of(url)
        with self.lock:
            if d not in self.queues:
                q = Queue()
                self.queues[d] = q
                t = Thread(target=self._worker, args=(d, q), daemon=True)
                self.threads.append(t)
                t.start()
        self.queues[d].put((idx, url))

    def _worker(self, d: str, q: Queue):
        last_request = 0.0
        while True:
            item = q.get()
            if item is None: break
            idx, url = item
            elapsed = time.time() - last_request
            wait = max(0.0, self.sleep_per_req - elapsed)
            status, path = fetch_and_save(url, sleep_before=wait)
            last_request = time.time()
            self.on_result(idx, url, status, path)

    def wait_drain(self):
        # Sinaliza fim e espera todas as queues drenarem (sem timeout —
        # domínios grandes podem levar horas; join com timeout curto
        # desistia da espera e reportava "FIM" com download incompleto)
        for q in self.queues.values():
            q.put(None)
        for t in self.threads:
            t.join()

def download_all(only_util: bool, sleep_per_req: float, max_items: int | None):
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    IMG_INDEX.parent.mkdir(parents=True, exist_ok=True)

    if not PARQUET_V0.exists():
        sys.exit(f'ERRO: {PARQUET_V0} não existe. Rode Etapa 1 antes.')
    df = pd.read_parquet(PARQUET_V0)
    df['_id'] = df.index.astype(str)

    if only_util:
        if not PASS1.exists():
            sys.exit('ERRO: --only-util precisa Stage 3 Pass 1 feito.')
        p1 = pd.read_json(PASS1, lines=True)
        util_ids = set(p1[p1['util_casaiq']]['_id'].astype(str))  # pd.read_json infere int64
        df = df[df['_id'].isin(util_ids)]

    # Pula URLs já no índice (restart-safe)
    done = set()
    if IMG_INDEX.exists():
        with open(IMG_INDEX) as f:
            for ln in f:
                try: done.add(json.loads(ln)['img_url'])
                except: continue
    df = df[~df['img_url'].isin(done)]
    if max_items:
        df = df.head(max_items)

    by_domain = defaultdict(int)
    for u in df['img_url']:
        by_domain[domain_of(u)] += 1
    print(f'[2] {len(df):,} a baixar | {len(by_domain)} domínios')
    print(f'    sleep_per_req={sleep_per_req}s | TARGET_SIZE={TARGET_SIZE} | dest={IMG_DIR}')
    for d, n in sorted(by_domain.items(), key=lambda x: -x[1])[:10]:
        print(f'    {d:40s} {n:>8,}')

    counts = defaultdict(int)
    lock = Lock()
    idx_file = open(IMG_INDEX, 'a')
    t0 = time.time()
    total = len(df)
    processed = [0]

    def on_result(idx, url, status, path):
        with lock:
            counts[status] += 1
            processed[0] += 1
            if status in ('ok', 'exists') and path:
                idx_file.write(json.dumps({
                    '_id': idx, 'img_url': url, 'img_path': path
                }, ensure_ascii=False) + '\n')
                idx_file.flush()
            else:
                log_err(url, status)
            if processed[0] % 200 == 0:
                rate = processed[0] / (time.time() - t0)
                eta_min = (total - processed[0]) / rate / 60 if rate > 0 else 0
                print(f'[2 {processed[0]}/{total}] {dict(counts)} ({rate:.1f}/s) ETA={eta_min:.0f}min')

    sched = DomainScheduler(sleep_per_req, on_result)
    for _, row in df.iterrows():
        sched.submit(row['_id'], row['img_url'])
    sched.wait_drain()
    idx_file.close()

    elapsed = (time.time() - t0) / 60
    print(f'[2] FIM em {elapsed:.1f}min: {dict(counts)}')

def stats():
    if IMG_INDEX.exists():
        n_idx = sum(1 for _ in open(IMG_INDEX))
        print(f'img_index entries: {n_idx:,}')
    n_disk = sum(1 for _ in IMG_DIR.glob('*.jpg')) if IMG_DIR.exists() else 0
    print(f'imgs no disco:     {n_disk:,}')
    if IMG_DIR.exists():
        size_mb = sum(f.stat().st_size for f in IMG_DIR.glob('*.jpg')) / (1024*1024)
        print(f'tamanho total:     {size_mb:.0f} MB')
    if ERROR_LOG.exists():
        n_err = sum(1 for _ in open(ERROR_LOG))
        print(f'erros logados:     {n_err:,}')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', nargs='?', default='run', choices=['run', 'stats'])
    ap.add_argument('--only-util', action='store_true')
    ap.add_argument('--sleep', type=float, default=1.5)
    ap.add_argument('--max-items', type=int, default=None)
    args = ap.parse_args()
    if args.mode == 'stats': stats()
    else: download_all(args.only_util, args.sleep, args.max_items)

if __name__ == '__main__':
    main()
