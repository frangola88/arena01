#!/usr/bin/env python3
"""CasaIQ Oracle → WhatsApp via CallMeBot.

Uso:
  notify.py "mensagem livre"
  notify.py --painel        # snapshot do estado atual
  notify.py --event "label" # evento específico
"""
import json, sys, time
from pathlib import Path
from urllib.parse import quote
import urllib.request

CONFIG = Path.home() / '.casaiq_whatsapp.json'
BATCH  = Path.home() / 'casaiq/batches'


def send(text: str) -> tuple[bool, str]:
    cfg = json.loads(CONFIG.read_text())
    phone, apikey = cfg['phone'], cfg['apikey']
    url = f'https://api.callmebot.com/whatsapp.php?phone={phone}&text={quote(text[:3000])}&apikey={apikey}'
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            body = r.read().decode('utf-8', errors='replace')
            return (r.status == 200), body[:500]
    except Exception as e:
        return False, f'{type(e).__name__}: {e}'


def _batch_status(name: str) -> str:
    sf = BATCH / name / 'status.json'
    if not sf.exists():
        return '—'
    try:
        d = json.loads(sf.read_text())
        done, total = d.get('done', 0), d.get('total', 0)
        rate = d.get('rate_per_s', 0)
        eta  = d.get('eta_min', 0)
        if total <= 0:
            return f'{done:,} (sem total)'
        pct = int(100 * done / total)
        if done >= total:
            return f'✅ {done:,}/{total:,}'
        eta_str = f'ETA {eta:.0f}min' if eta > 0 else f'{rate:.1f}/s'
        return f'⏳ {done:,}/{total:,} ({pct}%) {eta_str}'
    except Exception:
        return 'erro status.json'


def _coleta_total() -> int:
    sources = [
        Path.home() / 'br_vtex/data/export/superpro.jsonl',
        Path.home() / 'br_vtex/data/export/delupo.jsonl',
        Path.home() / 'br_vtex/data/export/ferimport.jsonl',
        Path.home() / 'casaiq_scraper/data/export/screwfix.jsonl',
        Path.home() / 'br_vtex_fedora_migrate/data/export/kennedy.jsonl',
        Path.home() / 'br_vtex_fedora_migrate/data/export/minas.jsonl',
        Path.home() / 'ali_local/data/export/aliexpress.jsonl',
    ]
    return sum(sum(1 for _ in open(p)) for p in sources if p.exists())


def painel_snapshot() -> str:
    lines = [f'🤖 *CasaIQ Oracle — {time.strftime("%H:%M %d/%m")}*', '']

    # Batches
    lines.append('*Pipeline ML:*')
    lines.append(f'• matting : {_batch_status("01_matting")}')
    lines.append(f'• dinov2  : {_batch_status("02_dinov2")}')
    lines.append(f'• faiss   : {_batch_status("03_faiss")}')
    lines.append('')

    # Gazetteer Oracle (embeddings.npy)
    emb_file = BATCH / '02_dinov2/embeddings.npy'
    if emb_file.exists():
        import struct
        # lê shape sem numpy: cabeçalho npy tem magic+version+header
        try:
            import numpy as np
            e = np.load(str(emb_file), mmap_mode='r')
            n_emb = e.shape[0]
        except Exception:
            n_emb = '?'
        lines.append(f'*Gazetteer Oracle:* {n_emb:,} emb')
    else:
        lines.append('*Gazetteer Oracle:* aguardando encode')
    lines.append('')

    # Coleta total
    try:
        total = _coleta_total()
        lines.append(f'*Coleta total:* {total:,} itens')
    except Exception:
        pass

    return '\n'.join(lines)


def main():
    if not CONFIG.exists():
        sys.exit(f'ERRO: {CONFIG} não existe')
    if len(sys.argv) < 2:
        sys.exit('uso: notify.py "msg" | --painel | --event "label"')

    arg = sys.argv[1]
    if arg == '--painel':
        msg = painel_snapshot()
    elif arg == '--event':
        label = sys.argv[2] if len(sys.argv) > 2 else 'evento'
        msg = f'⚡ *CasaIQ*\n{time.strftime("%H:%M")} — {label}'
    else:
        msg = ' '.join(sys.argv[1:])

    ok, body = send(msg)
    print(f'sent={ok}  resp={body[:200]}')
    if not ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
