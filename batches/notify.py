#!/usr/bin/env python3
"""CasaIQ Oracle → WhatsApp via CallMeBot.

Uso:
  notify.py "mensagem livre"
  notify.py --painel        # snapshot completo do estado
  notify.py --event "label" # evento específico

Lê credenciais de ~/.casaiq_whatsapp.json (mode 600).
"""
import json, sys, time
from pathlib import Path
from urllib.parse import quote
import urllib.request

CONFIG = Path.home() / '.casaiq_whatsapp.json'

def send(text: str) -> tuple[bool, str]:
    cfg = json.loads(CONFIG.read_text())
    phone = cfg['phone']
    apikey = cfg['apikey']
    text = text[:3000]
    url = f'https://api.callmebot.com/whatsapp.php?phone={phone}&text={quote(text)}&apikey={apikey}'
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            body = r.read().decode('utf-8', errors='replace')
            return (r.status == 200), body[:500]
    except Exception as e:
        return False, f'{type(e).__name__}: {e}'

def painel_snapshot() -> str:
    """Monta snapshot direto de arquivos locais na Oracle (sem SSH)."""
    lines = [f'🤖 *CasaIQ Oracle — {time.strftime("%H:%M %d/%m")}*', '']

    lines.append('*Coleta:*')
    sources = [
        ('superpro',  Path.home() / 'br_vtex/data/export/superpro.jsonl'),
        ('delupo',    Path.home() / 'br_vtex/data/export/delupo.jsonl'),
        ('ferimport', Path.home() / 'br_vtex/data/export/ferimport.jsonl'),
        ('screwfix',  Path.home() / 'casaiq_scraper/data/export/screwfix.jsonl'),
        ('kennedy',   Path.home() / 'br_vtex_fedora_migrate/data/export/kennedy.jsonl'),
        ('minas',     Path.home() / 'br_vtex_fedora_migrate/data/export/minas.jsonl'),
        ('ali',       Path.home() / 'ali_local/data/export/aliexpress.jsonl'),
    ]
    total = 0
    for name, path in sources:
        n = sum(1 for _ in open(path)) if path.exists() else 0
        total += n
        lines.append(f'• {name}: {n:,}')
    lines.append(f'• BR legados: 23.238')
    lines.append(f'*Total: {total + 23238:,}*')
    lines.append('')

    lines.append('*Batches:*')
    for batch in ['01_matting', '02_dinov2', '03_faiss', '04_sqlite_db']:
        sf = Path.home() / f'casaiq/batches/{batch}/status.json'
        if sf.exists():
            try:
                d = json.loads(sf.read_text())
                lines.append(f"• {batch}: {d.get('done',0):,}/{d.get('total',0):,} ETA={d.get('eta_min','?')}min")
            except Exception:
                lines.append(f'• {batch}: status.json com erro')
        else:
            out = Path.home() / f'casaiq/batches/{batch}/output'
            n = len(list(out.iterdir())) if out.exists() else 0
            lines.append(f'• {batch}: aguardando ({n} arquivos)')
    lines.append('')

    hc = Path.home() / 'casaiq/orchestrator/healthcheck.json'
    if hc.exists():
        try:
            h = json.loads(hc.read_text())
            lines.append(f"*Tmux vivos:* {h['n_alive']}/{h['n_alive']+h['n_dead']}")
            if h.get('dead'):
                lines.append(f"⚠️ mortos: {', '.join(h['dead'])}")
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
        msg = f'⚡ *CasaIQ evento*\n{time.strftime("%H:%M")} — {label}'
    else:
        msg = arg

    ok, body = send(msg)
    print(f'sent={ok}  resp={body[:200]}')
    if not ok:
        sys.exit(1)

if __name__ == '__main__':
    main()
