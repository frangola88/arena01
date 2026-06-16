"""Base de batch processor — todos os batches herdam dessa estrutura.

Garantias:
- Restart-safe: status.json + lista de items completados; relançar pula feitos
- Logging atomic: 1 linha JSONL por item processado
- Progresso a cada N itens: print + status.json atualizado
- Graceful shutdown em SIGINT/SIGTERM
"""
import json, os, signal, sys, time
from pathlib import Path

class BatchProcessor:
    def __init__(self, name: str, base_dir: Path):
        self.name = name
        self.dir = Path(base_dir)
        self.input_list = self.dir / 'input_list.txt'        # 1 item por linha
        self.output_dir = self.dir / 'output'                 # artefatos do batch
        self.log_file = self.dir / 'log.jsonl'                # 1 linha JSONL por item
        self.status_file = self.dir / 'status.json'
        self.errors_log = self.dir / 'errors.log'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._stop = False
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

    def _on_signal(self, *a):
        print('\n[BATCH] sinal recebido, vou terminar após o item atual...', flush=True)
        self._stop = True

    def load_input(self) -> list[str]:
        if not self.input_list.exists():
            sys.exit(f'ERRO: {self.input_list} não existe')
        return [l.strip() for l in self.input_list.read_text().splitlines() if l.strip()]

    def load_done(self) -> set:
        if not self.log_file.exists(): return set()
        done = set()
        with open(self.log_file) as f:
            for ln in f:
                try:
                    d = json.loads(ln)
                    if d.get('status') == 'ok':
                        done.add(d['key'])
                except json.JSONDecodeError:
                    continue
        return done

    def log_event(self, key: str, status: str, **extra):
        with open(self.log_file, 'a') as f:
            f.write(json.dumps({'key': key, 'status': status, 't': time.time(), **extra}) + '\n')

    def log_error(self, key: str, msg: str):
        with open(self.errors_log, 'a') as f:
            f.write(f'[{time.strftime("%H:%M:%S")}] {key}: {msg}\n')

    def write_status(self, total: int, done: int, ok: int, err: int, t0: float, current: str = ''):
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta_min = (total - done) / rate / 60 if rate > 0 else 0
        self.status_file.write_text(json.dumps({
            'batch': self.name,
            'total': total, 'done': done, 'ok': ok, 'err': err,
            'rate_per_s': round(rate, 2),
            'eta_min': round(eta_min, 1),
            'elapsed_min': round(elapsed / 60, 1),
            'current': current,
            'updated': time.strftime('%Y-%m-%d %H:%M:%S'),
        }, indent=2))

    def process(self, key: str) -> tuple[bool, dict]:
        """Override em cada batch. Retorna (sucesso, extras pra log)."""
        raise NotImplementedError

    def run(self, status_every: int = 50):
        all_inputs = self.load_input()
        done = self.load_done()
        pending = [k for k in all_inputs if k not in done]
        total = len(all_inputs)
        print(f'[{self.name}] {len(done)}/{total} já feitos, {len(pending)} pendentes', flush=True)

        t0 = time.time()
        ok = err = 0
        for i, key in enumerate(pending, 1):
            if self._stop: break
            try:
                success, extra = self.process(key)
                if success:
                    ok += 1
                    self.log_event(key, 'ok', **extra)
                else:
                    err += 1
                    self.log_event(key, 'err', **extra)
                    self.log_error(key, str(extra))
            except Exception as e:
                err += 1
                self.log_event(key, 'err', exception=str(e))
                self.log_error(key, f'EXCEPTION: {type(e).__name__}: {e}')
            if i % status_every == 0:
                self.write_status(total, len(done) + i, ok, err, t0, current=key)
                print(f'  [{i}/{len(pending)}] ok={ok} err={err} ({i/(time.time()-t0):.2f}/s)', flush=True)

        self.write_status(total, len(done) + ok + err, ok, err, t0)
        print(f'[{self.name}] FIM: ok={ok} err={err} em {(time.time()-t0)/60:.1f}min', flush=True)
