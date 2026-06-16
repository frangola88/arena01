"""Batch 02 — DINOv2 encoding dos canônicos matted.

Lê PNGs matted do batch 01, encoda cada um com DINOv2-small, salva em embeddings.npy
mais um índice key→idx no embeddings_index.jsonl.

Resume-safe: já-encodados ficam num pickle persistente. Relançar pula.
"""
import json, os, pickle, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / '_common'))
from batch_base import BatchProcessor

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

BATCH_DIR = Path(os.environ.get('CASAIQ_BATCH_DIR', str(Path.home() / 'casaiq/batches/02_dinov2')))
MATTED_DIR = Path(os.environ.get('CASAIQ_MATTED_DIR', str(Path.home() / 'casaiq/batches/01_matting/output')))
MODEL_NAME = os.environ.get('CASAIQ_DINO_MODEL', 'facebook/dinov2-small')
USE_SIZE = int(os.environ.get('CASAIQ_DINO_SIZE', '448'))   # qual escala usar (_224, _448, _896)

class DinoEncodeBatch(BatchProcessor):
    def __init__(self):
        super().__init__('02_dinov2', BATCH_DIR)
        print(f'[02_dinov2] carregando {MODEL_NAME}...', flush=True)
        self.proc = AutoImageProcessor.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME).eval()
        self.dim = self.model.config.hidden_size
        print(f'[02_dinov2] dim={self.dim} size={USE_SIZE}', flush=True)
        self.embeddings_pkl = BATCH_DIR / 'embeddings.pkl'
        self._embeddings = {}  # key → np.array(384,)
        if self.embeddings_pkl.exists():
            with open(self.embeddings_pkl, 'rb') as f:
                self._embeddings = pickle.load(f)
            print(f'[02_dinov2] {len(self._embeddings)} embeddings já no pkl', flush=True)
        self._save_every = 200
        self._counter = 0

    def _persist(self):
        # Salva em pickle (acumulativo)
        with open(self.embeddings_pkl, 'wb') as f:
            pickle.dump(self._embeddings, f)

    @torch.no_grad()
    def process(self, png_path: str) -> tuple[bool, dict]:
        key = Path(png_path).stem  # sha do matting
        if key in self._embeddings:
            return True, {'key': key, 'cached': True}

        # Procura versão na escala USE_SIZE
        candidate = MATTED_DIR / f'{key}_{USE_SIZE}.jpg'
        if not candidate.exists():
            return False, {'reason': f'no _{USE_SIZE}.jpg'}

        try:
            img = Image.open(candidate).convert('RGB')
            inputs = self.proc(images=img, return_tensors='pt')
            out = self.model(**inputs)
            cls = out.last_hidden_state[:, 0, :]  # (1, 384)
            cls = F.normalize(cls, dim=-1)
            vec = cls[0].numpy().astype(np.float32)
            self._embeddings[key] = vec
            self._counter += 1
            if self._counter % self._save_every == 0:
                self._persist()
            return True, {'key': key}
        except Exception as e:
            return False, {'reason': f'{type(e).__name__}', 'msg': str(e)[:200]}

    def run(self, status_every=50):
        super().run(status_every)
        self._persist()
        # Materializa também como .npy + index pra leitura rápida
        if self._embeddings:
            keys = sorted(self._embeddings)
            mat = np.stack([self._embeddings[k] for k in keys])
            np.save(BATCH_DIR / 'embeddings.npy', mat)
            with open(BATCH_DIR / 'embeddings_index.jsonl', 'w') as f:
                for i, k in enumerate(keys):
                    f.write(json.dumps({'idx': i, 'key': k}) + '\n')
            print(f'[02_dinov2] {len(keys)} embeddings em {BATCH_DIR / "embeddings.npy"}', flush=True)

if __name__ == '__main__':
    DinoEncodeBatch().run()
