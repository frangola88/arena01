"""Batch 01 — matting (remoção de fundo) com rembg/U2-Net.

Input: lista de paths .jpg em input_list.txt (1 path por linha)
Output: PNG com alpha channel em output/<sha>.png
        + multi-escala 224, 448, 896 em output/<sha>_<size>.png

Configurável:
  CASAIQ_BATCH_DIR    diretório base do batch (default ~/casaiq/batches/01_matting)
  CASAIQ_SIZES        lista de escalas (default "224,448,896")
"""
import hashlib, os, sys
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / '_common'))
from batch_base import BatchProcessor

from PIL import Image
from rembg import new_session, remove

BATCH_DIR = Path(os.environ.get('CASAIQ_BATCH_DIR', str(Path.home() / 'casaiq/batches/01_matting')))
SIZES = [int(x) for x in os.environ.get('CASAIQ_SIZES', '224,448,896').split(',')]

class MattingBatch(BatchProcessor):
    def __init__(self):
        super().__init__('01_matting', BATCH_DIR)
        print(f'[01_matting] inicializando rembg U2-Net...', flush=True)
        self.session = new_session('u2net')
        print(f'[01_matting] sizes: {SIZES}', flush=True)

    def _key(self, src_path: str) -> str:
        return hashlib.sha256(src_path.encode()).hexdigest()[:16]

    def process(self, src_path: str) -> tuple[bool, dict]:
        if not Path(src_path).exists():
            return False, {'reason': 'src_not_found'}
        try:
            with open(src_path, 'rb') as f:
                input_bytes = f.read()
            # Matting: retorna PNG com alpha
            matted_bytes = remove(input_bytes, session=self.session)
            matted = Image.open(BytesIO(matted_bytes)).convert('RGBA')
        except Exception as e:
            return False, {'reason': f'matting:{type(e).__name__}', 'msg': str(e)[:200]}

        key = self._key(src_path)
        # Salva original matted (resolução nativa, PNG com alpha)
        out_full = self.output_dir / f'{key}.png'
        matted.save(out_full, 'PNG', optimize=True)

        # Salva multi-escala (RGB com fundo branco — mais comum pra encoders)
        for size in SIZES:
            img = matted.copy()
            img.thumbnail((size, size), Image.LANCZOS)
            canvas = Image.new('RGB', (size, size), (255, 255, 255))
            offset = ((size - img.size[0]) // 2, (size - img.size[1]) // 2)
            # cola usando alpha do matted como mascara
            canvas.paste(img.convert('RGB'), offset, img.split()[-1])
            out = self.output_dir / f'{key}_{size}.jpg'
            canvas.save(out, 'JPEG', quality=88, optimize=True)

        return True, {'sha': key, 'src': src_path}

if __name__ == '__main__':
    MattingBatch().run(status_every=50)
