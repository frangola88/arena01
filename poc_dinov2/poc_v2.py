"""POC v2 — DINOv2 retrieval-augmented matching, melhorias sobre poc.py.

Mudanças em relação à v1:
1. Collective 896×896  →  64×64 patch grid (4× mais patches, localização fina)
2. Canonical 448×448   →  encoder vê mais detalhe da peça
3. cluster_bbox em vez de peak_to_bbox — top-K% patches formam bbox de cluster,
   elimina outliers e produz bounding box mais compacto e fiel à peça real.
"""
import json, time
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from transformers import AutoImageProcessor, AutoModel

import faiss  # noqa: E402 — após numpy

ROOT       = Path.home() / 'projetos' / 'casaiq' / 'poc_dinov2'
CANONICAL  = ROOT / 'canonical'
OUTPUT     = ROOT / 'output'
CANONICAL.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)

GAZETTEER_EMB   = Path.home() / 'projetos/casaiq/embeddings.npy'
GAZETTEER_FAISS = Path.home() / 'projetos/casaiq/data/gazetteer/gazetteer.faiss'
GAZETTEER_MAP   = Path.home() / 'projetos/casaiq/data/gazetteer/index_mapping.jsonl'

CANON_SIZE      = 448   # resolução de entrada do canonical (v1 usava 224)
COLLECTIVE_SIZE = 896   # resolução do collective (v1 usava 448) → 64×64 patches
PATCH           = 14    # DINOv2 patch size fixo
PATCHES_SIDE    = COLLECTIVE_SIZE // PATCH  # 64
TOP_K_PCT       = 0.08  # top 8% dos patches para bbox de cluster
HEAD = {'User-Agent': 'Mozilla/5.0 Chrome/120.0'}

# ── 1. canonicals ──────────────────────────────────────────────────────────────

def load_canonicals():
    picks = json.loads((ROOT / 'picks.json').read_text())
    images = []
    for p in picks:
        dest = CANONICAL / f'{p["idx"]}.jpg'
        if not dest.exists():
            print(f'  baixando [{p["idx"]}] {p["nome"][:50]}')
            r = requests.get(p['img_url'], headers=HEAD, timeout=20)
            dest.write_bytes(r.content)
        img = Image.open(dest).convert('RGB')
        img.thumbnail((CANON_SIZE, CANON_SIZE), Image.LANCZOS)
        canvas = Image.new('RGB', (CANON_SIZE, CANON_SIZE), (255, 255, 255))
        canvas.paste(img, ((CANON_SIZE - img.size[0]) // 2, (CANON_SIZE - img.size[1]) // 2))
        images.append({'idx': p['idx'], 'nome': p['nome'], 'img': canvas, 'meta': p})
    return images

# ── 2. coletiva sintética ──────────────────────────────────────────────────────

def make_collective(canonicals, pick_indices=(0, 2, 4), seed=42):
    rng = np.random.RandomState(seed)
    sz = 896

    # fundo cinza com ruído leve
    bg = np.full((sz, sz, 3), 190, dtype=np.uint8)
    bg = np.clip(bg + rng.randint(-20, 20, bg.shape), 0, 255).astype(np.uint8)
    canvas = Image.fromarray(bg)

    # distractores coloridos
    draw = ImageDraw.Draw(canvas)
    for _ in range(6):
        x = int(rng.randint(0, sz - 120))
        y = int(rng.randint(0, sz - 120))
        s = int(rng.randint(50, 130))
        c = tuple(int(v) for v in rng.randint(60, 200, 3))
        draw.rectangle([x, y, x + s, y + s], fill=c)

    placed, truth = [], {}
    for idx in pick_indices:
        c = canonicals[idx]
        scale = float(rng.uniform(0.45, 0.75))
        w = int(CANON_SIZE * scale)
        item = c['img'].copy().resize((w, w), Image.LANCZOS)
        for _ in range(40):
            px = int(rng.randint(0, sz - w))
            py = int(rng.randint(0, sz - w))
            if not any(not (px + w < ox or ox + ow < px or py + w < oy or oy + oh < py)
                       for ox, oy, ow, oh in placed):
                break
        canvas.paste(item, (px, py))
        placed.append((px, py, w, w))
        truth[idx] = (px, py, px + w, py + w)
    return canvas, truth

# ── 3. encoder ────────────────────────────────────────────────────────────────

class Encoder:
    def __init__(self, model='facebook/dinov2-small'):
        print(f'\n[DINOv2] carregando {model}...')
        self.proc  = AutoImageProcessor.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model).eval()
        dim = self.model.config.hidden_size
        params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f'  dim={dim}  params={params:.1f}M  device=cpu')

    @torch.no_grad()
    def encode_global(self, img: Image.Image) -> np.ndarray:
        inp = self.proc(images=img.resize((CANON_SIZE, CANON_SIZE)), return_tensors='pt')
        out = self.model(**inp)
        cls = F.normalize(out.last_hidden_state[:, 0, :], dim=-1)
        return cls[0].numpy()

    @torch.no_grad()
    def encode_patches(self, img: Image.Image) -> np.ndarray:
        """Retorna (PATCHES_SIDE, PATCHES_SIDE, dim) com vetores L2-normalizados."""
        import torchvision.transforms.functional as TF
        # Prepara tensor manualmente — evita resize/crop do BitImageProcessor
        img_r = img.resize((COLLECTIVE_SIZE, COLLECTIVE_SIZE), Image.LANCZOS)
        t = TF.to_tensor(img_r)                                # (3, 896, 896) [0,1]
        t = TF.normalize(t, [0.485, 0.456, 0.406],            # ImageNet stats
                            [0.229, 0.224, 0.225]).unsqueeze(0)
        out = self.model(pixel_values=t, interpolate_pos_encoding=True)
        feats = out.last_hidden_state[:, 1:, :]          # skip CLS → (1, 4096, 384)
        feats = F.normalize(feats, dim=-1)
        n = feats.shape[1]
        s = int(np.sqrt(n))
        return feats[0].reshape(s, s, -1).numpy()        # (64, 64, 384)

# ── 4. cluster bbox ────────────────────────────────────────────────────────────

def cluster_bbox(heatmap: np.ndarray, canvas_size: int,
                 top_k_pct: float = TOP_K_PCT) -> tuple[tuple, float]:
    """
    Seleciona top-K% patches por score de cosine similarity.
    Calcula bbox que engloba esse cluster no espaço canvas.
    Muito mais robusto que pegar só o pico máximo.
    """
    H, W = heatmap.shape
    k = max(1, int(H * W * top_k_pct))
    flat = heatmap.flatten()
    top_idx = np.argpartition(flat, -k)[-k:]
    rows, cols = np.unravel_index(top_idx, (H, W))

    # patch center → canvas coords
    scale = canvas_size / (W * PATCH)
    r0, r1 = int(rows.min() * PATCH * scale), int((rows.max() + 1) * PATCH * scale)
    c0, c1 = int(cols.min() * PATCH * scale), int((cols.max() + 1) * PATCH * scale)
    r0, r1 = max(0, r0), min(canvas_size, r1)
    c0, c1 = max(0, c0), min(canvas_size, c1)
    peak = float(flat[top_idx].mean())
    return (c0, r0, c1, r1), peak

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / (ua + 1e-9)

# ── 5. gazetteer lookup ────────────────────────────────────────────────────────

class GazetteerLookup:
    """Carrega gazetteer.faiss + embeddings.npy e faz lookup por similaridade visual."""

    def __init__(self):
        print('\n[Gazetteer] carregando...')
        self.emb = np.load(GAZETTEER_EMB)                       # (18370, 384)
        self.idx = faiss.read_index(str(GAZETTEER_FAISS))       # IndexFlatIP ou similar
        with open(GAZETTEER_MAP) as f:
            self.mapping = [json.loads(l) for l in f]
        print(f'  {self.idx.ntotal} embeddings  dim={self.idx.d}')

    def lookup(self, query_vec: np.ndarray, k: int = 3):
        """
        Busca os k produtos mais similares no gazetteer.
        Retorna (top1_vec, hits) onde hits = [{'gaz_idx', 'key', 'dist'}, ...].
        """
        q = query_vec.astype('float32').reshape(1, -1)
        D, I = self.idx.search(q, k)
        hits = [
            {'gaz_idx': int(i), 'key': self.mapping[int(i)]['key'], 'dist': float(d)}
            for d, i in zip(D[0], I[0])
        ]
        top1_vec = self.emb[I[0][0]].astype('float32')
        return top1_vec, hits

# ── 6. main ────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print('=== POC v2 — DINOv2 896px + cluster bbox + gazetteer FAISS ===\n')

    print('[1/6] carregando canonicals')
    canonicals = load_canonicals()

    print('[2/6] montando coletiva sintética 896×896')
    collective, truth = make_collective(canonicals, pick_indices=(0, 2, 4))
    collective.save(OUTPUT / 'collective_v2.jpg')

    enc = Encoder()
    gaz = GazetteerLookup()

    print('\n[3/6] encodando canonicals (global, 448px)')
    live_vecs = [enc.encode_global(c['img']) for c in canonicals]

    print('[4/6] lookup no gazetteer (top-3 por cosine similarity)')
    gaz_vecs, gaz_hits = [], []
    for c, lv in zip(canonicals, live_vecs):
        top1_vec, hits = gaz.lookup(lv, k=3)
        gaz_vecs.append(top1_vec)
        gaz_hits.append(hits)
        print(f'  [{c["idx"]}] {c["nome"][:40]:40s}  '
              f'top1_dist={hits[0]["dist"]:.4f}  sha={hits[0]["key"][:12]}')

    print('\n[5/6] encodando coletiva (patches, 896px)')
    t_enc = time.time()
    grid = enc.encode_patches(collective)
    print(f'  patch grid: {grid.shape}  ({time.time()-t_enc:.1f}s)')

    print('[6/6] matching — modo LIVE vs modo GAZETTEER')
    results = []
    for c, live_vec, gaz_vec, hits in zip(canonicals, live_vecs, gaz_vecs, gaz_hits):
        gt = truth.get(c['idx'])

        hm_live = grid @ live_vec
        bbox_live, score_live = cluster_bbox(hm_live, collective.size[0])

        hm_gaz = grid @ gaz_vec
        bbox_gaz, score_gaz = cluster_bbox(hm_gaz, collective.size[0])

        results.append({
            'idx': c['idx'], 'nome': c['nome'],
            'in_scene': gt is not None,
            # live
            'peak_live': float(hm_live.max()),
            'score_live': score_live,
            'bbox_live': bbox_live,
            'iou_live': float(iou(bbox_live, gt)) if gt else None,
            'hm_live': hm_live,
            # gazetteer
            'peak_gaz': float(hm_gaz.max()),
            'score_gaz': score_gaz,
            'bbox_gaz': bbox_gaz,
            'iou_gaz': float(iou(bbox_gaz, gt)) if gt else None,
            'hm_gaz': hm_gaz,
            # lookup meta
            'gaz_dist': hits[0]['dist'],
            'gaz_key': hits[0]['key'],
            'bbox_truth': gt,
        })

    # ── visualização ──────────────────────────────────────────────────────────
    colors = ['#ff4444', '#44ff44', '#4488ff', '#ffaa00', '#aa44ff']

    # fig 1: coletiva com bbox LIVE vs GAZ lado a lado
    fig, (ax_live, ax_gaz) = plt.subplots(1, 2, figsize=(22, 10))
    for ax, mode, bbox_key, score_key in [
        (ax_live, 'LIVE (encode ao vivo)', 'bbox_live', 'iou_live'),
        (ax_gaz,  'GAZETTEER (lookup FAISS)', 'bbox_gaz', 'iou_gaz'),
    ]:
        ax.imshow(collective)
        ax.set_title(f'Modo {mode} — tracejado=verdade  sólido=predito', fontsize=11)
        ax.axis('off')
        for r in results:
            col = colors[r['idx']]
            if r['bbox_truth']:
                x1, y1, x2, y2 = r['bbox_truth']
                ax.add_patch(Rectangle((x1,y1), x2-x1, y2-y1,
                                       fill=False, edgecolor=col, lw=2, linestyle='--'))
                ax.text(x1, y1-6, f'TRUE[{r["idx"]}]', color=col, fontsize=8, weight='bold')
            x1, y1, x2, y2 = r[bbox_key]
            ax.add_patch(Rectangle((x1,y1), x2-x1, y2-y1,
                                   fill=False, edgecolor=col, lw=2))
            iou_val = r[score_key]
            lbl = f'[{r["idx"]}]' + (f' IoU={iou_val:.2f}' if iou_val is not None else '')
            ax.text(x1, y2+6, lbl, color=col, fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT / 'poc_v2_result.png', dpi=110, bbox_inches='tight')
    print(f'\n  → {OUTPUT / "poc_v2_result.png"}')

    # fig 2: heatmaps live (top) e gaz (bottom)
    fig2, axes = plt.subplots(2, 5, figsize=(22, 8))
    for i, r in enumerate(results):
        vmax = max(0.5, r['peak_live'], r['peak_gaz'])
        for row, hm, mode in [(0, r['hm_live'], 'LIVE'), (1, r['hm_gaz'], 'GAZ')]:
            axes[row][i].imshow(hm, cmap='hot', vmin=0, vmax=vmax)
            tag = 'YES' if r['in_scene'] else 'no'
            peak = r['peak_live'] if mode == 'LIVE' else r['peak_gaz']
            axes[row][i].set_title(f'[{r["idx"]}] {mode}  {tag}\npeak={peak:.2f}', fontsize=9)
            axes[row][i].axis('off')
    plt.tight_layout()
    plt.savefig(OUTPUT / 'poc_v2_heatmaps.png', dpi=100)
    print(f'  → {OUTPUT / "poc_v2_heatmaps.png"}')

    # ── relatório ─────────────────────────────────────────────────────────────
    print('\n' + '='*80)
    print(f'Modelo: DINOv2-small  |  collective={COLLECTIVE_SIZE}px  '
          f'canon={CANON_SIZE}px  patches={PATCHES_SIDE}x{PATCHES_SIDE}')
    print(f'Tempo total: {time.time()-t0:.1f}s\n')
    print(f'{"idx":>3}  {"cena":>4}  {"gaz_dist":>8}  '
          f'{"peak_L":>6}  {"IoU_L":>5}  {"peak_G":>6}  {"IoU_G":>5}  nome')
    print('-' * 80)
    present, absent = [], []
    for r in results:
        il = f'{r["iou_live"]:.3f}' if r['iou_live'] is not None else '  n/a'
        ig = f'{r["iou_gaz"]:.3f}'  if r['iou_gaz']  is not None else '  n/a'
        tag = 'YES' if r['in_scene'] else ' no'
        print(f'{r["idx"]:>3}  {tag:>4}  {r["gaz_dist"]:>8.4f}  '
              f'{r["peak_live"]:>6.3f}  {il:>5}  {r["peak_gaz"]:>6.3f}  {ig:>5}  '
              f'{r["nome"][:35]}')
        (present if r['in_scene'] else absent).append(r)

    def avg(lst): return np.mean(lst) if lst else 0.0

    pr_iou_l = avg([r['iou_live'] for r in present if r['iou_live'] is not None])
    pr_iou_g = avg([r['iou_gaz']  for r in present if r['iou_gaz']  is not None])
    pr_pk_l  = avg([r['peak_live'] for r in present])
    pr_pk_g  = avg([r['peak_gaz']  for r in present])
    ab_pk_l  = avg([r['peak_live'] for r in absent])
    ab_pk_g  = avg([r['peak_gaz']  for r in absent])

    def tag_sep(v):  return '🟢 boa' if v > 0.10 else '🟡 marginal' if v > 0.05 else '🔴 ruim'
    def tag_iou(v):  return '🟢 >0.30' if v > 0.30 else '🟡 0.15-0.30' if v > 0.15 else '🔴 <0.15'

    print()
    print(f'{"":30s}  {"LIVE":>8}  {"GAZETTEER":>10}')
    print(f'  Avg peak presentes     :  {pr_pk_l:>8.3f}  {pr_pk_g:>10.3f}')
    print(f'  Avg peak ausentes      :  {ab_pk_l:>8.3f}  {ab_pk_g:>10.3f}')
    print(f'  Separação              :  {pr_pk_l-ab_pk_l:>8.3f} {tag_sep(pr_pk_l-ab_pk_l)}'
          f'   {pr_pk_g-ab_pk_g:>6.3f} {tag_sep(pr_pk_g-ab_pk_g)}')
    print(f'  Avg IoU presentes      :  {pr_iou_l:>8.3f} {tag_iou(pr_iou_l)}'
          f'   {pr_iou_g:>6.3f} {tag_iou(pr_iou_g)}')
    print('='*80)

if __name__ == '__main__':
    main()
