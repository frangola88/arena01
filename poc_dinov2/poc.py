"""POC retrieval-augmented matching com DINOv2.

Fluxo:
1) Pega 5 produtos do dataset (canônicos)
2) Baixa as imagens canônicas
3) Monta uma "foto coletiva" sintética colando 3 delas num fundo bagunçado
4) Encoda canônicos (CLS feature global, 384-dim) e coletiva (patch tokens, grid)
5) Pra cada canônico, faz cosine similarity com cada patch → heatmap
6) Acha pico do heatmap → bounding box do match
7) Visualiza: coletiva com bbox + canônicos + heatmaps

Saída: ~/projetos/casaiq/poc_dinov2/output/poc_result.png + relatório txt
"""
import json, os, random, sys, time
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from transformers import AutoImageProcessor, AutoModel

ROOT = Path.home() / 'projetos' / 'casaiq' / 'poc_dinov2'
CANONICAL = ROOT / 'canonical'
OUTPUT = ROOT / 'output'
CANONICAL.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)

CANON_SIZE = 224          # DINOv2 nativo
COLLECTIVE_SIZE = 448     # 2x — 32x32 patches no grid
PATCH = 14                # DINOv2 patch size
PATCHES_PER_SIDE = COLLECTIVE_SIZE // PATCH  # 32
HEAD = {'User-Agent': 'Mozilla/5.0 Chrome/120.0'}

UA_DEVICE = 'cpu'  # roda em CPU, ~1-2s/encoding

# ============ Step 1: download canonicals ============
def download_canonicals():
    picks = json.loads((ROOT / 'picks.json').read_text())
    images = []
    for p in picks:
        dest = CANONICAL / f'{p["idx"]}.jpg'
        if not dest.exists():
            print(f'  baixando [{p["idx"]}] {p["nome"][:50]}')
            r = requests.get(p['img_url'], headers=HEAD, timeout=20)
            dest.write_bytes(r.content)
        img = Image.open(dest).convert('RGB')
        # canônica = resize pra 224x224 com pad branco mantendo aspecto
        img.thumbnail((CANON_SIZE, CANON_SIZE), Image.LANCZOS)
        canvas = Image.new('RGB', (CANON_SIZE, CANON_SIZE), (255, 255, 255))
        canvas.paste(img, ((CANON_SIZE - img.size[0]) // 2, (CANON_SIZE - img.size[1]) // 2))
        images.append({'idx': p['idx'], 'nome': p['nome'], 'img': canvas, 'meta': p})
    return images

# ============ Step 2: build synthetic collective ============
def make_collective(canonicals, pick_indices=(0, 2, 4)):
    """Cria uma foto 'coletiva' sintética 768x768 com 3 dos canônicos colados.

    Retorna a imagem e a lista de bboxes verdadeiros (pra avaliar precisão).
    """
    rng = random.Random(42)
    canvas_size = 768
    # Fundo cinza com ruído moderado
    canvas = Image.new('RGB', (canvas_size, canvas_size), (200, 200, 195))
    noise = np.random.RandomState(42).randint(-25, 25, (canvas_size, canvas_size, 3), dtype=np.int16)
    arr = np.clip(np.array(canvas).astype(np.int16) + noise, 0, 255).astype(np.uint8)
    canvas = Image.fromarray(arr)

    # Algumas "ferramentas distratoras" — quadrados coloridos
    draw = ImageDraw.Draw(canvas)
    for _ in range(5):
        x = rng.randint(0, canvas_size - 120)
        y = rng.randint(0, canvas_size - 120)
        size = rng.randint(60, 140)
        color = (rng.randint(80, 180), rng.randint(80, 180), rng.randint(80, 180))
        draw.rectangle([x, y, x + size, y + size], fill=color)

    # Cola 3 canônicos em posições aleatórias mas não-sobrepostas
    placed = []
    truth_bboxes = []
    for idx in pick_indices:
        c = canonicals[idx]
        # Escala aleatória pra simular distância
        scale = rng.uniform(0.5, 0.85)
        target_w = int(CANON_SIZE * scale)
        item_img = c['img'].copy().resize((target_w, target_w), Image.LANCZOS)
        # acha posição que não sobrepõe muito
        for _ in range(30):
            px = rng.randint(0, canvas_size - target_w)
            py = rng.randint(0, canvas_size - target_w)
            overlap = any(not (px + target_w < ox or ox + ow < px or py + target_w < oy or oy + ow < py)
                          for ox, oy, ow in placed)
            if not overlap:
                break
        canvas.paste(item_img, (px, py))
        placed.append((px, py, target_w))
        truth_bboxes.append({'idx': idx, 'bbox': (px, py, px + target_w, py + target_w), 'nome': c['nome']})

    return canvas, truth_bboxes

# ============ Step 3: DINOv2 encoder ============
class Encoder:
    def __init__(self, model_name='facebook/dinov2-small'):
        print(f'\n[DINOv2] carregando {model_name}...')
        self.proc = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(UA_DEVICE).eval()
        self.dim = self.model.config.hidden_size
        print(f'  dim={self.dim} | params={sum(p.numel() for p in self.model.parameters())/1e6:.1f}M')

    @torch.no_grad()
    def encode_global(self, img: Image.Image) -> np.ndarray:
        """Vetor único 384-dim (CLS token)."""
        inputs = self.proc(images=img.resize((CANON_SIZE, CANON_SIZE)), return_tensors='pt').to(UA_DEVICE)
        out = self.model(**inputs)
        cls = out.last_hidden_state[:, 0, :]   # (1, 384)
        cls = F.normalize(cls, dim=-1)
        return cls[0].cpu().numpy()

    @torch.no_grad()
    def encode_patches(self, img: Image.Image, side: int = COLLECTIVE_SIZE) -> np.ndarray:
        """Grid de 384-dim vetores (PATCHES_PER_SIDE x PATCHES_PER_SIDE x 384)."""
        img_r = img.resize((side, side), Image.LANCZOS)
        inputs = self.proc(images=img_r, return_tensors='pt', do_resize=False).to(UA_DEVICE)
        out = self.model(**inputs, interpolate_pos_encoding=True)
        feats = out.last_hidden_state[:, 1:, :]   # (1, N, 384) — skip CLS
        feats = F.normalize(feats, dim=-1)
        n = feats.shape[1]
        side_p = int(np.sqrt(n))
        feats = feats.reshape(1, side_p, side_p, -1)
        return feats[0].cpu().numpy()  # (side_p, side_p, 384)

# ============ Step 4: matching + bbox ============
def cosine_heatmap(canonical_vec: np.ndarray, patch_grid: np.ndarray) -> np.ndarray:
    """canonical_vec: (384,)  patch_grid: (H,W,384)  → heatmap (H,W) cosine sim."""
    # Já estão L2-normalizados
    return patch_grid @ canonical_vec

def peak_to_bbox(heatmap: np.ndarray, canvas_size: int, win: int = 4):
    """Pega top-1 e retorna bbox estimado na resolução do canvas."""
    H, W = heatmap.shape
    py, px = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    # Janela em torno do pico (4 patches × 14px × scale)
    scale = canvas_size / (W * PATCH)
    cx = int((px + 0.5) * PATCH * scale)
    cy = int((py + 0.5) * PATCH * scale)
    half = int(win * PATCH * scale)
    bbox = (max(0, cx - half), max(0, cy - half),
            min(canvas_size, cx + half), min(canvas_size, cy + half))
    return bbox, heatmap.max()

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)

# ============ Step 5: visualization + run ============
def main():
    t0 = time.time()
    print('=== POC retrieval-augmented matching com DINOv2 ===\n')

    print('[1/5] download canonicals')
    canonicals = download_canonicals()

    print('[2/5] make synthetic collective')
    collective_img, truth = make_collective(canonicals, pick_indices=(0, 2, 4))
    collective_img.save(OUTPUT / 'collective.jpg')
    print(f'  3 canônicos colados; saved {OUTPUT / "collective.jpg"}')

    enc = Encoder()

    print('[3/5] encode canonicals (global)')
    canonical_vecs = []
    for c in canonicals:
        v = enc.encode_global(c['img'])
        canonical_vecs.append(v)

    print('[4/5] encode collective (patches)')
    coll_grid = enc.encode_patches(collective_img, side=COLLECTIVE_SIZE)
    print(f'  patch grid: {coll_grid.shape}')

    print('[5/5] matching')
    canvas_size = collective_img.size[0]
    results = []
    truth_lookup = {t['idx']: t['bbox'] for t in truth}
    for c, vec in zip(canonicals, canonical_vecs):
        hm = cosine_heatmap(vec, coll_grid)
        bbox, peak = peak_to_bbox(hm, canvas_size)
        gt = truth_lookup.get(c['idx'])
        score = iou(bbox, gt) if gt else None
        in_collective = gt is not None
        results.append({
            'idx': c['idx'], 'nome': c['nome'], 'in_collective': in_collective,
            'peak_score': float(peak), 'bbox_pred': bbox, 'bbox_truth': gt,
            'iou': float(score) if score is not None else None,
            'heatmap': hm,
        })

    # ============ visualization ============
    fig = plt.figure(figsize=(18, 14))
    # Top row: 1 large coletiva + bboxes
    ax = plt.subplot2grid((3, 5), (0, 0), colspan=5, rowspan=2)
    ax.imshow(collective_img)
    ax.set_title('Coletiva sintética — bbox vermelho = verdade, verde = predito DINOv2', fontsize=11)
    ax.axis('off')
    colors = ['#ff4444', '#44ff44', '#4488ff', '#ffaa00', '#aa44ff']
    for r in results:
        if r['bbox_truth']:
            x1, y1, x2, y2 = r['bbox_truth']
            ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=2, linestyle='--'))
            ax.text(x1, y1-5, f'TRUTH [{r["idx"]}]', color='red', fontsize=9, weight='bold')
        x1, y1, x2, y2 = r['bbox_pred']
        ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=colors[r['idx']], linewidth=2))
        score_str = f'peak={r["peak_score"]:.2f}'
        if r['iou'] is not None: score_str += f' IoU={r["iou"]:.2f}'
        ax.text(x1, y2+5, f'[{r["idx"]}] {score_str}', color=colors[r['idx']], fontsize=9)

    # Bottom row: 5 canônicos + heatmaps
    for i, r in enumerate(results):
        ax_c = plt.subplot2grid((3, 5), (2, i))
        ax_c.imshow(canonicals[i]['img'])
        in_str = '✅ na cena' if r['in_collective'] else '❌ NÃO na cena'
        title = f"[{r['idx']}] {canonicals[i]['nome'][:25]}\n{in_str} peak={r['peak_score']:.2f}"
        ax_c.set_title(title, fontsize=8)
        ax_c.axis('off')

    plt.tight_layout()
    out_png = OUTPUT / 'poc_result.png'
    plt.savefig(out_png, dpi=110, bbox_inches='tight')
    print(f'\n[viz] {out_png}')

    # heatmap detail
    fig2, axes = plt.subplots(1, 5, figsize=(20, 4.5))
    for i, r in enumerate(results):
        axes[i].imshow(r['heatmap'], cmap='hot', vmin=0, vmax=max(0.8, r['peak_score']))
        in_str = '✅' if r['in_collective'] else '❌'
        axes[i].set_title(f"[{r['idx']}] {in_str} peak={r['peak_score']:.2f}", fontsize=10)
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(OUTPUT / 'heatmaps.png', dpi=100)
    print(f'[viz] {OUTPUT / "heatmaps.png"}')

    # report
    lines = ['=== POC DINOv2 — Relatório ===\n']
    lines.append(f'Modelo: facebook/dinov2-small (~22M params, CPU)')
    lines.append(f'Foto coletiva: 768x768, 3 canônicos colados em escala 0.5-0.85x')
    lines.append(f'Patch grid no encoder: {coll_grid.shape[:2]} ({COLLECTIVE_SIZE}x{COLLECTIVE_SIZE} input)')
    lines.append(f'Tempo total: {time.time() - t0:.1f}s\n')
    lines.append(f'{"idx":>3} | {"in scene":>9} | {"peak score":>10} | {"IoU":>5} | nome')
    lines.append('-' * 90)
    for r in results:
        iou_str = f'{r["iou"]:.2f}' if r['iou'] is not None else 'n/a'
        in_str = 'YES' if r['in_collective'] else 'no'
        lines.append(f'{r["idx"]:>3} | {in_str:>9} | {r["peak_score"]:>10.3f} | {iou_str:>5} | {r["nome"][:60]}')
    lines.append('')

    # Heurística de sucesso
    true_present = [r for r in results if r['in_collective']]
    true_absent = [r for r in results if not r['in_collective']]
    avg_peak_present = np.mean([r['peak_score'] for r in true_present])
    avg_peak_absent = np.mean([r['peak_score'] for r in true_absent]) if true_absent else 0
    avg_iou = np.mean([r['iou'] for r in true_present if r['iou'] is not None])
    lines.append(f'Avg peak (canônicos PRESENTES): {avg_peak_present:.3f}')
    lines.append(f'Avg peak (canônicos AUSENTES):  {avg_peak_absent:.3f}')
    lines.append(f'Avg IoU dos presentes:          {avg_iou:.3f}')
    sep = avg_peak_present - avg_peak_absent
    lines.append(f'Separação (presente − ausente): {sep:.3f}  ' +
                 ('🟢 boa' if sep > 0.1 else '🟡 marginal' if sep > 0.05 else '🔴 ruim'))
    lines.append(f'IoU médio: {avg_iou:.3f}  ' +
                 ('🟢 funcional' if avg_iou > 0.3 else '🟡 marginal' if avg_iou > 0.15 else '🔴 ruim'))

    report = '\n'.join(lines)
    (OUTPUT / 'report.txt').write_text(report)
    print('\n' + report)

if __name__ == '__main__':
    main()
