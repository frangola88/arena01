"""
poc_dinov2/demo_grid_celulas.py — malha 100x100 de células com vizinhos
conhecidos, classificação ferramenta/fundo por célula (coerência de
gradiente + Otsu) e objetos = componentes conexos do grafo de células.

Sem reconstrução de superfície, sem griddata, sem sigmoid, sem
binary_closing — a malha é classificada inteira (densa), não amostrada
esparsamente e depois interpolada. Isso elimina o mecanismo que mais
contribuía pra fundir objetos próximos no pipeline atual.

Gera também a imagem de overlay (fundo neutro + contornos numerados dos
componentes "ferramenta"), pra avaliar visualmente antes de decidir se
vale mandar como segunda imagem numa chamada real à API.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import label, find_objects, binary_closing
from skimage.filters import threshold_otsu

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.analise_cena import _carregar_cinza  # noqa: E402

N_ROWS, N_COLS = 100, 100
AREA_MIN_CELULAS = 3  # nº mínimo de células p/ contar como objeto
CLOSING_ITER = 2      # bridga as lacunas do contorno (coerência acende na
                       # borda, fica escura no interior liso da ferramenta —
                       # sem isso cada ferramenta fragmenta em vários traços)


def grid_coerencia(img_gray: np.ndarray, n_rows: int = N_ROWS, n_cols: int = N_COLS):
    """Sinal de coerência de gradiente (energia x anisotropia) calculado
    densamente em TODA a malha n_rows x n_cols — sem amostragem esparsa."""
    from scipy.ndimage import sobel
    Ix = sobel(img_gray.astype(np.float32), axis=1)
    Iy = sobel(img_gray.astype(np.float32), axis=0)
    Ixx, Iyy, Ixy = Ix * Ix, Iy * Iy, Ix * Iy

    H, W = img_gray.shape
    bh, bw = H // n_rows, W // n_cols

    def block_mean(a):
        a = a[:n_rows * bh, :n_cols * bw].reshape(n_rows, bh, n_cols, bw)
        return a.mean(axis=(1, 3))

    Jxx, Jyy, Jxy = block_mean(Ixx), block_mean(Iyy), block_mean(Ixy)
    trace = Jxx + Jyy
    disc = np.sqrt(np.clip((trace / 2) ** 2 - (Jxx * Jyy - Jxy ** 2), 0, None))
    lam1, lam2 = trace / 2 + disc, trace / 2 - disc
    energia = lam1 + lam2
    coerencia = (lam1 - lam2) / (lam1 + lam2 + 1e-8)
    sinal = (energia * coerencia).astype(np.float32)
    return sinal, (bh, bw)


def classificar_celulas(sinal: np.ndarray):
    """Otsu sobre o histograma de coerência das células — sem percentil fixo."""
    th = threshold_otsu(sinal.ravel())
    mask_ferramenta = sinal > th
    return mask_ferramenta, float(th)


def componentes(mask: np.ndarray, area_min: int):
    """Componentes conexos (8-conectividade) no grafo de células."""
    estrutura = np.ones((3, 3), dtype=int)  # 8-conectividade
    labeled, n = label(mask, structure=estrutura)
    slices = find_objects(labeled)
    objs = []
    for i, sl in enumerate(slices, start=1):
        area = int((labeled[sl] == i).sum())
        if area < area_min:
            continue
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        objs.append({"id": i, "bbox_celulas": (y0, y1, x0, x1), "area_celulas": area})
    return objs, labeled


def fundo_toca_borda(labeled_bg: np.ndarray, n_bg: int):
    Hb, Wb = labeled_bg.shape
    borda_labels = set(labeled_bg[0, :]) | set(labeled_bg[-1, :]) | \
        set(labeled_bg[:, 0]) | set(labeled_bg[:, -1])
    borda_labels.discard(0)
    return borda_labels


def gerar_overlay(img_shape_px, objs, bh, bw, escala_overlay=0.25):
    """Imagem separada (fundo neutro escuro) com contornos numerados de cada
    objeto candidato — pra mandar como 2ª imagem de referência, sem alterar
    a foto real."""
    H, W = img_shape_px
    out_h, out_w = int(H * escala_overlay), int(W * escala_overlay)
    overlay = np.full((out_h, out_w, 3), 20, dtype=np.uint8)  # fundo cinza-escuro neutro

    sy, sx = escala_overlay, escala_overlay
    for i, o in enumerate(objs, start=1):
        y0, y1, x0, x1 = o["bbox_celulas"]
        px0, px1 = int(x0 * bw * sx), int(x1 * bw * sx)
        py0, py1 = int(y0 * bh * sy), int(y1 * bh * sy)
        cv2.rectangle(overlay, (px0, py0), (px1, py1), (60, 200, 60), 2)
        cx, cy = (px0 + px1) // 2, max(15, py0 + 15)
        cv2.putText(overlay, str(i), (cx - 6, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return overlay


def avaliar(caminho: str):
    print(f"\n=== {Path(caminho).name} ===")
    img_gray = _carregar_cinza(caminho)
    H, W = img_gray.shape

    sinal, (bh, bw) = grid_coerencia(img_gray, N_ROWS, N_COLS)
    mask_ferr, th = classificar_celulas(sinal)
    mask_ferr = binary_closing(mask_ferr, structure=np.ones((3, 3)), iterations=CLOSING_ITER)
    objs, labeled = componentes(mask_ferr, AREA_MIN_CELULAS)
    objs_bg, labeled_bg = componentes(~mask_ferr, area_min=1)

    print(f"  malha: {N_ROWS}x{N_COLS} células ({bh}x{bw}px/célula)")
    print(f"  otsu threshold: {th:.4f}")
    print(f"  n_objetos (componentes ferramenta, area>={AREA_MIN_CELULAS} células): {len(objs)}")
    print(f"  n_componentes_fundo: {len(objs_bg)}  "
          f"(maior: {max((o['area_celulas'] for o in objs_bg), default=0)} células)")
    for i, o in enumerate(objs, start=1):
        print(f"    obj {i}: área={o['area_celulas']:3d} células  bbox_celulas={o['bbox_celulas']}")

    # visual: máscara binária upsampled
    from PIL import Image
    mask_vis = (mask_ferr.astype(np.uint8) * 255)
    mask_img = Image.fromarray(mask_vis).resize((W // 4, H // 4), Image.NEAREST)
    stem = Path(caminho).stem
    mask_img.save(f"/tmp/grid_mask_{stem}.png")

    overlay = gerar_overlay((H, W), objs, bh, bw, escala_overlay=0.25)
    Image.fromarray(overlay).save(f"/tmp/grid_overlay_{stem}.png")
    print(f"  máscara: /tmp/grid_mask_{stem}.png")
    print(f"  overlay: /tmp/grid_overlay_{stem}.png")


if __name__ == "__main__":
    alvos = sys.argv[1:] or [
        "data/fotos_mestras/IMG_mestra04semflash.jpg",
        "data/fotos_mestras/IMG_mestra04comflash.jpg",
    ]
    for a in alvos:
        avaliar(a)
