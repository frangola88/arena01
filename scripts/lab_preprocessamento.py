"""
Laboratório de pré-processamento — compara 8 técnicas lado a lado.

Cada técnica:
  1. Transforma a imagem
  2. Detecta contornos
  3. Desenha bbox dos contornos válidos (filtrados por área)
  4. Anota contagem

Saída: imagem em grade 2x4 (1024 x 1024 ou similar) em /tmp/lab_preproc.jpg
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from PIL import Image, ImageOps

# Filtros de área (% da imagem) — mesmos do pipeline real
AREA_MIN_PCT = 0.015
AREA_MAX_PCT = 0.45


def _detectar_contornos_no_binario(binario: np.ndarray, area_total: int) -> tuple[int, np.ndarray]:
    """
    Recebe imagem binária (0/255), retorna (n_contornos_validos, imagem_anotada).
    """
    contornos, _ = cv2.findContours(binario, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_min = area_total * AREA_MIN_PCT
    area_max = area_total * AREA_MAX_PCT
    validos = []
    for c in contornos:
        a = cv2.contourArea(c)
        if area_min <= a <= area_max:
            validos.append(c)
    return len(validos), validos


def _anotar(img_color: np.ndarray, contornos: list, titulo: str, info: str) -> np.ndarray:
    """Desenha contornos + título em uma cópia da imagem colorida."""
    out = img_color.copy()
    for i, c in enumerate(contornos):
        cv2.drawContours(out, [c], -1, (0, 255, 0), 3)
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 100, 0), 2)
        cv2.putText(out, f"#{i+1}", (x + 5, y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Faixa de título no topo
    h_img = out.shape[0]
    cv2.rectangle(out, (0, 0), (out.shape[1], 60), (0, 0, 0), -1)
    cv2.putText(out, titulo, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(out, info, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


# ───────────── 8 TÉCNICAS ─────────────

def tec_baseline(img_bgr: np.ndarray) -> tuple[np.ndarray, list, str]:
    """Pipeline atual: grayscale + blur + Canny."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges_d = cv2.dilate(edges, np.ones((10, 10), np.uint8), iterations=2)
    edges_d = cv2.morphologyEx(edges_d, cv2.MORPH_CLOSE, np.ones((10, 10), np.uint8))
    n, contornos = _detectar_contornos_no_binario(edges_d, img_bgr.shape[0] * img_bgr.shape[1])
    return img_bgr, contornos, f"[BASELINE atual] {n} contorno(s)"


def tec_clahe(img_bgr: np.ndarray) -> tuple[np.ndarray, list, str]:
    """CLAHE: contraste local adaptativo, depois Canny."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges_d = cv2.dilate(edges, np.ones((10, 10), np.uint8), iterations=2)
    edges_d = cv2.morphologyEx(edges_d, cv2.MORPH_CLOSE, np.ones((10, 10), np.uint8))
    n, contornos = _detectar_contornos_no_binario(edges_d, img_bgr.shape[0] * img_bgr.shape[1])
    return img_bgr, contornos, f"[CLAHE + Canny] {n} contorno(s)"


def tec_hist_equalizado(img_bgr: np.ndarray) -> tuple[np.ndarray, list, str]:
    """Equalização global de histograma + Canny."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(eq, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges_d = cv2.dilate(edges, np.ones((10, 10), np.uint8), iterations=2)
    edges_d = cv2.morphologyEx(edges_d, cv2.MORPH_CLOSE, np.ones((10, 10), np.uint8))
    n, contornos = _detectar_contornos_no_binario(edges_d, img_bgr.shape[0] * img_bgr.shape[1])
    return img_bgr, contornos, f"[Hist Equalizado] {n} contorno(s)"


def tec_adaptive_threshold(img_bgr: np.ndarray) -> tuple[np.ndarray, list, str]:
    """Adaptive threshold gaussiano — binariza por janela local."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binario = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 21, 5)
    binario = cv2.morphologyEx(binario, cv2.MORPH_CLOSE, np.ones((10, 10), np.uint8))
    n, contornos = _detectar_contornos_no_binario(binario, img_bgr.shape[0] * img_bgr.shape[1])
    return img_bgr, contornos, f"[Adaptive Threshold] {n} contorno(s)"


def tec_otsu(img_bgr: np.ndarray) -> tuple[np.ndarray, list, str]:
    """Otsu binario — encontra threshold global automaticamente."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binario = cv2.threshold(blurred, 0, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binario = cv2.morphologyEx(binario, cv2.MORPH_CLOSE, np.ones((10, 10), np.uint8))
    n, contornos = _detectar_contornos_no_binario(binario, img_bgr.shape[0] * img_bgr.shape[1])
    return img_bgr, contornos, f"[Otsu] {n} contorno(s)"


def tec_bilateral(img_bgr: np.ndarray) -> tuple[np.ndarray, list, str]:
    """Bilateral filter (preserva bordas, suaviza textura) + Canny."""
    bilat = cv2.bilateralFilter(img_bgr, d=9, sigmaColor=75, sigmaSpace=75)
    gray = cv2.cvtColor(bilat, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_d = cv2.dilate(edges, np.ones((10, 10), np.uint8), iterations=2)
    edges_d = cv2.morphologyEx(edges_d, cv2.MORPH_CLOSE, np.ones((10, 10), np.uint8))
    n, contornos = _detectar_contornos_no_binario(edges_d, img_bgr.shape[0] * img_bgr.shape[1])
    return img_bgr, contornos, f"[Bilateral + Canny] {n} contorno(s)"


def tec_canal_L_lab(img_bgr: np.ndarray) -> tuple[np.ndarray, list, str]:
    """Canal L do espaço LAB (luminância pura) + CLAHE + Canny."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    L_eq = clahe.apply(L)
    blurred = cv2.GaussianBlur(L_eq, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges_d = cv2.dilate(edges, np.ones((10, 10), np.uint8), iterations=2)
    edges_d = cv2.morphologyEx(edges_d, cv2.MORPH_CLOSE, np.ones((10, 10), np.uint8))
    n, contornos = _detectar_contornos_no_binario(edges_d, img_bgr.shape[0] * img_bgr.shape[1])
    return img_bgr, contornos, f"[LAB-L + CLAHE + Canny] {n} contorno(s)"


def tec_canny_dual(img_bgr: np.ndarray) -> tuple[np.ndarray, list, str]:
    """Canny na imagem original UNIDO com Canny na invertida (pega claro E escuro)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    invertida = cv2.bitwise_not(blurred)
    edges1 = cv2.Canny(blurred, 50, 150)
    edges2 = cv2.Canny(invertida, 50, 150)
    edges = cv2.bitwise_or(edges1, edges2)
    edges_d = cv2.dilate(edges, np.ones((10, 10), np.uint8), iterations=2)
    edges_d = cv2.morphologyEx(edges_d, cv2.MORPH_CLOSE, np.ones((10, 10), np.uint8))
    n, contornos = _detectar_contornos_no_binario(edges_d, img_bgr.shape[0] * img_bgr.shape[1])
    return img_bgr, contornos, f"[Canny Dual orig+inv] {n} contorno(s)"


# ───────────── ORQUESTRAÇÃO ─────────────

def main():
    foto = "/home/cuco/projetos/casaiq/storage/fotos_originais/20260513_235112_376761.png"
    if not Path(foto).exists():
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            for p in Path(foto).parent.glob(ext):
                if "_debug" not in p.name and "_anotada" not in p.name:
                    foto = str(p); break

    print(f"Lab pré-processamento em: {foto}\n")

    pil = Image.open(foto)
    pil = ImageOps.exif_transpose(pil)
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    tecnicas = [
        tec_baseline, tec_clahe, tec_hist_equalizado, tec_adaptive_threshold,
        tec_otsu, tec_bilateral, tec_canal_L_lab, tec_canny_dual,
    ]

    paineis = []
    for tec in tecnicas:
        try:
            img_ref, contornos, info = tec(img_bgr)
            print(f"  {info}")
            painel = _anotar(img_ref, contornos, tec.__name__, info)
            paineis.append(painel)
        except Exception as e:
            print(f"  ❌ {tec.__name__} falhou: {e}")
            paineis.append(img_bgr.copy())

    # Redimensionar todos para o mesmo tamanho
    h, w = paineis[0].shape[:2]
    paineis_redim = [cv2.resize(p, (w, h)) for p in paineis]

    # Grade 2 linhas x 4 colunas
    linha1 = np.hstack(paineis_redim[:4])
    linha2 = np.hstack(paineis_redim[4:])
    grade = np.vstack([linha1, linha2])

    saida = "/tmp/lab_preproc.jpg"
    cv2.imwrite(saida, grade)
    print(f"\n✅ Comparativo salvo em: {saida}")
    print(f"   Dimensões: {grade.shape[1]}x{grade.shape[0]} px")


if __name__ == "__main__":
    main()
