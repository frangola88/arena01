"""
Segmentação visual com OpenCV — detecta objetos por contornos.

Estratégia:
1. Converte imagem para grayscale
2. Aplica desfoque gaussiano (suaviza textura da mesa/piso)
3. Detecta bordas com Canny
4. Dilata bordas para fechar contornos
5. Encontra contornos externos
6. Filtra por área (descarta ruído pequeno e mesa gigante)
7. Gera bbox + recortes
8. Salva visualização de debug com cores

Cores do debug:
  🟢 VERDE GROSSO  = contorno escolhido (vira recorte)
  🔵 AZUL          = bbox final (recortada com margem)
  ⚪ CINZA         = descartado por ser muito pequeno (ruído)
  🔴 VERMELHO      = descartado por ser muito grande (provavelmente mesa/fundo)
"""
import logging
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps

_log = logging.getLogger("casaiq.segmentacao_cv")


# ─── Parâmetros ajustáveis ────────────────────────────────────────────────────
# Calibrados empiricamente em foto real (chaves de fenda + régua sobre mesa).
# K10 + 1.5% min detectou exatamente 3 objetos sem falsos positivos.

# Limiares Canny: low e high. Quanto MENOR low, mais bordas detectadas (mais ruído).
CANNY_LOW = 50
CANNY_HIGH = 150

# Kernel de dilatação (fecha contornos abertos do mesmo objeto).
# Calibrado: 10 junta partes do mesmo objeto sem fundir objetos próximos.
KERNEL_DILATACAO = 10

# Filtros de área (% da imagem total)
AREA_MIN_PCT = 0.015   # 1.5% — descarta números impressos, marcações da régua, ruído
AREA_MAX_PCT = 0.45    # 45%  — acima disso é provavelmente mesa/parede/fundo

# Margem ao recortar (pixels) — captura um pouco do entorno do objeto
# Pequena: recorte apertado evita capturar régua/fundo atrás
MARGEM_RECORTE = 8

# Non-Maximum Suppression: se contorno A contém B em mais que isso, descarta A
# (mantém o filho específico, joga fora o pai genérico)
NMS_OVERLAP_THRESHOLD = 0.60


def _carregar_imagem_orientada(caminho: str) -> tuple[np.ndarray, Image.Image]:
    """
    Carrega imagem aplicando rotação EXIF, retorna (array_cv2, pil_image).
    
    Returns:
        (np_bgr, pil_rgb): array OpenCV BGR + objeto PIL RGB (para recortes finos).
    """
    pil = Image.open(caminho)
    pil = ImageOps.exif_transpose(pil)
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    
    # PIL é RGB, cv2 trabalha em BGR
    np_rgb = np.array(pil)
    np_bgr = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR)
    return np_bgr, pil


def detectar_objetos(
    caminho_foto: str,
    debug_path: str | None = None,
) -> list[dict]:
    """
    Detecta objetos na foto usando contornos OpenCV.
    
    Args:
        caminho_foto: caminho da imagem original
        debug_path: se fornecido, salva visualização de debug aqui
    
    Returns:
        Lista de dicts com:
        - "bbox": (x, y, w, h) em pixels
        - "area_pct": fração da imagem ocupada
        - "centro": (cx, cy) centro do objeto
    """
    img_bgr, pil_img = _carregar_imagem_orientada(caminho_foto)
    h_img, w_img = img_bgr.shape[:2]
    area_total = h_img * w_img
    
    # ─── 1. Pré-processamento (Bilateral filter) ───
    # bilateralFilter suaviza a textura do fundo (veios da mesa de madeira)
    # PRESERVANDO as bordas dos objetos. Escolhido empiricamente no laboratório
    # de pré-processamento — superou GaussianBlur, CLAHE, Otsu e adaptive threshold.
    bilateral = cv2.bilateralFilter(img_bgr, d=9, sigmaColor=75, sigmaSpace=75)
    gray = cv2.cvtColor(bilateral, cv2.COLOR_BGR2GRAY)
    
    # ─── 2. Detecção de bordas ───
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    
    # ─── 3. Dilatação para fechar contornos ───
    kernel = np.ones((KERNEL_DILATACAO, KERNEL_DILATACAO), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)
    edges_closed = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel)
    
    # ─── 4. Encontrar contornos externos ───
    contornos, _ = cv2.findContours(
        edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    # ─── 5. Classificar contornos ───
    area_min = area_total * AREA_MIN_PCT
    area_max = area_total * AREA_MAX_PCT
    
    objetos_escolhidos = []
    descartados_pequenos = []
    descartados_grandes = []
    
    for c in contornos:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        
        if area < area_min:
            descartados_pequenos.append((c, x, y, w, h, area))
        elif area > area_max:
            descartados_grandes.append((c, x, y, w, h, area))
        else:
            # Aplicar margem (com clamp dentro da imagem)
            x_m = max(0, x - MARGEM_RECORTE)
            y_m = max(0, y - MARGEM_RECORTE)
            w_m = min(w_img - x_m, w + 2 * MARGEM_RECORTE)
            h_m = min(h_img - y_m, h + 2 * MARGEM_RECORTE)
            
            objetos_escolhidos.append({
                "contorno": c,
                "bbox": (x_m, y_m, w_m, h_m),
                "bbox_raw": (x, y, w, h),
                "area_pct": area / area_total,
                "centro": (x + w // 2, y + h // 2),
            })
    
    # ─── 5b. NON-MAXIMUM SUPPRESSION: remove "pais" que contêm "filhos" ───
    # Se um contorno maior engloba >60% de outro menor, descarta o maior.
    # Evita o caso de detectar "2 chaves juntas" + "cada chave separada" = 3 objetos sobrepostos.
    antes_nms = len(objetos_escolhidos)
    objetos_escolhidos = _suprimir_sobreposicoes(objetos_escolhidos, NMS_OVERLAP_THRESHOLD)
    nms_removidos = antes_nms - len(objetos_escolhidos)
    
    _log.info("contornos_detectados", extra={
        "total_contornos": len(contornos),
        "escolhidos": len(objetos_escolhidos),
        "descartados_pequenos": len(descartados_pequenos),
        "descartados_grandes": len(descartados_grandes),
        "removidos_por_sobreposicao": nms_removidos,
        "dimensoes": f"{w_img}x{h_img}",
    })
    
    # ─── 6. Gerar visualização de debug ───
    if debug_path:
        _gerar_visualizacao_debug(
            img_bgr=img_bgr,
            escolhidos=objetos_escolhidos,
            descartados_pequenos=descartados_pequenos,
            descartados_grandes=descartados_grandes,
            edges=edges_closed,
            saida=debug_path,
        )
    
    # ─── 7. Ordenar por posição (top-left → bottom-right) ───
    objetos_escolhidos.sort(key=lambda o: (o["centro"][1] // 100, o["centro"][0]))
    
    # Remover o contorno bruto (não serializável)
    return [
        {
            "bbox": o["bbox"],
            "area_pct": o["area_pct"],
            "centro": o["centro"],
        }
        for o in objetos_escolhidos
    ]


def _iou_contido(bbox_a: tuple, bbox_b: tuple) -> float:
    """
    Retorna a fração de bbox_b que está contida em bbox_a (0.0 - 1.0).
    Útil para detectar quando A é "pai" de B (engloba grande parte).
    bbox = (x, y, w, h)
    """
    ax1, ay1, aw, ah = bbox_a
    bx1, by1, bw, bh = bbox_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    
    # Interseção
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    
    area_inter = (ix2 - ix1) * (iy2 - iy1)
    area_b = bw * bh
    
    return area_inter / area_b if area_b > 0 else 0.0


def _suprimir_sobreposicoes(objetos: list[dict], threshold: float) -> list[dict]:
    """
    Remove contornos que contêm outros (NMS por contenção).
    Estratégia: para cada par (A, B), se B está >threshold dentro de A,
    e A é maior que B, descarta A (mantém o "filho" mais específico).
    
    Preserva os ÍNDICES originais — mantém os menores sempre.
    """
    if len(objetos) <= 1:
        return objetos
    
    # Ordenar por área crescente (menores primeiro = mais específicos)
    ordenados = sorted(objetos, key=lambda o: o["bbox"][2] * o["bbox"][3])
    
    a_manter = []
    for obj in ordenados:
        eh_pai_de_algum = False
        for ja_mantido in a_manter:
            # Verifica se o JÁ MANTIDO (menor) está contido em obj (maior atual)
            overlap = _iou_contido(obj["bbox"], ja_mantido["bbox"])
            if overlap >= threshold:
                # obj é pai do ja_mantido → descarta obj
                eh_pai_de_algum = True
                _log.info("contorno_descartado_por_conter_outro", extra={
                    "bbox_pai": obj["bbox"],
                    "bbox_filho": ja_mantido["bbox"],
                    "overlap": round(overlap, 2),
                })
                break
        if not eh_pai_de_algum:
            a_manter.append(obj)
    
    return a_manter


def gerar_imagem_anotada(caminho_foto: str, objetos: list[dict], saida: str) -> str:
    """
    Gera uma cópia da foto original com bboxes numeradas em VERDE GROSSO
    (#1, #2, #3...). Essa imagem é enviada ao Claude para associar nomes.
    
    Diferente do debug visual (que tem 2 painéis e legendas), esta imagem
    é LIMPA — só a foto + retângulos + números, para o LLM focar.
    """
    img_bgr, _ = _carregar_imagem_orientada(caminho_foto)
    anotada = img_bgr.copy()

    for i, obj in enumerate(objetos, start=1):
        x, y, w, h = obj["bbox"]
        cv2.rectangle(anotada, (x, y), (x + w, y + h), (0, 255, 0), 4)
        label = f"#{i}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        cv2.rectangle(anotada, (x, y), (x + lw + 10, y + lh + 12), (0, 255, 0), -1)
        cv2.putText(anotada, label, (x + 5, y + lh + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

    cv2.imwrite(saida, anotada)
    _log.info("imagem_anotada_salva", extra={"caminho": saida, "n_bboxes": len(objetos)})
    return saida


def gerar_recorte(caminho_foto: str, bbox: tuple, saida: str) -> str:
    """
    Gera um recorte da foto usando bbox (x, y, w, h).
    
    Args:
        caminho_foto: foto original
        bbox: tupla (x, y, w, h) em pixels
        saida: caminho onde salvar o recorte
    
    Returns:
        Caminho do recorte salvo.
    """
    pil = Image.open(caminho_foto)
    pil = ImageOps.exif_transpose(pil)
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    
    x, y, w, h = bbox
    crop = pil.crop((x, y, x + w, y + h))
    crop.save(saida, "JPEG", quality=92)
    return saida


def gerar_icone_anotado(
    caminho_foto: str,
    saida: str,
    bbox_norm: dict | None = None,
    numero: int = 1,
    total: int = 1,
    nome: str = "",
) -> str:
    """
    Gera o ícone do item como a FOTO INTEIRA com o objeto destacado.

    Estratégia visual:
      - Fundo escurecido (70% de opacidade preta) sobre toda a foto
      - Região do objeto restaurada ao brilho original (spotlight)
      - Retângulo colorido ao redor do objeto
      - Badge circular com o número do objeto
      - Rodapé com nome (quando fornecido)

    Quando bbox_norm é None (não há bbox confiável), a foto inteira é
    mostrada sem escurecimento — apenas com o badge no canto superior.

    Args:
        caminho_foto: foto original
        saida: caminho de saída (.jpg)
        bbox_norm: dict {x1,y1,x2,y2} em 0.0–1.0, ou None
        numero: índice deste objeto (1-based)
        total: total de objetos na foto
        nome: nome do objeto para o rodapé

    Returns:
        Caminho da imagem salva.
    """
    img_bgr, _ = _carregar_imagem_orientada(caminho_foto)
    h, w = img_bgr.shape[:2]

    # Paleta de cores para cada objeto (até 8 objetos distintos)
    PALETA = [
        (0, 210, 0),    # verde
        (255, 100, 0),  # laranja
        (0, 120, 255),  # azul
        (180, 0, 255),  # roxo
        (0, 210, 210),  # ciano
        (255, 200, 0),  # amarelo
        (255, 0, 120),  # rosa
        (0, 160, 80),   # verde escuro
    ]
    cor = PALETA[(numero - 1) % len(PALETA)]

    resultado = img_bgr.copy()

    if bbox_norm:
        # Converte bbox para pixels
        x1 = max(0, int(bbox_norm["x1"] * w))
        y1 = max(0, int(bbox_norm["y1"] * h))
        x2 = min(w, int(bbox_norm["x2"] * w))
        y2 = min(h, int(bbox_norm["y2"] * h))

        # Escurece a foto inteira
        overlay = resultado.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        resultado = cv2.addWeighted(resultado, 0.35, overlay, 0.65, 0)

        # Restaura a região do objeto ao original
        resultado[y1:y2, x1:x2] = img_bgr[y1:y2, x1:x2]

        # Retângulo colorido ao redor do objeto (linha dupla para destaque)
        cv2.rectangle(resultado, (x1-2, y1-2), (x2+2, y2+2), (0, 0, 0), 5)
        cv2.rectangle(resultado, (x1-2, y1-2), (x2+2, y2+2), cor, 3)

        # Badge circular com número (posicionado no canto superior do objeto)
        badge_cx = x1 + 20
        badge_cy = y1 + 20
        cv2.circle(resultado, (badge_cx, badge_cy), 18, (0, 0, 0), -1)
        cv2.circle(resultado, (badge_cx, badge_cy), 18, cor, 3)
        cv2.putText(resultado, str(numero),
                    (badge_cx - 7 if numero < 10 else badge_cx - 11, badge_cy + 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    else:
        # Sem bbox: foto inteira sem escurecimento, badge no canto
        badge_cx, badge_cy = 22, 22
        cv2.circle(resultado, (badge_cx, badge_cy), 18, (0, 0, 0), -1)
        cv2.circle(resultado, (badge_cx, badge_cy), 18, cor, 3)
        cv2.putText(resultado, str(numero),
                    (badge_cx - 7 if numero < 10 else badge_cx - 11, badge_cy + 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    # Rodapé com nome (se fornecido)
    # cv2.putText só suporta ASCII — normaliza acentos via unicodedata
    if nome:
        import unicodedata
        nome_ascii = unicodedata.normalize("NFKD", nome).encode("ascii", "ignore").decode("ascii")
        nome_display = nome_ascii[:30] + "..." if len(nome_ascii) > 30 else nome_ascii
        rod_h = 32
        cv2.rectangle(resultado, (0, h - rod_h), (w, h), (0, 0, 0), -1)
        cv2.putText(resultado, nome_display,
                    (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    cor, 2, cv2.LINE_AA)

    # Rodapé de contexto "X de Y" quando há múltiplos objetos
    if total > 1:
        label = f"{numero}/{total}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(resultado, (w - lw - 14, 4), (w - 4, lh + 12), (0, 0, 0), -1)
        cv2.putText(resultado, label,
                    (w - lw - 10, lh + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1, cv2.LINE_AA)

    cv2.imwrite(saida, resultado)
    _log.info("icone_anotado_salvo", extra={
        "saida": saida, "numero": numero, "total": total,
        "tem_bbox": bbox_norm is not None,
    })
    return saida


def _gerar_visualizacao_debug(
    img_bgr: np.ndarray,
    escolhidos: list[dict],
    descartados_pequenos: list,
    descartados_grandes: list,
    edges: np.ndarray,
    saida: str,
) -> None:
    """
    Cria uma imagem de debug com 2 painéis lado a lado:
      • Esquerda: bordas detectadas (preto e branco)
      • Direita: foto original com contornos coloridos sobrepostos
    
    Cores:
      🟢 Verde grosso = escolhidos (viram recorte)
      🔵 Azul        = bbox com margem
      ⚪ Cinza       = descartados por pequenos (ruído)
      🔴 Vermelho    = descartados por grandes (mesa/fundo)
    """
    h, w = img_bgr.shape[:2]
    
    # Painel esquerdo: bordas detectadas
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cv2.putText(edges_color, "BORDAS DETECTADAS", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    
    # Painel direito: original anotada
    anotada = img_bgr.copy()
    
    # Descartados pequenos (cinza)
    for c, x, y, ww, hh, area in descartados_pequenos:
        cv2.drawContours(anotada, [c], -1, (128, 128, 128), 1)
    
    # Descartados grandes (vermelho semi-transparente)
    for c, x, y, ww, hh, area in descartados_grandes:
        cv2.drawContours(anotada, [c], -1, (0, 0, 200), 2)
        cv2.putText(anotada, "GRANDE", (x + 10, y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
    
    # Escolhidos (verde grosso + bbox azul)
    for i, obj in enumerate(escolhidos, start=1):
        c = obj["contorno"]
        x_m, y_m, w_m, h_m = obj["bbox"]
        
        # Contorno verde
        cv2.drawContours(anotada, [c], -1, (0, 255, 0), 3)
        # Bbox com margem em azul
        cv2.rectangle(anotada, (x_m, y_m), (x_m + w_m, y_m + h_m), (255, 100, 0), 2)
        # Número
        cv2.putText(anotada, f"#{i}", (x_m + 5, y_m + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    
    # Legenda
    legenda_y = h - 140
    cv2.rectangle(anotada, (10, legenda_y - 10), (380, h - 10), (255, 255, 255), -1)
    cv2.rectangle(anotada, (10, legenda_y - 10), (380, h - 10), (0, 0, 0), 2)
    cv2.putText(anotada, f"Escolhidos: {len(escolhidos)}", (20, legenda_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 0), 2)
    cv2.putText(anotada, f"Descartados (pequenos): {len(descartados_pequenos)}",
                (20, legenda_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
    cv2.putText(anotada, f"Descartados (grandes): {len(descartados_grandes)}",
                (20, legenda_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 180), 2)
    cv2.putText(anotada, f"Total contornos: {len(escolhidos) + len(descartados_pequenos) + len(descartados_grandes)}",
                (20, legenda_y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Concatenar lado a lado
    # Redimensionar para mesma altura (já são da mesma altura)
    composta = np.hstack([edges_color, anotada])
    
    cv2.imwrite(saida, composta)
    _log.info("debug_salvo", extra={"caminho": saida})
