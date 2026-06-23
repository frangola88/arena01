"""
Pipeline de crop em 2 estágios com gatilho opcional de 3º.

Stage 1 — fornecido pelo chamador (bbox_normalizada do analisar_foto_completa)
Stage 2 — crop generoso enviado ao Claude → bbox refinada dentro do crop
Stage 3 — ativado se S2 retornar qualidade='multiplos' ou confiança < 0.70

Retorna (caminho_saida, tag)
  tag: 'S2_bbox' | 'S3_bbox'
"""
import logging
import os
import tempfile
import unicodedata
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from core.segmentacao_cv import _carregar_imagem_orientada
from core.llm import chamar_visao, extrair_json

_log = logging.getLogger("casaiq.crop_refinador")

# ── Rembg (matting) — sessão singleton ───────────────────────────────────────
_rembg_session = None   # None = ainda não tentou; False = tentou e falhou

def _get_rembg_session():
    global _rembg_session
    if _rembg_session is None:
        try:
            from rembg import new_session
            _rembg_session = new_session("u2net")
            _log.info("rembg_sessao_criada")
        except Exception as e:
            _log.warning("rembg_nao_disponivel", extra={"erro": str(e)})
            _rembg_session = False
    return None if _rembg_session is False else _rembg_session


def _aplicar_matting(img_bgr: np.ndarray, bbox: dict,
                     escuro_bgr: np.ndarray, W: int, H: int) -> np.ndarray | None:
    """
    Usa rembg para remover o fundo da região bbox e blends suavemente com
    o fundo escuro via canal alpha.  Retorna imagem full-size com blend
    aplicado, ou None em caso de falha (caller usa fallback spotlight).
    """
    session = _get_rembg_session()
    if session is None:
        return None

    from rembg import remove

    bx1 = max(0, int(bbox["x1"] * W))
    by1 = max(0, int(bbox["y1"] * H))
    bx2 = min(W, int(bbox["x2"] * W))
    by2 = min(H, int(bbox["y2"] * H))
    roi = img_bgr[by1:by2, bx1:bx2]
    if roi.size == 0:
        return None

    try:
        roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        buf = BytesIO()
        roi_pil.save(buf, "PNG")
        matted = np.array(Image.open(BytesIO(
            remove(buf.getvalue(), session=session)
        )).convert("RGBA"))
        alpha = matted[:, :, 3].astype(np.float32) / 255.0   # (H_roi, W_roi)
    except Exception as e:
        _log.warning("matting_falhou", extra={"erro": str(e)})
        return None

    result = escuro_bgr.copy()
    a3 = alpha[:, :, np.newaxis]
    esc_roi = escuro_bgr[by1:by2, bx1:bx2].astype(np.float32)
    roi_f   = roi.astype(np.float32)
    result[by1:by2, bx1:bx2] = (roi_f * a3 + esc_roi * (1.0 - a3)).astype(np.uint8)
    return result

PALETA_BGR = [
    (0, 210, 0), (255, 100, 0), (0, 120, 255), (180, 0, 255),
    (0, 210, 210), (255, 200, 0), (255, 0, 120), (0, 160, 80),
]

PROMPT_CROP = """Esta imagem é um recorte de uma foto de inventário doméstico.
Identifique a ferramenta ou objeto PRINCIPAL e responda APENAS com JSON válido:

{
  "nome": "nome simples do objeto principal",
  "bbox_normalizada": {"x1": float, "y1": float, "x2": float, "y2": float},
  "qualidade_crop": "ok",
  "confianca": float,
  "observacao": "string curta"
}

Regras:
- bbox_normalizada: retângulo da ferramenta PRINCIPAL em coords 0.0-1.0 DESTE CROP
- qualidade_crop:
    "ok"         = objeto único e completo dominando o frame
    "multiplos"  = mais de um objeto claramente visível
    "incompleto" = objeto cortado pela borda
- confianca: 0.0-1.0
- observacao: frase curta (ex: "formão completo isolado", "dois ponteiros presentes")"""


# ── Helpers internos ──────────────────────────────────────────────────────────

def _ascii(nome: str, n: int = 26) -> str:
    s = unicodedata.normalize("NFKD", nome).encode("ascii", "ignore").decode("ascii")
    return s[:n] + "..." if len(s) > n else s


def _crop_generoso(img_bgr: np.ndarray, obj: dict, W: int, H: int,
                   padding: float = 0.30) -> tuple[np.ndarray, tuple]:
    """Crop com padding generoso para dar contexto ao Claude no S2/S3."""
    bbox = obj.get("bbox_normalizada", {"x1": 0, "y1": 0, "x2": 1, "y2": 1})
    bw = (bbox["x2"] - bbox["x1"]) * W
    bh = (bbox["y2"] - bbox["y1"]) * H
    px1 = max(0, int(bbox["x1"] * W - bw * padding))
    py1 = max(0, int(bbox["y1"] * H - bh * padding))
    px2 = min(W, int(bbox["x2"] * W + bw * padding))
    py2 = min(H, int(bbox["y2"] * H + bh * padding))
    return img_bgr[py1:py2, px1:px2].copy(), (px1, py1, px2, py2)


def _mapear_coords(r: dict, origem: tuple, W: int, H: int) -> dict:
    """Mapeia bbox em coords do crop de volta para coords da imagem original."""
    px1, py1, px2, py2 = origem
    cw, ch = px2 - px1, py2 - py1
    bb = r.get("bbox_normalizada", {})
    return {
        "x1": (px1 + bb.get("x1", 0) * cw) / W,
        "y1": (py1 + bb.get("y1", 0) * ch) / H,
        "x2": (px1 + bb.get("x2", 1) * cw) / W,
        "y2": (py1 + bb.get("y2", 1) * ch) / H,
    }


def _chamar_claude(crop_bgr: np.ndarray) -> dict:
    """Salva crop em arquivo temporário, chama Claude, retorna dict parseado."""
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(tmp.name, crop_bgr)
    tmp.close()
    try:
        texto, _ = chamar_visao(PROMPT_CROP, tmp.name)
        return extrair_json(texto)
    except Exception as e:
        _log.warning("claude_crop_falhou", extra={"erro": str(e)})
        return {"qualidade_crop": "erro", "confianca": 0.0, "observacao": str(e)}
    finally:
        os.unlink(tmp.name)


def _gerar_crop_final(img_bgr: np.ndarray, bbox: dict,
                      W: int, H: int, numero: int, nome: str, cor: tuple) -> np.ndarray | None:
    """Crop com matting (rembg) + fundo escuro + badge numerado.
    Fallback para spotlight retangular se rembg não disponível."""
    pad = 0.10
    bw = (bbox["x2"] - bbox["x1"]) * W
    bh = (bbox["y2"] - bbox["y1"]) * H
    x1 = max(0, int(bbox["x1"] * W - bw * pad))
    y1 = max(0, int(bbox["y1"] * H - bh * pad))
    x2 = min(W, int(bbox["x2"] * W + bw * pad))
    y2 = min(H, int(bbox["y2"] * H + bh * pad))

    escuro = cv2.addWeighted(img_bgr, 0.10, np.zeros_like(img_bgr), 0.90, 0)
    out = _aplicar_matting(img_bgr, bbox, escuro, W, H)
    if out is None:
        out = escuro.copy()
        bx1, by1 = int(bbox["x1"] * W), int(bbox["y1"] * H)
        bx2, by2 = int(bbox["x2"] * W), int(bbox["y2"] * H)
        out[by1:by2, bx1:bx2] = img_bgr[by1:by2, bx1:bx2]
    crop = out[y1:y2, x1:x2].copy()

    if crop.size == 0:
        return None

    ch, cw = crop.shape[:2]
    bcx, bcy = min(22, cw - 4), min(22, ch - 4)
    cv2.circle(crop, (bcx, bcy), 16, (0, 0, 0), -1)
    cv2.circle(crop, (bcx, bcy), 16, cor, 3)
    cv2.putText(crop, str(numero),
                (bcx - 6 if numero < 10 else bcx - 10, bcy + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    rod = 26
    cv2.rectangle(crop, (0, ch - rod), (cw, ch), (0, 0, 0), -1)
    cv2.putText(crop, _ascii(nome), (5, ch - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2, cv2.LINE_AA)
    return crop


# ── API pública ───────────────────────────────────────────────────────────────

def _bbox_tocando_bordas(bbox: dict, crop_shape: tuple, tolerancia: float = 0.05) -> list[str]:
    """
    Verifica se a bbox está tocando as bordas do crop (sinalizando possível corte).
    Retorna lista de lados que estão muito próximos da borda: ['left', 'right', 'top', 'bottom']

    bbox: coordenadas normalizadas [0, 1] dentro do crop
    crop_shape: (altura, largura) do crop
    tolerancia: quanto dos 0.0-1.0 contar como "tocando borda" (padrão: 5%)
    """
    cortados = []
    if bbox.get("x1", 0.0) < tolerancia:
        cortados.append("left")
    if bbox.get("x2", 1.0) > (1.0 - tolerancia):
        cortados.append("right")
    if bbox.get("y1", 0.0) < tolerancia:
        cortados.append("top")
    if bbox.get("y2", 1.0) > (1.0 - tolerancia):
        cortados.append("bottom")
    return cortados


def _expandir_bbox_por_corte(bbox: dict, W: int, H: int,
                             lados_cortados: list[str], expansao: float = 0.15) -> dict:
    """
    Expande a bbox na(s) direção(ões) onde foram detectadas possíveis cortes.

    Args:
        bbox: bbox original em coords normalizadas
        W, H: dimensões da imagem
        lados_cortados: ['left', 'right', 'top', 'bottom']
        expansao: quanto expandir como fração da dimensão (padrão: 15%)

    Returns:
        bbox expandida (clipeada para [0, 1])
    """
    bw = (bbox["x2"] - bbox["x1"]) * W
    bh = (bbox["y2"] - bbox["y1"]) * H
    exp_x = (bw * expansao) / W
    exp_y = (bh * expansao) / H

    novo_bbox = bbox.copy()
    if "left" in lados_cortados:
        novo_bbox["x1"] = max(0.0, novo_bbox["x1"] - exp_x)
    if "right" in lados_cortados:
        novo_bbox["x2"] = min(1.0, novo_bbox["x2"] + exp_x)
    if "top" in lados_cortados:
        novo_bbox["y1"] = max(0.0, novo_bbox["y1"] - exp_y)
    if "bottom" in lados_cortados:
        novo_bbox["y2"] = min(1.0, novo_bbox["y2"] + exp_y)

    return novo_bbox


def refinar_crop(caminho_foto: str, obj: dict, saida: str,
                 numero: int, total_objetos: int, nome: str) -> tuple[str, str]:
    """
    Gera o crop refinado de um objeto em 2 estágios (+ gatilhos opcionais).

    Stages:
    - Stage 2: crop generoso, Claude refina a bbox
    - Stage 2.5: se bbox toca bordas (possível corte), expande e roda Claude novamente
    - Stage 3: se qualidade='multiplos' ou conf < 0.70, roda com padding menor

    Args:
        caminho_foto  : foto original (EXIF corrigido internamente)
        obj           : dict com 'bbox_normalizada' (obrigatório)
        saida         : caminho de saída .jpg
        numero        : índice 1-based do objeto
        total_objetos : total de objetos na foto (para contexto)
        nome          : nome do objeto

    Returns:
        (saida, tag) — tag: 'S2_bbox' | 'S2.5_bbox_expandida' | 'S3_bbox'
    """
    img_bgr, _ = _carregar_imagem_orientada(caminho_foto)
    H, W = img_bgr.shape[:2]
    cor = PALETA_BGR[(numero - 1) % len(PALETA_BGR)]

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    crop_rough, origem = _crop_generoso(img_bgr, obj, W, H)
    if crop_rough.size == 0:
        raise ValueError("crop_rough vazio no stage 2")

    r2 = _chamar_claude(crop_rough)
    qual2 = r2.get("qualidade_crop", "erro")
    conf2 = r2.get("confianca", 0.0)
    bbox2 = _mapear_coords(r2, origem, W, H)

    _log.info("crop_s2", extra={
        "nome": nome, "qual": qual2, "conf": f"{conf2:.2f}",
        "foto": Path(caminho_foto).name,
    })

    bbox_final = bbox2
    tag = "S2_bbox"

    # ── Stage 2.5 (gatilho: bbox tocando bordas = possível corte) ──────────────
    lados_cortados = _bbox_tocando_bordas(r2.get("bbox_normalizada", {}),
                                          crop_rough.shape, tolerancia=0.05)
    if lados_cortados:
        bbox2_expandida = _expandir_bbox_por_corte(bbox2, W, H, lados_cortados, expansao=0.15)
        obj_s25 = {"bbox_normalizada": bbox2_expandida}
        crop_s25, origem_s25 = _crop_generoso(img_bgr, obj_s25, W, H, padding=0.12)
        if crop_s25.size > 0:
            r25 = _chamar_claude(crop_s25)
            qual25 = r25.get("qualidade_crop", "erro")
            conf25 = r25.get("confianca", 0.0)
            bbox25 = _mapear_coords(r25, origem_s25, W, H)
            bbox_final = bbox25
            tag = "S2.5_bbox_expandida"
            _log.info("crop_s2.5_expandida_por_corte", extra={
                "nome": nome, "lados": lados_cortados,
                "qual": qual25, "conf": f"{conf25:.2f}",
            })

    # ── Stage 3 (gatilho: qualidade ou confiança baixa) ──────────────────────
    precisa_s3 = (qual2 == "multiplos") or (conf2 < 0.70)
    if precisa_s3 and tag != "S2.5_bbox_expandida":  # não acumular estágios
        obj_s3 = {"bbox_normalizada": bbox_final}
        crop_s3, origem_s3 = _crop_generoso(img_bgr, obj_s3, W, H, padding=0.08)
        if crop_s3.size > 0:
            r3 = _chamar_claude(crop_s3)
            qual3 = r3.get("qualidade_crop", "erro")
            conf3 = r3.get("confianca", 0.0)
            bbox3 = _mapear_coords(r3, origem_s3, W, H)
            bbox_final = bbox3
            tag = "S3_bbox"
            _log.info("crop_s3", extra={
                "nome": nome, "qual": qual3, "conf": f"{conf3:.2f}",
            })

    # ── Crop final ────────────────────────────────────────────────────────────
    crop_out = _gerar_crop_final(img_bgr, bbox_final, W, H, numero, nome, cor)
    if crop_out is None:
        raise ValueError("crop_final vazio")

    Path(saida).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(saida, crop_out)
    return saida, tag
