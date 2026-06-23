"""
core/verificacoes_cruzadas.py — checagens entre o sinal clássico (analise_cena,
etapa 0) e o sinal semântico do Claude (etapa 1).

`analise_cena` produz um sinal ORTOGONAL a toda a IA do pipeline: não usa rede
neural, cor semântica nem modelo treinado. Por isso serve de "pino de
referência" — não decide nada, mas detecta divergências entre o que o método
clássico vê e o que o Claude declarou. Este módulo materializa duas dessas
checagens (ver memória casaiq_pipeline_architecture, tabela de verificações):

  - ratio_contagem  : n_claude vs n_estimado → Claude perdeu/fragmentou objetos
  - centroide_em_bg : centroide do Claude cai em bloco classificado como FUNDO

São funções puras sobre a `AnaliseCena` + a lista de objetos do Claude: nenhuma
I/O, nenhum custo de API. Degradam para "sem_sinal" quando a pré-análise veio
vazia (n_estimado=0 / máscara 1×1), de modo que nunca produzem falso alarme a
partir de uma análise que falhou.

Primeiro incremento rumo ao `score_qualidade` por objeto: por enquanto o sinal é
apenas logado e anexado ao resultado (`_suspeita_bg`), sem bloquear o pipeline.
"""
from __future__ import annotations

import logging

import numpy as np

from core.analise_cena import AnaliseCena

_log = logging.getLogger("casaiq.verificacoes")

# Limiares de divergência de contagem — conservadores de propósito: o sinal é
# ruidoso. "Embalagem fechada = 1" faz o Claude contar menos que os blobs da
# cena, e a analise_cena subconta cenas densas. Só alarmamos em divergências
# grandes, para o veredito ser acionável e não barulho.
RATIO_PERDA = 0.5   # n_claude < 0.5·n_estimado → Claude provavelmente perdeu objetos
RATIO_FRAG = 2.0    # n_claude > 2.0·n_estimado → Claude fragmentando ou cena esparsa

# Nível de object-ness (superficie normalizada [0,1], realçada por sigmoid) que,
# logo fora da bbox, indica objeto transbordando → bbox apertada/objeto cortado.
LIM_BORDA = 0.5


def ratio_contagem(n_claude: int, n_estimado: int) -> dict:
    """Compara a contagem do Claude com a estimativa clássica da cena.

    Returns dict {ratio, veredito, n_claude, n_estimado}, onde veredito ∈
    {"ok", "claude_perdeu_objetos", "claude_fragmentando", "sem_sinal"}.
    """
    if n_estimado <= 0:
        return {"ratio": None, "veredito": "sem_sinal",
                "n_claude": n_claude, "n_estimado": n_estimado}

    ratio = n_claude / n_estimado
    if ratio < RATIO_PERDA:
        veredito = "claude_perdeu_objetos"
    elif ratio > RATIO_FRAG:
        veredito = "claude_fragmentando"
    else:
        veredito = "ok"
    return {"ratio": round(ratio, 3), "veredito": veredito,
            "n_claude": n_claude, "n_estimado": n_estimado}


def centroide_em_bg(centroide: dict | None, cena: AnaliseCena) -> bool:
    """True se o centroide do Claude cair num bloco classificado como FUNDO.

    O `bg_mask` da analise_cena marca regiões de baixa object-ness (fundo por
    homogeneidade). Se o Claude ancorou o objeto numa dessas regiões, é suspeita
    de bbox/centroide em fundo. Soft signal — não rejeita, sinaliza.

    centroide: {"cx": float, "cy": float} normalizado [0,1] (formato do Claude).
    """
    if not centroide:
        return False
    bg = cena.bg_mask
    Hb, Wb = bg.shape
    if Hb <= 1 or Wb <= 1:          # análise degradada → sem sinal
        return False
    cx = float(np.clip(centroide.get("cx", 0.5), 0.0, 1.0))
    cy = float(np.clip(centroide.get("cy", 0.5), 0.0, 1.0))
    bx = int(round(cx * (Wb - 1)))
    by = int(round(cy * (Hb - 1)))
    return bool(bg[by, bx])


# ─── helpers de mapa ─────────────────────────────────────────────────────────

def _bloco_do_centroide(centroide: dict, shape: tuple[int, int]) -> tuple[int, int]:
    """Mapeia centroide normalizado {cx,cy} → índice (by, bx) no grid de blocos."""
    Hb, Wb = shape
    cx = float(np.clip(centroide.get("cx", 0.5), 0.0, 1.0))
    cy = float(np.clip(centroide.get("cy", 0.5), 0.0, 1.0))
    return int(round(cy * (Hb - 1))), int(round(cx * (Wb - 1)))


def _crop_norm(mapa: np.ndarray, bbox: dict) -> np.ndarray:
    """Recorta um mapa (H,W) na região da bbox normalizada {x1,y1,x2,y2}."""
    H, W = mapa.shape
    x1 = int(np.clip(bbox.get("x1", 0.0), 0.0, 1.0) * (W - 1))
    x2 = int(np.clip(bbox.get("x2", 1.0), 0.0, 1.0) * (W - 1))
    y1 = int(np.clip(bbox.get("y1", 0.0), 0.0, 1.0) * (H - 1))
    y2 = int(np.clip(bbox.get("y2", 1.0), 0.0, 1.0) * (H - 1))
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    return mapa[y1:y2 + 1, x1:x2 + 1]


def _resample(m: np.ndarray, H: int, W: int) -> np.ndarray:
    ys = np.linspace(0, m.shape[0] - 1, H).astype(int)
    xs = np.linspace(0, m.shape[1] - 1, W).astype(int)
    return m[np.ix_(ys, xs)]


# ─── checagens por objeto (componentes do score_qualidade) ───────────────────

def centroide_em_obj(centroide: dict | None, cena: AnaliseCena) -> bool:
    """True se o centroide do Claude cair num bloco classificado como OBJETO
    pela análise clássica (contraparte positiva de centroide_em_bg)."""
    if not centroide:
        return False
    om = cena.obj_mask
    Hb, Wb = om.shape
    if Hb <= 1 or Wb <= 1:
        return False
    by, bx = _bloco_do_centroide(centroide, (Hb, Wb))
    return bool(om[by, bx])


def forca_no_centroide(centroide: dict | None, cena: AnaliseCena) -> float:
    """Object-ness clássica [0,1] no bloco do centroide (0 se sem sinal)."""
    if not centroide:
        return 0.0
    sup = cena.superficie
    Hb, Wb = sup.shape
    if Hb <= 1 or Wb <= 1:
        return 0.0
    by, bx = _bloco_do_centroide(centroide, (Hb, Wb))
    return float(np.clip(sup[by, bx], 0.0, 1.0))


def w_borda_bbox(bbox: dict | None, cena: AnaliseCena, faixa: float = 0.04,
                 th: float | None = None) -> dict:
    """Object-ness numa faixa logo FORA de cada borda da bbox.

    Analogia ao W(y) da tomografia de cena: se a object-ness continua alta
    imediatamente fora da bbox, o objeto provavelmente se estende além dela —
    bbox apertada / objeto cortado. Sinal clássico para o gate do 3º estágio
    do crop, sem custo de API.

    Returns {top,bottom,left,right: float, max: float, cortado: [lados]}.
    """
    vazio = {"top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0,
             "max": 0.0, "cortado": []}
    sup = cena.superficie
    Hb, Wb = sup.shape
    if Hb <= 1 or Wb <= 1 or not bbox:
        return vazio

    x1 = float(np.clip(bbox.get("x1", 0.0), 0.0, 1.0))
    x2 = float(np.clip(bbox.get("x2", 1.0), 0.0, 1.0))
    y1 = float(np.clip(bbox.get("y1", 0.0), 0.0, 1.0))
    y2 = float(np.clip(bbox.get("y2", 1.0), 0.0, 1.0))

    def faixa_media(yy1, yy2, xx1, xx2) -> float:
        r1, r2 = int(yy1 * (Hb - 1)), int(yy2 * (Hb - 1))
        c1, c2 = int(xx1 * (Wb - 1)), int(xx2 * (Wb - 1))
        r1, r2 = max(0, min(r1, r2)), min(Hb - 1, max(r1, r2))
        c1, c2 = max(0, min(c1, c2)), min(Wb - 1, max(c1, c2))
        sub = sup[r1:r2 + 1, c1:c2 + 1]
        return float(sub.mean()) if sub.size else 0.0

    f = faixa
    vals = {
        "top":    faixa_media(max(0.0, y1 - f), y1, x1, x2),
        "bottom": faixa_media(y2, min(1.0, y2 + f), x1, x2),
        "left":   faixa_media(y1, y2, max(0.0, x1 - f), x1),
        "right":  faixa_media(y1, y2, x2, min(1.0, x2 + f)),
    }
    if th is None:
        th = LIM_BORDA
    cortado = [k for k, v in vals.items() if v > th]
    vals["max"] = max(vals.values())
    vals["cortado"] = cortado
    return vals


def concordancia_mapas(sup_a: np.ndarray, sup_b: np.ndarray) -> float:
    """Concordância [0,1] entre dois mapas (ex.: superficie_cena × heatmap_dinov2).

    Reamostra ambos ao menor grid comum e devolve a correlação de Pearson
    remapeada para [0,1] (1 = mapas concordam; 0.5 = sem relação; baixo =
    discordam, sinal de match FAISS ruim). Retorna 0.5 (neutro) se algum mapa
    for degenerado.
    """
    a = np.asarray(sup_a, dtype=float)
    b = np.asarray(sup_b, dtype=float)
    if a.size < 4 or b.size < 4:
        return 0.5
    H = max(2, min(a.shape[0], b.shape[0]))
    W = max(2, min(a.shape[1], b.shape[1]))
    a2 = _resample(a, H, W) - _resample(a, H, W).mean()
    b2 = _resample(b, H, W) - _resample(b, H, W).mean()
    denom = np.sqrt((a2 ** 2).sum() * (b2 ** 2).sum())
    if denom < 1e-8:
        return 0.5
    r = float((a2 * b2).sum() / denom)          # [-1, 1]
    return float(np.clip((r + 1.0) / 2.0, 0.0, 1.0))


# Pesos do score combinado (somam 1.0 com 'mapa'; renormalizados sem ele).
PESOS_SCORE = {"centro": 0.30, "forca": 0.20, "borda": 0.15, "conf": 0.20, "mapa": 0.15}


def score_qualidade(obj: dict, cena: AnaliseCena,
                    heatmap: np.ndarray | None = None) -> dict:
    """Combina os sinais ortogonais (clássico + Claude + DINOv2) num score [0,1].

    Objetivo: filtrar inventário sem revisão manual e localizar onde o pipeline
    erra. NÃO bloqueia — é um diagnóstico anexável ao objeto.

    Componentes (cada um [0,1], maior = melhor):
      centro : centroide em obj_mask (1) / nem obj nem bg (0.5) / em bg (0)
      forca  : object-ness clássica no centroide
      borda  : 1 − object-ness fora da bbox (baixo overflow = não cortado)
      conf   : confiança declarada pelo Claude
      mapa   : concordância superficie_cena × heatmap_dinov2 (só se heatmap dado)

    Returns {score, componentes:{...}, flags:[...]}.
    """
    centroide = obj.get("centroide_normalizado")
    bbox = obj.get("bbox_normalizada")

    em_bg = centroide_em_bg(centroide, cena)
    em_obj = centroide_em_obj(centroide, cena)
    forca = forca_no_centroide(centroide, cena)
    wb = w_borda_bbox(bbox, cena) if bbox else {"cortado": [], "max": 0.0}
    conf = float(obj.get("confianca", 0.5) or 0.5)

    comps = {
        "centro": 1.0 if em_obj else (0.0 if em_bg else 0.5),
        "forca":  forca,
        "borda":  float(np.clip(1.0 - wb.get("max", 0.0), 0.0, 1.0)),
        "conf":   float(np.clip(conf, 0.0, 1.0)),
    }
    pesos = dict(PESOS_SCORE)
    if heatmap is not None and bbox and cena.superficie.shape[0] > 1:
        sup_bb = _crop_norm(cena.superficie, bbox)
        hm_bb = _crop_norm(np.asarray(heatmap, dtype=float), bbox)
        comps["mapa"] = concordancia_mapas(sup_bb, hm_bb)
    else:
        pesos.pop("mapa")

    tot = sum(pesos[k] for k in comps)
    score = sum(comps[k] * pesos[k] for k in comps) / tot if tot > 0 else 0.0

    flags = []
    if em_bg:
        flags.append("centroide_em_bg")
    if wb.get("cortado"):
        flags.append("bbox_cortando:" + ",".join(wb["cortado"]))
    if forca < 0.20:
        flags.append("baixa_forca_cena")
    if conf < 0.60:
        flags.append("baixa_confianca_claude")

    return {
        "score": round(score, 3),
        "componentes": {k: round(v, 3) for k, v in comps.items()},
        "flags": flags,
    }


def verificar_cena(cena: AnaliseCena, objetos: list[dict]) -> dict:
    """Roda as checagens cruzadas e devolve um relatório estruturado.

    objetos: lista de dicts do Claude (cada um pode ter 'nome' e
             'centroide_normalizado').

    Returns:
        {
          "contagem": {ratio, veredito, n_claude, n_estimado},
          "centroide_em_bg": [nomes suspeitos],
          "n_suspeitos_bg": int,
        }
    """
    contagem = ratio_contagem(len(objetos), cena.n_objetos_estimado)

    suspeitos_bg = [
        o.get("nome", "?")
        for o in objetos
        if centroide_em_bg(o.get("centroide_normalizado"), cena)
    ]

    return {
        "contagem": contagem,
        "centroide_em_bg": suspeitos_bg,
        "n_suspeitos_bg": len(suspeitos_bg),
    }
