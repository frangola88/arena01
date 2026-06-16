"""
core/analise_cena.py — pré-análise de cena por seções ortogonais (tomografia leve).

Roda UMA vez por foto-mestre, ANTES de qualquer chamada de modelo/API.
Reconstrói uma superfície clássica de "object-ness" a partir de seções
horizontais e verticais da variância local, no espírito da batimetria de
seções transversais em hidrologia / da transformada de Radon em tomografia.

Pipeline (zero ML, zero $$, ~0.1-0.4s CPU em 4000×3000):
    1. coerência de gradiente em blocos de 16px → mapa de "object-ness"
       (energia × anisotropia do structure tensor: distingue borda de
        objeto — gradiente coerente/linear — de textura de fundo —
        gradiente isotrópico; ver _coerencia_local)
    2. pré-suavização gaussiana (σ=1)           → atenua só o granulado fino
    3. amostra 48 seções H + 48 V (passo 4)     → ~2.6k pontos esparsos
    4. griddata cubic                           → reconstrói superfície
    5. pós-suavização gaussiana (σ=4)           → tira oscilações do cubic
    6. filtro sigmoid (k=12, μ=0.45)            → exacerba contraste fundo/objeto
    7. limiar por percentil + morfologia        → obj_mask / bg_mask
    8. componentes conectados                   → n_objetos_estimado + picos

Histórico: a versão original usava VARIÂNCIA local (+σ=4). Em cenas com muitos
objetos finos próximos (ex.: ferramentas alinhadas), a variância exigia σ alto
pra suprimir textura de madeira/tecido, e esse σ alto fundia objetos vizinhos —
subcontava (8/7 onde a verdade era 14). A coerência de gradiente separa textura
de borda real sem precisar do σ pesado, recuperando a contagem (~10-13 vs 14)
sem explodir em ruído. Validado em data/fotos_mestras/ (ver poc_dinov2/
demo_analise_cena_coerencia.py). _variancia_local fica disponível por
compatibilidade/testes.

Produtos consumíveis pelos passos seguintes:
    - n_objetos_estimado  : prior para o prompt do Claude ("espere ~N objetos")
    - complexidade        : simples | media | densa  → escolhe estratégia do agente
    - bg_mask             : fundo por HOMOGENEIDADE (ortogonal a cor; não sofre
                            do paradoxo same-color do vetor de fundo cromático)
    - obj_mask            : regiões candidatas a objeto
    - picos               : centróides + força de cada blob
    - bordas_ativas       : objeto tocando borda da foto (sinal de corte)

Comparações que motivaram os defaults (foto branco-no-branco, 12 obj reais):
    - griddata cubic 48×48 → 0.21s, 11 obj  (Kriging RBF: 36s, 5 obj — descartado)
    - sigmoid mantém 11 obj e nitidez; top-hat fragmenta (17); CLAHE infla (13)
    Ver poc_dinov2/demo_analise_cena.py e _compare/_filtro.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_log = logging.getLogger("casaiq.analise_cena")


# ─── parâmetros (defaults validados na POC) ──────────────────────────────────────

BLOCK_SZ   = 16        # px por bloco do sinal de object-ness
N_SECOES   = 48        # seções horizontais
M_SECOES   = 48        # seções verticais
PASSO_SAMP = 4         # subamostragem dentro de cada seção
SIGMA_SUAV = 1.0       # pré-suavização do sinal de coerência (blocos) —
                       # baixo de propósito: a coerência já suprime textura,
                       # não precisa do σ alto que a variância exigia (e que
                       # fundia objetos próximos)
SIGMA_SUP  = 4.0       # pós-suavização da superfície reconstruída
SIGMOID_K  = 12.0      # inclinação do filtro sigmoid
SIGMOID_MU = 0.45      # ponto de inflexão do sigmoid
TH_BG_PCT  = 25        # percentil abaixo → fundo
TH_OBJ_PCT = 70        # percentil acima → objeto candidato
AREA_MIN   = 4         # área mínima de blob (blocos) p/ contar como objeto
LIM_SIMPLES = 5        # ≤ → "simples"
LIM_MEDIA   = 12       # ≤ → "media"; acima → "densa"


@dataclass
class AnaliseCena:
    """Resultado da pré-análise de uma foto-mestre."""
    n_objetos_estimado: int
    complexidade: str                       # simples | media | densa
    picos: list[dict]                       # [{cx, cy, forca, area_blocos}] em coords 0-1
    bordas_ativas: dict                     # {top, bottom, left, right} bool
    superficie: np.ndarray = field(repr=False)   # (Hb,Wb) float32 0-1, object-ness
    obj_mask: np.ndarray = field(repr=False)     # (Hb,Wb) bool
    bg_mask: np.ndarray = field(repr=False)      # (Hb,Wb) bool — fundo por homogeneidade
    shape_blocos: tuple = (0, 0)
    shape_original: tuple = (0, 0)
    tempo_s: float = 0.0

    def resumo_prompt(self) -> str:
        """String curta para injetar no contexto do prompt do Claude."""
        cortes = [k for k, v in self.bordas_ativas.items() if v]
        txt = (f"~{self.n_objetos_estimado} objetos (cena {self.complexidade})")
        if cortes:
            txt += f"; possível corte na borda: {', '.join(cortes)}"
        return txt


# ─── núcleo numérico ─────────────────────────────────────────────────────────────

def _variancia_local(img_gray: np.ndarray, block_sz: int) -> np.ndarray:
    """Sinal legado (variância por bloco). Mantido por compatibilidade e
    testes; o pipeline de produção usa _coerencia_local."""
    H, W = img_gray.shape
    Hb, Wb = H // block_sz, W // block_sz
    bl = img_gray[:Hb * block_sz, :Wb * block_sz].reshape(Hb, block_sz, Wb, block_sz)
    return bl.var(axis=(1, 3)).astype(np.float32)


def _coerencia_local(img_gray: np.ndarray, block_sz: int) -> np.ndarray:
    """Sinal de object-ness por coerência de gradiente (structure tensor).

    Para cada bloco de block_sz px calcula o tensor de estrutura 2×2 a partir
    dos gradientes Sobel (médias de Ix², Iy², IxIy no bloco), seus autovalores
    λ1≥λ2, e devolve  energia × anisotropia = (λ1+λ2) · (λ1−λ2)/(λ1+λ2).

    - energia (λ1+λ2): quanta variação de borda há no bloco.
    - anisotropia (λ1−λ2)/(λ1+λ2) ∈ [0,1]: 0 = gradiente isotrópico (textura,
      ruído) → suprimido; ~1 = gradiente coerente/linear (borda real) → realçado.

    É isso que separa borda de ferramenta de grão de madeira/trama de tecido
    sem precisar do blur pesado que a variância exigia.
    """
    from scipy.ndimage import sobel
    Ix = sobel(img_gray.astype(np.float32), axis=1)
    Iy = sobel(img_gray.astype(np.float32), axis=0)
    H, W = img_gray.shape
    Hb, Wb = H // block_sz, W // block_sz

    def _block_mean(a: np.ndarray) -> np.ndarray:
        a = a[:Hb * block_sz, :Wb * block_sz].reshape(Hb, block_sz, Wb, block_sz)
        return a.mean(axis=(1, 3))

    Jxx, Jyy, Jxy = _block_mean(Ix * Ix), _block_mean(Iy * Iy), _block_mean(Ix * Iy)
    trace = Jxx + Jyy
    disc = np.sqrt(np.clip((trace / 2) ** 2 - (Jxx * Jyy - Jxy ** 2), 0, None))
    lam1, lam2 = trace / 2 + disc, trace / 2 - disc
    energia = lam1 + lam2
    anisotropia = (lam1 - lam2) / (lam1 + lam2 + 1e-8)
    return (energia * anisotropia).astype(np.float32)


def _normalizar(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(arr.min()), float(arr.max())
    return (arr - lo) / (hi - lo + 1e-8)


def _amostrar_secoes(mapa: np.ndarray, n_h: int, m_v: int, passo: int):
    Hb, Wb = mapa.shape
    ys_h = np.linspace(0, Hb - 1, n_h, dtype=int)
    xs_v = np.linspace(0, Wb - 1, m_v, dtype=int)
    pts, vals = [], []
    xs_all = np.arange(0, Wb, passo)
    ys_all = np.arange(0, Hb, passo)
    for y in ys_h:
        for x in xs_all:
            pts.append((x / (Wb - 1), y / (Hb - 1)))
            vals.append(float(mapa[y, x]))
    for x in xs_v:
        for y in ys_all:
            pts.append((x / (Wb - 1), y / (Hb - 1)))
            vals.append(float(mapa[y, x]))
    return np.array(pts, dtype=np.float64), np.array(vals, dtype=np.float64)


def _reconstruir(shape, pts, vals) -> np.ndarray:
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter
    Hb, Wb = shape
    gx = np.linspace(0, 1, Wb)
    gy = np.linspace(0, 1, Hb)
    GX, GY = np.meshgrid(gx, gy)
    sup = griddata(pts, vals, (GX, GY), method="cubic", fill_value=0.0)
    sup = np.clip(sup, 0, None).astype(np.float32)
    sup = gaussian_filter(sup, sigma=SIGMA_SUP)
    return _normalizar(sup)


def _sigmoid(sup: np.ndarray, k: float, mu: float) -> np.ndarray:
    return _normalizar(1.0 / (1.0 + np.exp(-k * (sup - mu))))


def _detectar_objetos(superficie: np.ndarray, th_obj_pct: float):
    from scipy.ndimage import label, binary_closing, binary_opening
    th = np.percentile(superficie, th_obj_pct)
    mask = superficie > th
    mask = binary_closing(mask, iterations=2)
    mask = binary_opening(mask, iterations=1)
    labeled, n = label(mask)
    Hb, Wb = superficie.shape
    picos = []
    for i in range(1, n + 1):
        ys, xs = np.where(labeled == i)
        if len(ys) < AREA_MIN:
            continue
        picos.append({
            "cx": float(xs.mean() / (Wb - 1)),
            "cy": float(ys.mean() / (Hb - 1)),
            "forca": float(superficie[ys, xs].max()),
            "area_blocos": int(len(ys)),
        })
    picos.sort(key=lambda p: -p["forca"])
    return picos, mask


def _carregar_cinza(fonte) -> np.ndarray:
    """Aceita caminho (str/Path), array BGR/RGB (HxWx3) ou cinza (HxW)."""
    if isinstance(fonte, (str, Path)):
        try:
            import cv2
            img = cv2.imread(str(fonte))
            if img is None:
                raise ValueError(f"imagem ilegível: {fonte}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except ImportError:
            from PIL import Image
            return np.asarray(Image.open(fonte).convert("L"))
    arr = np.asarray(fonte)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        # luminância simples (não importa BGR vs RGB para variância)
        return (arr[..., :3].mean(axis=2)).astype(np.float32)
    raise ValueError(f"formato de imagem não suportado: shape={arr.shape}")


# ─── API pública ─────────────────────────────────────────────────────────────────

def analisar_cena(fonte, *, aplicar_sigmoid: bool = True) -> AnaliseCena:
    """
    Pré-analisa a foto-mestre e devolve uma `AnaliseCena`.

    Args:
        fonte: caminho da imagem, ou ndarray (BGR/RGB/cinza).
        aplicar_sigmoid: aplica o realce não-linear de contraste (default True).

    Nunca levanta para o chamador em caso de erro de imagem: loga e devolve
    uma análise vazia (n_objetos_estimado=0, complexidade='simples'), para não
    quebrar o pipeline de segmentação que a consome como hint opcional.
    """
    t0 = time.time()
    try:
        from scipy.ndimage import gaussian_filter

        img_gray = _carregar_cinza(fonte)
        H_orig, W_orig = img_gray.shape

        sinal = _coerencia_local(img_gray, BLOCK_SZ)
        var_norm = _normalizar(gaussian_filter(sinal, sigma=SIGMA_SUAV))
        Hb, Wb = var_norm.shape

        pts, vals = _amostrar_secoes(var_norm, N_SECOES, M_SECOES, PASSO_SAMP)
        superficie = _reconstruir((Hb, Wb), pts, vals)
        if aplicar_sigmoid:
            superficie = _sigmoid(superficie, SIGMOID_K, SIGMOID_MU)

        th_bg = np.percentile(superficie, TH_BG_PCT)
        bg_mask = superficie < th_bg
        picos, obj_mask = _detectar_objetos(superficie, TH_OBJ_PCT)

        n = len(picos)
        complexidade = ("simples" if n <= LIM_SIMPLES
                        else "media" if n <= LIM_MEDIA
                        else "densa")
        th_obj = np.percentile(superficie, TH_OBJ_PCT)
        bordas = {
            "top":    bool(superficie[0,  :].max() > th_obj),
            "bottom": bool(superficie[-1, :].max() > th_obj),
            "left":   bool(superficie[:,  0].max() > th_obj),
            "right":  bool(superficie[:, -1].max() > th_obj),
        }

        res = AnaliseCena(
            n_objetos_estimado=n,
            complexidade=complexidade,
            picos=picos,
            bordas_ativas=bordas,
            superficie=superficie,
            obj_mask=obj_mask,
            bg_mask=bg_mask,
            shape_blocos=(Hb, Wb),
            shape_original=(H_orig, W_orig),
            tempo_s=round(time.time() - t0, 3),
        )
        _log.info("analise_cena_ok", extra={
            "n_obj": n, "complexidade": complexidade,
            "bordas": bordas, "tempo_s": res.tempo_s,
        })
        return res

    except Exception as e:  # pragma: no cover - degrada graciosamente
        _log.warning("analise_cena_falhou", extra={"erro": str(e)})
        return AnaliseCena(
            n_objetos_estimado=0,
            complexidade="simples",
            picos=[],
            bordas_ativas={"top": False, "bottom": False, "left": False, "right": False},
            superficie=np.zeros((1, 1), dtype=np.float32),
            obj_mask=np.zeros((1, 1), dtype=bool),
            bg_mask=np.zeros((1, 1), dtype=bool),
            tempo_s=round(time.time() - t0, 3),
        )


if __name__ == "__main__":
    import sys
    alvo = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/cuco/Downloads/IMG_20260610_112623028.jpg"
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    r = analisar_cena(alvo)
    print(f"n_objetos_estimado : {r.n_objetos_estimado}")
    print(f"complexidade       : {r.complexidade}")
    print(f"bordas_ativas      : {r.bordas_ativas}")
    print(f"shape_blocos       : {r.shape_blocos}")
    print(f"tempo_s            : {r.tempo_s}")
    print(f"resumo_prompt      : {r.resumo_prompt()}")
