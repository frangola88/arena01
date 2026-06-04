"""
GazetteerMatcher — refinamento de bbox via DINOv2 + índice FAISS.

Fluxo de uso no agent_1:
  1. Claude identifica objeto (nome + bbox_normalizada semântica/imprecisa).
  2. GazetteerMatcher.refinar_bbox(foto, bbox_claude, nome=nome) →
       a. [Texto] Se nome fornecido, tenta lookup no índice texto→faiss_idx.
          Se hit, usa embedding canônico (produto exato do gazetteer).
       b. [Visual] Encoda crop da região do Claude → query vec.
          FAISS lookup se dist >= limiar; senão usa query_vec.
       c. Encoda foto inteira como grid de patches.
       d. Cosine similarity → cluster_bbox → bbox refinada em pixels.
       e. Normaliza para [0,1] e retorna.
  3. A bbox refinada substitui a do Claude no ícone anotado.

Lookup por texto (buscar_por_nome):
  - Requer text_index.json gerado por pipeline/stage4_pack/build_text_index.py
  - Formato: {"slug": [faiss_idx, ...]}
  - Slug: nome+brand normalizado (sem acentos, só a-z0-9)
  - Resolve casos onde a foto é difícil mas o nome é exato

Singleton: instanciado uma vez no startup do app e reutilizado.
Falha graceful: se DINOv2 não estiver disponível, devolve bbox_claude inalterada.
"""
import logging
import re
import threading
import unicodedata
from pathlib import Path

import numpy as np

_log = logging.getLogger("casaiq.gazetteer")

# Caminhos padrão (sobrescrevíveis via config)
_DEFAULT_EMB      = Path.home() / "projetos/casaiq/embeddings.npy"
_DEFAULT_FAISS    = Path.home() / "projetos/casaiq/data/gazetteer/gazetteer.faiss"
_DEFAULT_MAP      = Path.home() / "projetos/casaiq/data/gazetteer/index_mapping.jsonl"
_DEFAULT_TEXT_IDX = Path.home() / "projetos/casaiq/data/gazetteer/text_index.json"

# Resolução do patch grid (deve ser múltiplo de 14, patch size do DINOv2)
_COLLECTIVE_SZ = 896
_PATCH         = 14
_TOP_K_PCT     = 0.08   # top 8% dos patches para cluster bbox
_FAISS_MIN_SIM = 0.75   # só usa embedding do gazetteer se dist >= esse limiar


class GazetteerMatcher:
    """Singleton thread-safe. Use `get_instance()` em vez de construir diretamente."""

    _instance: "GazetteerMatcher | None" = None
    _lock = threading.Lock()

    def __init__(self, emb_path: Path, faiss_path: Path, map_path: Path,
                 text_idx_path: Path = _DEFAULT_TEXT_IDX,
                 model_name: str = "facebook/dinov2-small"):
        import faiss
        import torch
        import torch.nn.functional as F
        import torchvision.transforms.functional as TF
        from transformers import AutoImageProcessor, AutoModel

        self._torch  = torch
        self._F      = F
        self._TF     = TF

        _log.info("gazetteer_init_start", extra={"model": model_name})

        # DINOv2
        self._proc  = AutoImageProcessor.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).eval()
        self._dim   = self._model.config.hidden_size   # 384

        # Gazetteer embeddings
        self._emb = np.load(emb_path).astype("float32")  # (N, 384)
        self._idx = faiss.read_index(str(faiss_path))

        import json
        with open(map_path) as f:
            self._mapping = [json.loads(l) for l in f]

        # Índice texto→embedding: slug → lista de faiss_idx
        self._text_idx: dict[str, list[int]] = {}
        if text_idx_path.exists():
            self._text_idx = json.loads(text_idx_path.read_text())
            _log.info("text_index_carregado", extra={"n_slugs": len(self._text_idx)})
        else:
            _log.info("text_index_ausente", extra={"path": str(text_idx_path)})

        n = self._idx.ntotal
        _log.info("gazetteer_init_done", extra={
            "model": model_name, "n_embeddings": n, "dim": self._dim,
            "text_index": len(self._text_idx),
        })

    # ── texto → embedding ────────────────────────────────────────────────────

    @staticmethod
    def _slug_norm(s: str) -> str:
        """Normalização agressiva igual à do consolidate.py: lower, sem acento, só [a-z0-9 ]."""
        if not s:
            return ""
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
        s = re.sub(r"[^a-zA-Z0-9 ]", " ", s).lower()
        return re.sub(r"\s+", " ", s).strip()

    def buscar_por_nome(self, nome: str, brand: str = "") -> "np.ndarray | None":
        """
        Lookup texto→embedding no índice pré-computado.

        Tenta slugs em ordem de especificidade:
          1. slug(nome + brand)  — mais específico
          2. slug(nome)          — só nome
        Retorna o embedding médio dos hits, ou None se índice vazio/sem match.
        """
        if not self._text_idx:
            return None

        candidatos: list[np.ndarray] = []
        for slug in [self._slug_norm(f"{nome} {brand}"), self._slug_norm(nome)]:
            if not slug:
                continue
            idxs = self._text_idx.get(slug)
            if idxs:
                candidatos = [self._emb[i] for i in idxs[:3]]  # max 3 hits
                break

        if not candidatos:
            return None

        vec = np.mean(candidatos, axis=0).astype("float32")
        # Re-normaliza após média
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    # ── encoding ──────────────────────────────────────────────────────────────

    @property
    def _no_grad(self):
        return self._torch.no_grad()

    def _encode_global(self, img) -> np.ndarray:
        """CLS token 384-dim, L2-normalizado."""
        from PIL import Image
        img_r = img.resize((448, 448), Image.LANCZOS)
        with self._torch.no_grad():
            inp = self._proc(images=img_r, return_tensors="pt")
            out = self._model(**inp)
            cls = self._F.normalize(out.last_hidden_state[:, 0, :], dim=-1)
        return cls[0].numpy().astype("float32")

    def _encode_patches(self, img) -> np.ndarray:
        """
        Grid (64, 64, 384) via tiling 4×4 da imagem redimensionada para 896px.
        Cada tile 224px (resolução nativa do DINOv2) é processado nativamente,
        sem precisar de interpolate_pos_encoding. Batch único → ~1 forward pass.
        """
        from PIL import Image
        _tile  = 224
        _tiles = _COLLECTIVE_SZ // _tile   # 4

        img_r = img.resize((_COLLECTIVE_SZ, _COLLECTIVE_SZ), Image.LANCZOS)
        tiles = [
            img_r.crop((c * _tile, r * _tile, (c + 1) * _tile, (r + 1) * _tile))
            for r in range(_tiles) for c in range(_tiles)
        ]   # 16 tiles, ordem row-major

        with self._torch.no_grad():
            inp  = self._proc(images=tiles, return_tensors="pt")
            out  = self._model(**inp)
            feats = out.last_hidden_state[:, 1:, :]   # (16, 256, 384) — skip CLS
            feats = self._F.normalize(feats, dim=-1)

        ppt = int(feats.shape[1] ** 0.5)   # patches por tile = 16
        grid = feats.reshape(_tiles, _tiles, ppt, ppt, self._dim)
        grid = grid.permute(0, 2, 1, 3, 4).reshape(
            _tiles * ppt, _tiles * ppt, self._dim
        )   # (64, 64, 384)
        return grid.numpy().astype("float32")

    # ── gazetteer lookup ──────────────────────────────────────────────────────

    def _lookup(self, query_vec: np.ndarray, k: int = 1):
        """Retorna (gaz_vec, dist_top1) ou (None, 0) se índice indisponível."""
        try:
            q = query_vec.reshape(1, -1)
            D, I = self._idx.search(q, k)
            dist = float(D[0][0])
            vec  = self._emb[int(I[0][0])]
            return vec, dist
        except Exception as e:
            _log.warning("gazetteer_lookup_erro", extra={"erro": str(e)})
            return None, 0.0

    # ── prior espacial ───────────────────────────────────────────────────────

    @staticmethod
    def _spatial_prior(centroide: dict, S: int, sigma_frac: float = 0.20) -> np.ndarray:
        """
        Gaussiana 2D centrada no centroide do objeto.

        centroide: {"cx": float, "cy": float} em coords normalizadas [0,1].
        sigma_frac: raio da Gaussiana como fração do tamanho do grid (padrão 20%).
                    Para S=64: sigma = 0.20 × 64 = 12.8 patches ≈ 28% da imagem.

        Retorna (S, S) normalizado em [0.15, 1.0].
        O pico (=1.0) está em (cx*S, cy*S); cai para ~0.15 a >3σ de distância.

        Efeito: patches próximos ao centroide recebem boost; patches distantes
        são suprimidos. Com dois objetos similares na mesma bbox, o centroide
        desambigua qual deles o Claude está descrevendo.
        """
        cx = float(centroide.get("cx", 0.5)) * S
        cy = float(centroide.get("cy", 0.5)) * S
        sigma = sigma_frac * S

        ys = np.arange(S, dtype=float)
        xs = np.arange(S, dtype=float)
        grid_x, grid_y = np.meshgrid(xs, ys)         # (S, S)

        dist2 = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
        prior = np.exp(-dist2 / (2 * sigma ** 2))    # (S, S) em [0, 1]

        # Floor 0.15 — preserva sinal DINOv2 mesmo longe do centroide
        prior = 0.15 + 0.85 * prior
        return prior.astype("float32")

    # ── máscara de cor ────────────────────────────────────────────────────────

    def _color_weight(self, img, cores: list[dict], S: int,
                      n_cores: int = 3) -> np.ndarray:
        """
        Vetor de assinatura de cor — top-N cores dominantes (padrão N=3).

        Benchmark empírico (5 objetos, 3 fotos, cenas simples):
          N=1 → IoU +0.000 (nulo — cosseno K=1 degenera)
          N=2 → IoU +0.074 (melhor em cenas simples)
          N=3 → IoU +0.071 (quase igual; melhor em cenas densas)
          N≥4 → IoU +0.071 (estabiliza, sem ganho adicional)

        N=3 é o padrão mais seguro: em cenas simples empata com N=2 (Δ=0.003),
        mas em cenas densas (muitos objetos similares) a 3ª cor discrimina
        melhor — ex: 15+ chaves de fenda onde bege-translúcido e prata-metálico
        separam grupos de handles que seriam idênticos com só 2 cores.

        Para cada patch (S×S) calcula um vetor K-dimensional de resposta às
        K cores-alvo fornecidas pelo Claude. Compara esse vetor com as proporções
        esperadas (area_pct) via similaridade cosseno.

        Isso discrimina objetos com a mesma cor dominante mas proporções diferentes
        (ex: dois alicates amarelo+preto, onde um tem amarelo mais saturado ou
        mais abundante que o outro).

        cores: lista de até 3 {"hex": "#RRGGBB", "area_pct": float, ...}
               ordenada do mais ao menos abundante.
        Retorna array (S, S) em [0.2, 1.0].
        """
        from PIL import Image as _PILImage
        import re

        def hex_to_rgb(h: str) -> np.ndarray:
            h = h.lstrip("#")
            return np.array([int(h[i:i+2], 16) for i in (0, 2, 4)], dtype=float)

        # ── parse e valida cores ──────────────────────────────────────────────
        validas = []
        for c in cores[:n_cores]:   # N=3 padrão — melhor equilíbrio entre cenas simples e densas
            h = c.get("hex", "")
            if not h or not re.match(r"^#[0-9A-Fa-f]{6}$", h):
                continue
            validas.append((hex_to_rgb(h), float(c.get("area_pct", 25))))

        if not validas:
            return np.ones((S, S))

        # ── média RGB por patch ───────────────────────────────────────────────
        tile = _COLLECTIVE_SZ   # 896
        img_r = img.resize((tile, tile), _PILImage.LANCZOS)
        arr = np.array(img_r, dtype=float)          # (896, 896, 3)
        p = _PATCH                                   # 14
        patches_rgb = arr.reshape(S, p, S, p, 3).mean(axis=(1, 3))  # (S, S, 3)

        # ── sims: proximidade gaussiana — calculado UMA VEZ ──────────────────
        sigma = 30.0
        K     = len(validas)
        sims  = np.zeros((S, S, K), dtype=float)
        prop  = np.zeros(K,         dtype=float)
        for k, (rgb_k, area_k) in enumerate(validas):
            dist        = np.linalg.norm(patches_rgb - rgb_k, axis=2)
            sims[:,:,k] = np.exp(-(dist ** 2) / (2 * sigma ** 2))
            prop[k]     = area_k

        # ── componente Gaussiana — apenas cores cromáticas ───────────────────
        # Cores acromáticas (preto, branco, cinza, prata) são onipresentes na
        # cena (cabos de ferramentas, sombras, metal). Incluí-las no Gaussian
        # amplifica patches comuns em toda a bbox → expansão ou sobre-aperto.
        # Solução: Gaussian só para cores com saturação > 25%
        # (sat = (max_RGB - min_RGB) / 255 > 0.25).
        SAT_MIN = 0.25
        gauss        = np.zeros((S, S), dtype=float)
        chrom_weight = 0.0
        total_weight = prop.sum() + 1e-9
        for k, (rgb_k, area_k) in enumerate(validas):
            sat = float(rgb_k.max() - rgb_k.min()) / 255.0
            if sat < SAT_MIN:
                continue                            # achromático: ignora no Gaussian
            gauss        += sims[:, :, k] * area_k
            chrom_weight += area_k

        chrom_frac = chrom_weight / total_weight    # 0–1: fração cromática do objeto
        if gauss.max() > 0:
            gauss = gauss / gauss.max()             # [0, 1]
        else:
            gauss = np.ones((S, S))                 # sem cor cromática → neutro
        # Se objeto tem baixa cromaticidade (ex: chave preta), blend suaviza o sinal
        gauss = chrom_frac * gauss + (1 - chrom_frac) * np.ones((S, S))

        # ── componente Cosseno — TODAS as K cores (inclui acromáticas) ────────
        # O cosseno não sofre do problema de onipresença: ele testa PROPORÇÃO.
        # Patches com mix certo recebem score alto; magnitudes são normalizadas.
        prop_n    = prop / (prop.sum() + 1e-9)
        sims_unit = sims / (np.linalg.norm(sims, axis=2, keepdims=True) + 1e-9)
        prop_unit = prop_n / (np.linalg.norm(prop_n) + 1e-9)
        cosine    = np.clip((sims_unit * prop_unit).sum(axis=2), 0, 1)

        # ── média geométrica: sqrt(G × C) ─────────────────────────────────────
        # G discrimina magnitude cromática; C discrimina proporção do mix.
        # Patches que "passam nos dois testes" recebem score alto.
        combined = np.sqrt(gauss * cosine)
        if combined.max() <= 0:
            return np.ones((S, S))
        combined = combined / combined.max()
        combined = 0.15 + 0.85 * combined           # [0.15, 1.0]
        return combined.astype("float32")

    # ── cluster bbox ──────────────────────────────────────────────────────────

    @staticmethod
    def _cluster_bbox(heatmap: np.ndarray, canvas_w: int, canvas_h: int,
                      top_k_pct: float = _TOP_K_PCT):
        """
        Seleciona top-K% patches e retorna bbox (x1, y1, x2, y2) em pixels.
        """
        H, W = heatmap.shape
        k = max(1, int(H * W * top_k_pct))
        flat = heatmap.flatten()
        top_idx = np.argpartition(flat, -k)[-k:]
        rows, cols = np.unravel_index(top_idx, (H, W))

        scale_x = canvas_w / (W * _PATCH)
        scale_y = canvas_h / (H * _PATCH)

        c0 = int(cols.min() * _PATCH * scale_x)
        c1 = int((cols.max() + 1) * _PATCH * scale_x)
        r0 = int(rows.min() * _PATCH * scale_y)
        r1 = int((rows.max() + 1) * _PATCH * scale_y)

        return (
            max(0, c0), max(0, r0),
            min(canvas_w, c1), min(canvas_h, r1),
        )

    # ── API pública ───────────────────────────────────────────────────────────

    def refinar_bbox(self, foto_path: str, bbox_norm: dict,
                     nome: str = "", brand: str = "",
                     cores: list | None = None,
                     centroide: dict | None = None,
                     margem: float = 0.04) -> dict:
        """
        Recebe bbox_normalizada do Claude, devolve bbox refinada pelo DINOv2.

        Sinais combinados (quando disponíveis):
          DINOv2    — similaridade visual com produto canônico (gazetteer)
          cor       — √(Gaussiana_cromática × Cosseno_proporção)
          centroide — prior espacial Gaussiana 2D + ancoragem da bbox final

        Estratégia:
          1. Encoda crop da região do Claude → query vec.
          2. FAISS lookup → match_vec canônico (ou query_vec como fallback).
          3. Patch grid da foto inteira (64×64).
          4. heatmap = DINOv2 × [cor] × [spatial_prior(centroide)]
          5. Top-K% patches dentro da bbox do Claude → bbox dos patches.
          6. Se centroide fornecido: re-ancora a bbox no centroide
             (preserva tamanho derivado dos patches, move centro para onde
             Claude indicou o objeto de fato estar).
        """
        from PIL import Image, ImageOps
        try:
            img = Image.open(foto_path)
            img = ImageOps.exif_transpose(img).convert("RGB")
            w, h = img.size

            # 1. Crop com margem → encode global
            cx1 = max(0.0, bbox_norm["x1"] - margem)
            cy1 = max(0.0, bbox_norm["y1"] - margem)
            cx2 = min(1.0, bbox_norm["x2"] + margem)
            cy2 = min(1.0, bbox_norm["y2"] + margem)
            crop = img.crop((cx1 * w, cy1 * h, cx2 * w, cy2 * h))
            query_vec = self._encode_global(crop)

            # 2. match_vec: texto → FAISS visual → live crop
            text_vec = self.buscar_por_nome(nome, brand) if nome else None
            if text_vec is not None:
                match_vec = text_vec
                gaz_dist  = 1.0
                _log.debug("match_via_texto", extra={"nome": nome, "brand": brand})
            else:
                gaz_vec, gaz_dist = self._lookup(query_vec)
                if gaz_vec is not None and gaz_dist >= _FAISS_MIN_SIM:
                    match_vec = gaz_vec
                else:
                    match_vec = query_vec

            # 3. Patch grid (64×64×384)
            grid = self._encode_patches(img)
            S = grid.shape[0]

            # 4. Heatmap composto: DINOv2 × cor
            #    O centroide NÃO entra no heatmap — entra apenas na etapa 6
            #    (re-ancoragem). Aplicar prior espacial aqui comprimiria o
            #    cluster de patches → bbox pequena → re-ancoragem com hw/hh
            #    minúsculos. As duas operações fazem coisas similares; usar
            #    só a re-ancoragem é mais direto e preserva a estimativa de
            #    TAMANHO vinda dos patches.
            heatmap = grid @ match_vec                          # DINOv2 base

            if cores:
                heatmap = heatmap * self._color_weight(img, cores, S)

            # 5. Top-K% dentro da bbox do Claude
            pr0 = int(cy1 * S);  pr1 = min(S, int(cy2 * S) + 1)
            pc0 = int(cx1 * S);  pc1 = min(S, int(cx2 * S) + 1)
            sub_hm = heatmap[pr0:pr1, pc0:pc1]
            if sub_hm.size == 0:
                return bbox_norm

            k = max(1, int(sub_hm.size * _TOP_K_PCT))
            top_idx = np.argpartition(sub_hm.flatten(), -k)[-k:]
            rows_l, cols_l = np.unravel_index(top_idx, sub_hm.shape)
            rows_g = rows_l + pr0
            cols_g = cols_l + pc0

            scale_x = w / S
            scale_y = h / S
            px1 = int(cols_g.min() * scale_x)
            py1 = int(rows_g.min() * scale_y)
            px2 = int((cols_g.max() + 1) * scale_x)
            py2 = int((rows_g.max() + 1) * scale_y)

            # 6. Re-ancora no centroide (preserva tamanho, move centro)
            #    Os patches dizem QUAL O TAMANHO do objeto.
            #    O centroide diz ONDE o objeto está de fato.
            #    Combinar elimina bbox assimétricas causadas por patches outliers.
            if centroide:
                cx_px = float(centroide["cx"]) * w
                cy_px = float(centroide["cy"]) * h
                hw = (px2 - px1) / 2    # half-width dos patches
                hh = (py2 - py1) / 2    # half-height dos patches
                px1 = int(cx_px - hw)
                px2 = int(cx_px + hw)
                py1 = int(cy_px - hh)
                py2 = int(cy_px + hh)

            px1, py1 = max(0, px1), max(0, py1)
            px2, py2 = min(w, px2), min(h, py2)

            # 7. Normaliza e valida
            refined = {
                "x1": px1 / w, "y1": py1 / h,
                "x2": px2 / w, "y2": py2 / h,
            }
            area_r = (refined["x2"] - refined["x1"]) * (refined["y2"] - refined["y1"])
            area_c = (bbox_norm["x2"] - bbox_norm["x1"]) * (bbox_norm["y2"] - bbox_norm["y1"])
            if area_r < 0.005 or area_r > 0.90 or area_r > area_c * 2.5:
                _log.warning("bbox_refinada_invalida_usando_claude",
                             extra={"area_r": round(area_r, 4),
                                    "area_c": round(area_c, 4), "foto": foto_path})
                return bbox_norm

            _log.info("bbox_refinada", extra={
                "area_claude": round(area_c, 3),
                "area_dino":   round(area_r, 3),
                "gaz_dist":    round(gaz_dist, 4),
                "centroide":   centroide is not None,
                "match_mode":  "texto" if text_vec is not None else (
                               "faiss_visual" if gaz_dist >= _FAISS_MIN_SIM else "visual"),
                "foto":        foto_path,
            })
            return refined

        except Exception as e:
            _log.warning("refinar_bbox_erro_usando_claude",
                         extra={"erro": str(e), "foto": foto_path})
            return bbox_norm

    # ── singleton ─────────────────────────────────────────────────────────────

    @classmethod
    def get_instance(cls) -> "GazetteerMatcher | None":
        """
        Retorna a instância singleton, inicializando na primeira chamada.
        Retorna None se gazetteer estiver desabilitado ou arquivos ausentes.
        """
        if cls._instance is not None:
            return cls._instance
        with cls._lock:
            if cls._instance is not None:
                return cls._instance
            from core.config import (
                GAZETTEER_ENABLED, GAZETTEER_EMB_PATH,
                GAZETTEER_FAISS_PATH, GAZETTEER_MAP_PATH,
            )
            if not GAZETTEER_ENABLED:
                _log.info("gazetteer_desabilitado")
                return None
            for p in [GAZETTEER_EMB_PATH, GAZETTEER_FAISS_PATH, GAZETTEER_MAP_PATH]:
                if not Path(p).exists():
                    _log.warning("gazetteer_arquivo_ausente", extra={"path": str(p)})
                    return None
            try:
                cls._instance = cls(
                    emb_path=Path(GAZETTEER_EMB_PATH),
                    faiss_path=Path(GAZETTEER_FAISS_PATH),
                    map_path=Path(GAZETTEER_MAP_PATH),
                )
            except Exception as e:
                _log.error("gazetteer_init_falhou", extra={"erro": str(e)})
                return None
        return cls._instance
