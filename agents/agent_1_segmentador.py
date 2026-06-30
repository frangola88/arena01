"""
Agente 1: Segmentador v5 — Skill semântica + CV geométrico (match por centroide).

ARQUITETURA (v5):

  1. SKILL (Claude) analisa foto inteira → produz lista de objetos com
     bbox_normalizada e atributos ricos.

  2. OpenCV roda em paralelo → produz contornos com bordas exatas.

  3. Para CADA objeto da skill, gera o crop em cascata de qualidade:

       A. SKILL_BBOX + CV (melhor):
          - Skill tem bbox → busca contorno CV cujo CENTROIDE está dentro
            desse bbox → usa o contorno CV para o recorte (bordas precisas).

       B. SKILL_BBOX direto (ok, só para bboxes pequenos):
          - Skill tem bbox, mas nenhum centroide CV cai dentro dela
          - Aceita apenas se bbox_area < 30% da foto (bbox pequena = mais confiável)

       C. CV fallback (pior):
          - Sem bbox válida da skill → usa próximo contorno CV disponível
          - Risco: contorno pode não corresponder ao objeto certo

       D. FOTO INTEIRA (nunca erra):
          - Nenhuma opção acima funcionou → usa a foto inteira como ícone
          - Honesto: mostra o contexto completo sem risco de crop errado

  MOTIVAÇÃO DA MUDANÇA:
  Claude bboxes são semanticamente corretos (sabem onde o objeto está)
  mas geometricamente imprecisas para cenas densas. OpenCV é geometricamente
  preciso mas semanticamente cego (não sabe qual contorno é qual objeto).
  O match por centroide combina os dois: "procure o contorno cujo centro
  está na região que o Claude indicou" → recorte preciso.
"""
import logging
from pathlib import Path
from PIL import Image, ImageOps
from core.config import RECORTES_DIR
from core.visao_global import analisar_foto_completa, _CACHE_ANALISE
from core.analise_cena import analisar_cena
from core.verificacoes_cruzadas import verificar_cena, score_qualidade, w_borda_bbox, LIM_BORDA, FAIXA_BORDA
from core.segmentacao_cv import detectar_objetos, gerar_icone_anotado
from core.gazetteer import GazetteerMatcher
from core.crop_refinador import refinar_crop
from core.vetorizador_raster import vetorizar_superficie, casar_poligono_a_bbox

_log = logging.getLogger("casaiq.agent_1")


# Lista-bloqueio: nomes que nunca devem virar item de inventário
PALAVRAS_BLOQUEADAS = {
    "mesa", "piso", "chao", "chão", "parede", "tampo", "bancada", "balcao", "balcão",
    "prateleira", "armario", "armário", "gaveta", "porta", "fundo", "base", "superficie",
    "superfície", "azulejo", "ladrilho", "padrao", "padrão", "textura",
    "sombra", "borrão", "borrao", "reflexo",
    "indefinido", "desconhecido", "nada", "vazio", "nenhum",
}


def _nome_eh_aceitavel(nome: str) -> bool:
    if not nome or not nome.strip():
        return False
    n = nome.lower().strip()
    for bloq in PALAVRAS_BLOQUEADAS:
        if bloq in n:
            return False
    return True


def _encontrar_contorno_cv_em_bbox(
    bbox_norm: dict,
    contornos_cv: list[dict],
    largura: int,
    altura: int,
    indices_usados: set,
) -> tuple[int | None, dict | None]:
    """
    Procura o contorno OpenCV cujo CENTROIDE cai dentro da bbox normalizada.

    Estratégia: o centroide de um contorno é o ponto mais representativo do
    objeto. Se o Claude disse "o objeto X está nessa região", qualquer contorno
    cujo centro esteja nessa região é um bom candidato para representar X.

    Returns:
        (idx, contorno_dict) se encontrar, ou (None, None) se não encontrar.
        Quando há múltiplos candidatos, retorna o de maior área (mais representativo).
    """
    x1_px = int(bbox_norm["x1"] * largura)
    y1_px = int(bbox_norm["y1"] * altura)
    x2_px = int(bbox_norm["x2"] * largura)
    y2_px = int(bbox_norm["y2"] * altura)

    candidatos = []
    for idx, cv_obj in enumerate(contornos_cv):
        if idx in indices_usados:
            continue
        cx, cy = cv_obj["centro"]
        if x1_px <= cx <= x2_px and y1_px <= cy <= y2_px:
            candidatos.append((idx, cv_obj))

    if not candidatos:
        return None, None

    # Prefere o contorno de maior área dentro da bbox (objeto mais representativo)
    candidatos.sort(key=lambda t: t[1]["area_pct"], reverse=True)
    return candidatos[0]


def _bbox_normalizada_para_pixels(bbox_norm: dict, largura: int, altura: int,
                                   margem_pct: float = 0.02) -> tuple[int, int, int, int]:
    """
    Converte bbox normalizada {x1,y1,x2,y2 em 0-1} para (x, y, w, h) em pixels.
    Adiciona pequena margem percentual.
    """
    x1 = max(0.0, bbox_norm["x1"] - margem_pct)
    y1 = max(0.0, bbox_norm["y1"] - margem_pct)
    x2 = min(1.0, bbox_norm["x2"] + margem_pct)
    y2 = min(1.0, bbox_norm["y2"] + margem_pct)

    x_px = int(x1 * largura)
    y_px = int(y1 * altura)
    w_px = int((x2 - x1) * largura)
    h_px = int((y2 - y1) * altura)
    return x_px, y_px, w_px, h_px


def _dimensoes_imagem(caminho: str) -> tuple[int, int]:
    """Retorna (largura, altura) considerando rotação EXIF."""
    pil = Image.open(caminho)
    pil = ImageOps.exif_transpose(pil)
    return pil.size  # (w, h)


def _gate_borda(
    bbox_refinada: dict | None,
    bbox_original: dict | None,
    cena,
    *,
    foto_id=None,
    nome=None,
) -> dict | None:
    """
    Gate W(y): rejeita bbox refinada se o sinal de borda indicar objeto cortado.

    Calcula w_borda_bbox(bbox_refinada, cena, faixa=FAIXA_BORDA, th=LIM_BORDA).
    Se wb["max"] > LIM_BORDA (0.5), loga "objeto_cortado_rejeitando_bbox_refinada"
    e retorna bbox_original (fallback para a bbox original do Claude).
    Caso contrário retorna bbox_refinada (mantém refinamento DINOv2).

    Comportamento-idêntico ao bloco inline que substituiu.
    """
    wb = w_borda_bbox(bbox_refinada, cena, faixa=FAIXA_BORDA, th=LIM_BORDA)
    if wb and wb["max"] > LIM_BORDA:
        _log.warning("objeto_cortado_rejeitando_bbox_refinada", extra={
            "nome": nome, "w_max": round(wb["max"], 3),
            "cortado_em": wb["cortado"], "foto_id": foto_id,
        })
        return bbox_original
    return bbox_refinada


def segmentar_foto(caminho_foto: str, foto_id: int) -> list[dict]:
    """
    Pipeline v4: skill manda em quantidade + nomes + bboxes.

    Returns:
        Lista de {"nome": str, "recorte_path": str}.
    """
    _log.info("segmentando_v4_skill_first",
              extra={"caminho_foto": caminho_foto, "foto_id": foto_id})

    # ─── 0. PRÉ-ANÁLISE DE CENA: tomografia leve, sem IA (~0.2-0.7s) ───
    # Estima nº de objetos, complexidade e cortes de borda a partir de seções
    # ortogonais da variância local. Vira pista não-vinculante para o prompt.
    cena = analisar_cena(caminho_foto)
    _log.info("pre_analise_cena", extra={
        "n_obj_estimado": cena.n_objetos_estimado,
        "complexidade": cena.complexidade,
        "bordas": cena.bordas_ativas,
        "tempo_s": cena.tempo_s,
        "foto_id": foto_id,
    })

    # ─── 0b. VETORIZAÇÃO RASTER→VECTOR: aditiva, não bloqueia ───────────────
    # Vetoriza a superfície de object-ness passando as bboxes dos objetos para
    # calcular iou_vs_claude por polígono. Resultado guardado em _vet_resultado
    # para casar objeto↔polígono no loop abaixo (via casar_poligono_a_bbox).
    # Falha silenciosa: todos os objetos recebem geometria_vetor="" e o
    # pipeline continua normalmente.
    _vet_resultado: dict | None = None
    try:
        # As bboxes dos objetos são coletadas AQUI para a vetorização única.
        # Neste ponto objetos_skill ainda não foi definido (vem após a skill),
        # então passamos lista vazia — o match por IoU acontece depois no loop.
        _vet_resultado = vetorizar_superficie(cena.superficie)
        _log.info("vetorizacao_raster_resumo", extra={
            "total_polys":  _vet_resultado["stats"]["total_polys"],
            "total_area":   _vet_resultado["stats"]["total_area"],
            "coverage_pct": _vet_resultado["stats"]["coverage_pct"],
            "latency_ms":   _vet_resultado["performance"]["latency_ms"],
            "geojson_len":  len(_vet_resultado["geojson"]),
            "foto_id":      foto_id,
        })
    except Exception as _e:
        _log.warning("vetorizacao_raster_falhou", extra={"erro": str(_e), "foto_id": foto_id})

    # ─── 1. SKILL: análise rica completa (com pista da cena) ───
    analise = analisar_foto_completa(caminho_foto, hint_cena=cena.resumo_prompt())
    _CACHE_ANALISE[caminho_foto] = analise  # garante cache para o pipeline reusar

    objetos_skill = analise.get("objetos", [])
    _log.info("skill_resultado", extra={
        "total": len(objetos_skill),
        "nomes": [o["nome"] for o in objetos_skill],
        "com_bbox": sum(1 for o in objetos_skill if o.get("bbox_normalizada")),
        "sem_bbox": sum(1 for o in objetos_skill if not o.get("bbox_normalizada")),
        "foto_id": foto_id,
    })

    # ─── 2. OpenCV: roda em paralelo (debug visual + fallback geométrico) ───
    debug_path = str(Path(caminho_foto).with_name(
        Path(caminho_foto).stem + "_debug.jpg"
    ))
    contornos_cv = detectar_objetos(caminho_foto, debug_path=debug_path)
    _log.info("opencv_fallback_disponivel", extra={
        "contornos": len(contornos_cv), "foto_id": foto_id,
    })

    if not objetos_skill:
        _log.warning("skill_vazia_sem_resultado", extra={"foto_id": foto_id})
        return []

    # ─── 3. Para cada objeto da skill, gerar ícone anotado ────────────────────
    # ESTRATÉGIA v6: a foto inteira É o ícone — cada objeto recebe uma versão
    # anotada com spotlight (fundo escurecido + destaque na região do objeto)
    # e um badge numerado. Assim:
    #   • Nunca mostramos o objeto errado (régua em vez de chave)
    #   • O usuário vê o contexto completo da foto
    #   • O bbox da skill é usado apenas para o DESTAQUE visual, não para crop
    #   • Mesmo se o bbox for ligeiramente impreciso, o contexto completo compensa
    largura, altura = _dimensoes_imagem(caminho_foto)
    resultado = []

    # Carrega GazetteerMatcher (singleton, None se desabilitado/indisponível)
    gaz = GazetteerMatcher.get_instance()

    # Filtra só os objetos aceitáveis para saber o total e numerar corretamente
    objetos_aceitaveis = [o for o in objetos_skill if _nome_eh_aceitavel(o.get("nome", ""))]
    total_objetos = len(objetos_aceitaveis)

    # ─── Verificações cruzadas: clássico (etapa 0) × Claude (etapa 1) ─────────
    # analise_cena não usa IA, então divergências revelam onde o Claude
    # provavelmente errou. Não bloqueia — loga (warning se divergir) e anota a
    # suspeita por objeto no resultado. Primeiro passo do score_qualidade.
    verif = verificar_cena(cena, objetos_aceitaveis)
    _diverge = (verif["contagem"]["veredito"] not in ("ok", "sem_sinal")
                or verif["n_suspeitos_bg"] > 0)
    (_log.warning if _diverge else _log.info)("verificacao_cruzada", extra={
        "ratio_contagem": verif["contagem"],
        "centroide_em_bg": verif["centroide_em_bg"],
        "foto_id": foto_id,
    })

    for numero, obj in enumerate(objetos_aceitaveis, start=1):
        nome = obj["nome"]
        icone_path = str(RECORTES_DIR / f"foto_{foto_id}_obj_{numero:02d}.jpg")
        bbox_norm = obj.get("bbox_normalizada")

        # Refinamento DINOv2 com 3 sinais: visual + cor + centroide
        cores      = obj.get("cores_dominantes") or []
        centroide  = obj.get("centroide_normalizado") or None
        bbox_antes = bbox_norm  # guarda original para fallback W(y)
        if gaz is not None and bbox_norm is not None:
            bbox_norm = gaz.refinar_bbox(
                caminho_foto, bbox_norm,
                nome=nome,
                cores=cores,
                centroide=centroide,
            )
            _log.debug("bbox_refinada_dino", extra={
                "nome":      nome,
                "bbox":      bbox_norm,
                "n_cores":   len(cores),
                "centroide": centroide,
            })

            # Gate W(y): rejeita objetos cortados (object-ness alta fora da bbox)
            bbox_norm = _gate_borda(bbox_norm, bbox_antes, cena, foto_id=foto_id, nome=nome)

        obj_com_bbox = {**obj, "bbox_normalizada": bbox_norm} if bbox_norm else obj

        # ── Crop refinado (2 estágios + gatilho 3º) ──────────────────────────
        fonte_crop = "spotlight_fallback"
        try:
            _, fonte_crop = refinar_crop(
                caminho_foto=caminho_foto,
                obj=obj_com_bbox,
                saida=icone_path,
                numero=numero,
                total_objetos=total_objetos,
                nome=nome,
            )
            _log.info("crop_refinado_gerado", extra={
                "nome": nome, "numero": numero, "fonte": fonte_crop,
                "foto_id": foto_id,
            })
        except Exception as e:
            # Fallback seguro: spotlight na foto inteira
            _log.warning("crop_refinador_falhou_fallback_spotlight", extra={
                "nome": nome, "erro": str(e), "foto_id": foto_id,
            })
            try:
                gerar_icone_anotado(
                    caminho_foto=caminho_foto,
                    saida=icone_path,
                    bbox_norm=bbox_norm,
                    numero=numero,
                    total=total_objetos,
                    nome=nome,
                )
            except Exception as e2:
                _log.warning("fallback_spotlight_falhou", extra={
                    "nome": nome, "erro": str(e2), "foto_id": foto_id,
                })
                continue

        # ── Score de qualidade: combina sinais clássicos + Claude + DINOv2 ───
        # heatmap DINOv2 vem por side-channel do refinar_bbox (None se gaz off).
        heatmap_dino = getattr(gaz, "ultimo_heatmap", None) if gaz is not None else None
        sq = score_qualidade(obj_com_bbox, cena, heatmap=heatmap_dino)
        if sq["flags"]:
            _log.info("score_qualidade_flags", extra={
                "nome": nome, "numero": numero, "score": sq["score"],
                "flags": sq["flags"], "foto_id": foto_id,
            })

        # ── Casamento objeto↔polígono vetorial por IoU ───────────────────────
        # Para cada objeto, casa o melhor polígono da vetorização raster→vector
        # usando a bbox do objeto (após refinamento DINOv2). Usa casar_poligono_a_bbox
        # que retorna GeoJSON Feature ou "" se IoU < IOU_MATCH_MIN ou falha.
        geometria_vetor = ""
        if _vet_resultado is not None and bbox_norm is not None:
            try:
                geometria_vetor = casar_poligono_a_bbox(
                    _vet_resultado,
                    bbox_norm,
                    cena.superficie.shape,
                )
                _log.info("geometria_vetor_casada", extra={
                    "nome":    nome,
                    "tem_geom": bool(geometria_vetor),
                    "foto_id": foto_id,
                })
            except Exception as _ge:
                _log.warning("casar_poligono_falhou", extra={
                    "nome": nome, "erro": str(_ge), "foto_id": foto_id,
                })

        resultado.append({
            "nome": nome,
            "recorte_path": icone_path,
            "_fonte_bbox": fonte_crop,
            "_suspeita_bg": "centroide_em_bg" in sq["flags"],
            "_score_qualidade": sq["score"],
            "_flags_qualidade": sq["flags"],
            "geometria_vetor": geometria_vetor,
        })

    _log.info("segmentacao_v7_concluida", extra={
        "objetos_finais": len(resultado),
        "nomes": [r["nome"] for r in resultado],
        "S2_bbox":  sum(1 for r in resultado if r["_fonte_bbox"] == "S2_bbox"),
        "S3_bbox":  sum(1 for r in resultado if r["_fonte_bbox"] == "S3_bbox"),
        "fallback": sum(1 for r in resultado if "fallback" in r["_fonte_bbox"]),
        "suspeitas_bg": sum(1 for r in resultado if r.get("_suspeita_bg")),
        "score_baixo": sum(1 for r in resultado if r.get("_score_qualidade", 1.0) < 0.5),
        "score_medio": round(
            sum(r.get("_score_qualidade", 0.0) for r in resultado) / len(resultado), 3
        ) if resultado else 0.0,
        "foto_id": foto_id,
    })
    return resultado
