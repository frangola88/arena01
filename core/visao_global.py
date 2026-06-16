"""
Visão global da foto via skill estruturada.

Esta camada substitui parte significativa do pipeline antigo:
  - Agent 1 (segmentar nomes)
  - Agent 2 (analisar cada recorte)
  - Parte do Agent 3 (sugerir categoria)

O Claude vê a foto INTEIRA UMA vez e produz uma análise rica de cada objeto,
guiado pela skill em core/skills/inventario_visao.md.

Output: lista de dicts com TODOS os campos do schema da skill, prontos para inserir.
"""
import logging
from core.llm import chamar_visao, extrair_json
from core.skill_loader import carregar_skill
from core.database import get_db

_log = logging.getLogger("casaiq.visao_global")


# Categorias padrão se banco indisponível (fallback)
CATEGORIAS_FALLBACK = [
    "Ferramentas", "Utensílios Cozinha", "Eletrônicos", "Decoração", "Limpeza",
    "Têxtil", "Papelaria", "Higiene", "Outros",
]


# Correções de grafias erradas frequentes produzidas por LLMs em pt-BR
import re as _re

# Substituições regex: (padrão, substituto) — aplicadas em sequência
_CORRECOES_REGEX = [
    (_re.compile(r'\ball+icate\b'), "alicate"),     # allicate, alllicate, ...
    (_re.compile(r'\bmartello\b'), "martelo"),
    (_re.compile(r'\bparafusso\b'), "parafuso"),
    (_re.compile(r'\belétrico\b'), "elétrico"),     # normaliza acento
]

# Substituições exatas (aplicadas após regex)
_CORRECOES_EXATAS = {
    "chave de fenda philips": "chave de fenda phillips",
}


def _normalizar_nome(nome: str) -> str:
    """Corrige grafias erradas comuns de LLMs e normaliza espaços."""
    nome = " ".join(nome.split())
    for pat, sub in _CORRECOES_REGEX:
        nome = pat.sub(sub, nome)
    return _CORRECOES_EXATAS.get(nome, nome)


def _validar_bbox(bbox) -> dict | None:
    """
    Valida bbox normalizada vinda do LLM.
    Retorna dict {"x1","y1","x2","y2"} clipado a [0,1] ou None se inválida.
    """
    if not isinstance(bbox, dict):
        return None
    try:
        x1 = float(bbox.get("x1", 0))
        y1 = float(bbox.get("y1", 0))
        x2 = float(bbox.get("x2", 0))
        y2 = float(bbox.get("y2", 0))
    except (TypeError, ValueError):
        return None

    # Clipa a [0, 1]
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))

    # Validar geometria
    if x2 <= x1 or y2 <= y1:
        return None
    largura = x2 - x1
    altura = y2 - y1
    area = largura * altura

    # Filtros sanity:
    # - área entre 0.5% e 95% (pequenos demais = ruído; gigantes = foto inteira)
    # - lado mínimo 5% (caixa muito fininha provavelmente é erro)
    if area < 0.005 or area > 0.95:
        return None
    if largura < 0.03 or altura < 0.03:
        return None

    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _obter_categorias() -> list[str]:
    """Lê categorias do banco; usa fallback se vazio/erro."""
    try:
        conn = get_db()
        rows = conn.execute("SELECT nome FROM categorias ORDER BY nome").fetchall()
        conn.close()
        nomes = [r["nome"] for r in rows]
        return nomes if nomes else CATEGORIAS_FALLBACK
    except Exception as e:
        _log.warning("erro_buscar_categorias_usando_fallback", extra={"erro": str(e)})
        return CATEGORIAS_FALLBACK


def analisar_foto_completa(caminho_foto: str, hint_cena: str = "") -> dict:
    """
    Passa a foto pela skill 'inventario_visao' e retorna análise rica.

    Args:
        caminho_foto: caminho da imagem.
        hint_cena: pré-análise opcional (núm. estimado de objetos, complexidade,
            cortes de borda) vinda de core.analise_cena — injetada como pista
            não-vinculante no prompt. String vazia = sem pista.

    Returns:
        {
            "contexto_da_cena": str,
            "fundo": str,
            "objetos": [<dicts com schema completo da skill>]
        }
        Em caso de erro, retorna {"objetos": []} (sem quebrar pipeline).
    """
    categorias = _obter_categorias()
    categorias_str = "\n".join(f"- {c}" for c in categorias)

    bloco_cena = ""
    if hint_cena:
        bloco_cena = (
            "## Pré-análise da cena (pista — NÃO vinculante)\n\n"
            "Uma análise geométrica automática (sem IA) estimou o seguinte sobre esta foto. "
            "Use como orientação, não como verdade: confie no que você VÊ.\n\n"
            f"- {hint_cena}\n\n"
            "Se sua contagem divergir muito da estimativa, reveja a foto à procura de "
            "objetos pequenos, sobrepostos ou cortados pela borda antes de finalizar.\n"
        )

    try:
        prompt = carregar_skill(
            "inventario_visao",
            variaveis={
                "CATEGORIAS_DISPONIVEIS": categorias_str,
                "ANALISE_CENA": bloco_cena,
            },
        )
    except Exception as e:
        _log.error("erro_carregar_skill", extra={"erro": str(e)})
        return {"objetos": []}

    _log.info("analise_completa_iniciada", extra={
        "foto": caminho_foto,
        "categorias_passadas": len(categorias),
        "tamanho_prompt": len(prompt),
    })

    try:
        resposta, modelo = chamar_visao(prompt, caminho_foto)
        dados = extrair_json(resposta)

        if not isinstance(dados, dict):
            _log.warning("resposta_nao_dict", extra={"tipo": type(dados).__name__})
            return {"objetos": []}

        objetos = dados.get("objetos", [])
        if not isinstance(objetos, list):
            objetos = []

        # Validar e limpar cada objeto
        validos = []
        for o in objetos:
            if not isinstance(o, dict):
                continue
            nome = (o.get("nome") or "").strip().lower()
            nome = _normalizar_nome(nome)
            conf = float(o.get("confianca", 0.5) or 0.5)
            if not nome or conf < 0.4:
                continue

            # Sanitizar categoria
            cat = (o.get("categoria_sugerida") or "Outros").strip()
            if cat not in categorias:
                _log.info("categoria_invalida_substituida",
                          extra={"original": cat, "nova": "Outros"})
                cat = "Outros"
            o["categoria_sugerida"] = cat

            # Validar bbox (None se inválida; pipeline usa foto inteira nesse caso)
            o["bbox_normalizada"] = _validar_bbox(o.get("bbox_normalizada"))

            # Garantir campos mínimos
            o["nome"] = nome
            o["confianca"] = conf
            o["quantidade"] = int(o.get("quantidade", 1) or 1)
            o["palavras_chave"] = list(o.get("palavras_chave") or [])
            o["descricao_posicao"] = (o.get("descricao_posicao") or "").strip()

            validos.append(o)

        resultado = {
            "contexto_da_cena": dados.get("contexto_da_cena", ""),
            "fundo": dados.get("fundo", ""),
            "objetos": validos,
        }

        _log.info("analise_completa_concluida", extra={
            "total": len(validos),
            "nomes": [o["nome"] for o in validos],
            "contexto": resultado["contexto_da_cena"],
            "modelo": modelo,
        })
        return resultado

    except Exception as e:
        _log.warning("erro_analise_completa", extra={"erro": str(e)})
        return {"objetos": []}


# ────────────────────────────────────────────────────────────────────────────
# Cache: evita chamar Claude duas vezes para a mesma foto.
# ────────────────────────────────────────────────────────────────────────────

_CACHE_ANALISE: dict[str, dict] = {}


def listar_objetos_globais(caminho_foto: str) -> list[dict]:
    """
    Compat com Agent 1: devolve lista no formato antigo,
        [{"nome": str, "quantidade": int, "descricao_posicao": str}, ...]
    Por baixo, faz análise rica completa e cacheia.
    """
    analise = analisar_foto_completa(caminho_foto)
    _CACHE_ANALISE[caminho_foto] = analise

    return [
        {
            "nome": o["nome"],
            "quantidade": o["quantidade"],
            "descricao_posicao": o.get("descricao_posicao", ""),
        }
        for o in analise.get("objetos", [])
    ]


def obter_analise_completa_cacheada(caminho_foto: str) -> dict | None:
    """Retorna análise rica cacheada (se foto já foi processada nesta sessão)."""
    return _CACHE_ANALISE.get(caminho_foto)


def obter_dados_objeto_por_nome(caminho_foto: str, nome: str) -> dict | None:
    """
    Busca o objeto pelo nome dentro da análise rica cacheada.
    Útil para o pipeline pular o Agent 2.
    """
    analise = _CACHE_ANALISE.get(caminho_foto)
    if not analise:
        return None
    nome_l = (nome or "").strip().lower()
    for o in analise.get("objetos", []):
        if o["nome"] == nome_l:
            return o
    return None


# ────────────────────────────────────────────────────────────────────────────
# Associação: amarra a lista global aos bboxes do OpenCV.
# ────────────────────────────────────────────────────────────────────────────

PROMPT_ASSOCIAR = """
Voce esta vendo uma foto com bboxes verdes numerados (#1, #2, ...).
Ja identifiquei previamente os objetos visiveis nesta foto.

Lista de objetos identificados:
{lista_objetos}

Sua tarefa: para cada bbox numerado, dizer qual objeto da lista ele contem.

Responda APENAS com JSON valido:
{{
  "associacoes": [
    {{"recorte": 1, "nome_da_lista": "chave de fenda phillips", "confianca": 0.9}},
    {{"recorte": 2, "nome_da_lista": "regua", "confianca": 0.85}}
  ]
}}

REGRAS:
- "nome_da_lista" DEVE ser EXATAMENTE um dos nomes acima (copia literal).
- Se um bbox NAO corresponde a nenhum objeto da lista (ruido/fundo), use "nenhum".
- "confianca" entre 0.0 e 1.0.
- Quantidade de bboxes na foto: {n_recortes}
"""


def associar_nomes_a_recortes(
    caminho_foto_original: str,
    caminho_imagem_anotada: str,
    lista_objetos: list[dict],
    n_recortes: int,
) -> list[dict]:
    """
    Recebe foto anotada (com #1, #2, #3 desenhados) e devolve
        [{"recorte_idx": int, "nome": str, "confianca": float}, ...].
    """
    if not lista_objetos:
        return []

    lista_str = "\n".join([
        f"  - {o['nome']}"
        + (f" (qtd: {o['quantidade']})" if o.get("quantidade", 1) > 1 else "")
        + (f" — {o.get('descricao_posicao', '')}" if o.get("descricao_posicao") else "")
        for o in lista_objetos
    ])

    prompt = PROMPT_ASSOCIAR.format(
        lista_objetos=lista_str,
        n_recortes=n_recortes,
    )

    _log.info("associando_nomes", extra={
        "n_recortes": n_recortes,
        "n_objetos_listados": len(lista_objetos),
    })

    try:
        resposta, modelo = chamar_visao(prompt, caminho_imagem_anotada)
        dados = extrair_json(resposta)
        if not isinstance(dados, dict):
            return []
        associacoes = dados.get("associacoes", [])

        nomes_validos = {o["nome"] for o in lista_objetos}
        resultado = []
        for a in associacoes:
            if not isinstance(a, dict):
                continue
            recorte_idx = int(a.get("recorte", 0))
            nome = (a.get("nome_da_lista") or "").strip().lower()
            conf = float(a.get("confianca", 0.5) or 0.5)

            if recorte_idx < 1 or recorte_idx > n_recortes:
                continue
            if nome == "nenhum":
                continue
            if nome not in nomes_validos:
                _log.warning("nome_associado_invalido",
                             extra={"nome": nome, "validos": list(nomes_validos)})
                continue

            resultado.append({
                "recorte_idx": recorte_idx,
                "nome": nome,
                "confianca": conf,
            })

        _log.info("associacao_concluida",
                  extra={"associacoes": len(resultado), "modelo": modelo})
        return resultado
    except Exception as e:
        _log.warning("erro_associacao", extra={"erro": str(e)})
        return []
