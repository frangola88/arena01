"""
Agente 1: Segmentador de Objetos

Estratégia em 2 etapas:
  A) Lista objetos visíveis (sem bbox — mais confiável com qualquer VLM)
  B) Para cada objeto, tenta bbox opcional; fallback = foto inteira

Roteador: visão via Ollama por padrão.
Claude é acionado se Ollama retornar lista vazia ou confiança média < LIMIAR.
"""
from PIL import Image
from core.config import RECORTES_DIR
from core.llm import chamar_visao, extrair_json
from core.roteador import TarefaTexto

PROMPT_LISTA = """
Analise a foto e liste TODOS os objetos visíveis, um por linha.
Maximo 15 objetos. Seja especifico: "alicate de bico" nao apenas "alicate".
Responda APENAS com JSON valido:
{"objetos": [{"nome": "alicate de bico"}, {"nome": "fita isolante preta"}]}
Se a foto estiver vazia: {"objetos": []}
"""

PROMPT_BBOX = """
Na foto, localize o objeto: "{nome}"
Coordenadas como fracao 0.0-1.0 (x1,y1=superior-esquerdo; x2,y2=inferior-direito).
Responda APENAS com JSON:
{{"x1": 0.1, "y1": 0.2, "x2": 0.4, "y2": 0.6}}
Se nao conseguir localizar com confianca: {{"erro": "nao_localizado"}}
"""


def _recortar(imagem: Image.Image, bbox: dict, foto_id: int, idx: int) -> str:
    w, h = imagem.size
    left  = max(0, int(bbox["x1"] * w))
    upper = max(0, int(bbox["y1"] * h))
    right = min(w, max(int(bbox["x2"] * w), left + 20))
    lower = min(h, max(int(bbox["y2"] * h), upper + 20))
    caminho = str(RECORTES_DIR / f"foto_{foto_id}_obj_{idx:02d}.jpg")
    imagem.crop((left, upper, right, lower)).save(caminho, "JPEG", quality=90)
    return caminho


def segmentar_foto(caminho_foto: str, foto_id: int) -> list[dict]:
    """
    Retorna lista de {"nome": str, "recorte_path": str}.
    Primeiro tenta Ollama; se lista vazia, aciona Claude (via roteador).
    """
    print(f"[Agente 1] Segmentando: {caminho_foto}")
    imagem = Image.open(caminho_foto).convert("RGB")

    # Etapa A: lista (tenta local primeiro, Claude se lista vier vazia)
    nomes = []
    for tentativa, conf_anterior in enumerate([None, 0.0]):
        try:
            resposta, modelo = chamar_visao(PROMPT_LISTA, caminho_foto,
                                            confianca_anterior=conf_anterior)
            dados = extrair_json(resposta)
            nomes = [o["nome"] for o in dados.get("objetos", []) if o.get("nome")]
            print(f"[Agente 1] {len(nomes)} objeto(s) via {modelo} (tentativa {tentativa+1})")
            if nomes:
                break
        except Exception as e:
            print(f"[Agente 1] ERRO tentativa {tentativa+1}: {e}")
            if tentativa == 1:
                return [{"nome": "objeto desconhecido", "recorte_path": caminho_foto}]

    if not nomes:
        return []

    # Etapa B: bbox opcional
    resultado = []
    for i, nome in enumerate(nomes):
        recorte_path = caminho_foto
        try:
            resp_bbox, _ = chamar_visao(PROMPT_BBOX.format(nome=nome), caminho_foto)
            bbox = extrair_json(resp_bbox)
            if ("erro" not in bbox
                    and all(k in bbox for k in ["x1", "y1", "x2", "y2"])
                    and bbox["x2"] > bbox["x1"] and bbox["y2"] > bbox["y1"]):
                recorte_path = _recortar(imagem, bbox, foto_id, i)
        except Exception:
            pass
        resultado.append({"nome": nome, "recorte_path": recorte_path})

    return resultado
