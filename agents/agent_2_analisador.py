"""
Agente 2: Analisador Visual

Roteador: Ollama por padrão.
Se confiança retornada < LIMIAR_CONFIANCA, o pipeline solicita nova análise via Claude.
Isso é tratado AQUI: após análise local, se confiança baixa e Claude disponível,
faz segunda chamada automática e retorna o melhor resultado.
"""
import logging
from core.llm import chamar_visao, extrair_json
from core.roteador import deve_usar_claude_visao
from core.config import LIMIAR_CONFIANCA

_log = logging.getLogger("casaiq.agent_2")

PROMPT_ANALISE = """
Analise o objeto "{nome}" nesta imagem.

REGRAS CRITICAS PARA O CAMPO "nome":
- Use o nome GENERICO MAIS SIMPLES. Exemplo: "chave de fenda" (NAO "chave de fenda phillips com cabo vermelho").
- NAO inclua cor, material ou tamanho no nome. Esses sao campos separados.
- So especifique o tipo (phillips/fenda chata/etc) se VISIVELMENTE confirmado na imagem (ex: ponta em X visivel para phillips).
- Em duvida sobre o tipo, use o termo generico ("chave de fenda" sem subtipo).
- NUNCA chame uma chave de fenda chata de "phillips".

REGRAS CRITICAS PARA "confianca":
- 0.9+ apenas se voce ve o objeto claramente e tem certeza do que e.
- 0.6-0.8 se tem ideia mas alguns detalhes sao incertos.
- < 0.5 se a imagem e ambigua ou esta parcialmente obstruida.

Responda APENAS com JSON valido (sem texto fora do JSON):
{{
  "nome": "chave de fenda",
  "descricao": "1-2 frases descritivas",
  "cor": "cor(es) principais",
  "tamanho": "pequeno",
  "tamanho_estimado_cm": "LxAxP estimado",
  "peso_estimado_g": null,
  "material": "material principal",
  "estado": "bom",
  "funcao": "para que serve em uma frase",
  "palavras_chave": ["palavra1", "palavra2", "palavra3"],
  "confianca": 0.85
}}
Valores de "tamanho": pequeno | medio | grande
Valores de "estado": novo | bom | regular | ruim
"""


def analisar_objeto(recorte_path: str, nome_sugerido: str) -> dict:
    """
    Analisa objeto. Se confiança < LIMIAR e Claude disponível,
    solicita segunda opinião automaticamente via roteador.
    Retorna o resultado de maior confiança.
    """
    _log.info("analisando", extra={"nome_sugerido": nome_sugerido})
    prompt = PROMPT_ANALISE.format(nome=nome_sugerido)
    resultados = []

    # Primeira análise (pode ser local ou Claude, segundo roteador)
    try:
        resposta, modelo = chamar_visao(prompt, recorte_path)
        dados = extrair_json(resposta)
        if isinstance(dados.get("palavras_chave"), str):
            dados["palavras_chave"] = dados["palavras_chave"].split(",")
        dados["_modelo"] = modelo
        resultados.append(dados)
        _log.info("analise_primaria", extra={
            "nome": dados.get("nome", nome_sugerido),
            "confianca": dados.get("confianca", 0),
            "modelo": modelo,
        })
    except Exception as e:
        _log.warning("erro_analise_primaria", extra={"erro": str(e)})

    # Segunda opinião automática: se confiança baixa, pede segunda via Claude
    confianca_atual = resultados[0].get("confianca", 0.0) if resultados else 0.0
    if confianca_atual < LIMIAR_CONFIANCA and deve_usar_claude_visao(confianca_anterior=confianca_atual):
        _log.info("segunda_opiniao_solicitada", extra={
            "confianca": confianca_atual, "limiar": LIMIAR_CONFIANCA,
        })
        try:
            resposta2, modelo2 = chamar_visao(prompt, recorte_path, confianca_anterior=confianca_atual)
            dados2 = extrair_json(resposta2)
            if isinstance(dados2.get("palavras_chave"), str):
                dados2["palavras_chave"] = dados2["palavras_chave"].split(",")
            dados2["_modelo"] = modelo2
            resultados.append(dados2)
            _log.info("segunda_opiniao_recebida", extra={
                "confianca": dados2.get("confianca", 0), "modelo": modelo2,
            })
        except Exception as e:
            _log.warning("erro_segunda_opiniao", extra={"erro": str(e)})

    if not resultados:
        return {"nome": nome_sugerido, "descricao": "", "cor": "", "tamanho": "",
                "tamanho_estimado_cm": "", "peso_estimado_g": None, "material": "",
                "estado": "bom", "funcao": "", "palavras_chave": [],
                "confianca": 0.0, "_modelo": "erro"}

    # Retorna resultado de maior confiança
    return max(resultados, key=lambda d: d.get("confianca", 0.0))
