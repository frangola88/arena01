"""
Agente 2: Analisador Visual

Roteador: Ollama por padrão.
Se confiança retornada < LIMIAR_CONFIANCA, o pipeline solicita nova análise via Claude.
Isso é tratado AQUI: após análise local, se confiança baixa e Claude disponível,
faz segunda chamada automática e retorna o melhor resultado.
"""
from core.llm import chamar_visao, extrair_json
from core.roteador import deve_usar_claude_visao
from core.config import LIMIAR_CONFIANCA

PROMPT_ANALISE = """
Analise o objeto "{nome}" nesta imagem.
Responda APENAS com JSON valido (sem texto fora do JSON):
{{
  "nome": "nome especifico do objeto",
  "descricao": "1-2 frases",
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
"confianca": 0.0 (incerto) a 1.0 (certeza)
"""


def analisar_objeto(recorte_path: str, nome_sugerido: str) -> dict:
    """
    Analisa objeto. Se confiança < LIMIAR e Claude disponível,
    solicita segunda opinião automaticamente via roteador.
    Retorna o resultado de maior confiança.
    """
    print(f"[Agente 2] Analisando: '{nome_sugerido}'")
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
        print(f"[Agente 2] '{dados.get('nome', nome_sugerido)}' conf={dados.get('confianca',0):.2f} via {modelo}")
    except Exception as e:
        print(f"[Agente 2] ERRO primeira análise: {e}")

    # Segunda opinião automática: se confiança baixa, pede segunda via Claude
    confianca_atual = resultados[0].get("confianca", 0.0) if resultados else 0.0
    if confianca_atual < LIMIAR_CONFIANCA and deve_usar_claude_visao(confianca_anterior=confianca_atual):
        print(f"[Agente 2] Confiança {confianca_atual:.2f} < {LIMIAR_CONFIANCA}. Segunda opinião via Claude.")
        try:
            resposta2, modelo2 = chamar_visao(prompt, recorte_path, confianca_anterior=confianca_atual)
            dados2 = extrair_json(resposta2)
            if isinstance(dados2.get("palavras_chave"), str):
                dados2["palavras_chave"] = dados2["palavras_chave"].split(",")
            dados2["_modelo"] = modelo2
            resultados.append(dados2)
            print(f"[Agente 2] Segunda opinião: conf={dados2.get('confianca',0):.2f} via {modelo2}")
        except Exception as e:
            print(f"[Agente 2] Segunda opinião falhou: {e}")

    if not resultados:
        return {"nome": nome_sugerido, "descricao": "", "cor": "", "tamanho": "",
                "tamanho_estimado_cm": "", "peso_estimado_g": None, "material": "",
                "estado": "bom", "funcao": "", "palavras_chave": [],
                "confianca": 0.0, "_modelo": "erro"}

    # Retorna resultado de maior confiança
    return max(resultados, key=lambda d: d.get("confianca", 0.0))
