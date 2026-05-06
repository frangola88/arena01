"""
Agente 3: Enriquecedor de Metadados

Sempre usa modelo local (llama3.2) — tarefa simples de classificação.
O roteador em deve_usar_claude_texto(CLASSIFICACAO/ENRIQUECIMENTO) retorna False.
"""
from core.llm import chamar_texto, extrair_json
from core.roteador import TarefaTexto

PROMPT_ENRIQUECIMENTO = """
Objeto de inventario domestico:
  Nome: {nome}
  Descricao: {descricao}
  Funcao: {funcao}
  Palavras-chave: {palavras_chave}
  Categorias disponiveis: {categorias}

Responda APENAS com JSON valido:
{{
  "categoria": "nome exato de uma das categorias acima",
  "palavras_chave_extras": ["sinonimo1", "uso1", "uso2"],
  "nome_normalizado": "nome em minusculas sem acentos desnecessarios"
}}
"""


def _formatar_palavras_chave(tokens: list[str]) -> str:
    """
    Formato: |token1|token2|token3|
    Separador | (nunca vírgula, que pode aparecer em nomes).
    Busca SQL: WHERE palavras_chave LIKE '%|token|%'
    """
    limpos = []
    for item in tokens:
        for t in str(item).lower().strip().split():
            t = t.strip("|, ")
            if t and len(t) > 1:
                limpos.append(t)
    if not limpos:
        return ""
    return "|" + "|".join(dict.fromkeys(limpos)) + "|"


def enriquecer_objeto(analise: dict, categorias_disponiveis: list[str]) -> dict:
    """
    Enriquece metadados. categorias_disponiveis DEVE ser passado pelo pipeline
    (buscado do banco antes de entrar no loop de objetos).
    """
    print(f"[Agente 3] Enriquecendo: '{analise.get('nome', '')}'")
    resultado = analise.copy()
    try:
        prompt = PROMPT_ENRIQUECIMENTO.format(
            nome=analise.get("nome", ""),
            descricao=analise.get("descricao", ""),
            funcao=analise.get("funcao", ""),
            palavras_chave=analise.get("palavras_chave", []),
            categorias=", ".join(categorias_disponiveis) if categorias_disponiveis else "Outros",
        )
        resposta, _ = chamar_texto(prompt, tarefa=TarefaTexto.ENRIQUECIMENTO)
        dados = extrair_json(resposta)
        resultado["categoria_nome"] = dados.get("categoria", "Outros")
        resultado["nome"] = dados.get("nome_normalizado") or analise.get("nome", "")
        todas = list(analise.get("palavras_chave", []))
        todas += dados.get("palavras_chave_extras", [])
        todas.append(resultado["nome"])
        resultado["palavras_chave"] = _formatar_palavras_chave(todas)
    except Exception as e:
        print(f"[Agente 3] ERRO: {e}. Usando 'Outros'.")
        resultado["categoria_nome"] = "Outros"
        resultado["palavras_chave"] = _formatar_palavras_chave(
            list(analise.get("palavras_chave", [])) + [analise.get("nome", "")]
        )
    return resultado
