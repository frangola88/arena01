"""
Agente Assistente: Chat com o inventário.

Roteamento:
  text_to_sql    -> Claude preferido (muito mais preciso para SQL complexo)
  resposta_chat  -> Claude preferido (guias, sugestões, multi-objeto)

Se Claude indisponível, Ollama assume. Se ambos falharem, retorna mensagem de erro.
"""
import json
import re
from core.llm import chamar_texto
from core.roteador import TarefaTexto
from core.sql_safe import (
    MAX_PERGUNTA_CHARS, SQLInseguro,
    validar_select, garantir_limit, conectar_readonly, schema_para_prompt,
)

PROMPT_SQL = """
Converta a pergunta em SQL SELECT para o banco de inventario domestico.

Schema:
{schema}

Joins uteis:
  JOIN localizacoes l ON objetos.localizacao_id = l.id
  JOIN categorias   c ON objetos.categoria_id   = c.id

Busca por palavras_chave: WHERE palavras_chave LIKE '%|token|%'
Exemplo: WHERE objetos.palavras_chave LIKE '%|alicate|%'

Pergunta: {pergunta}

Retorne APENAS o SQL SELECT. Sem markdown. Sem explicacoes. Limite 50 resultados.
"""

PROMPT_RESPOSTA = """
Usuario perguntou (portugues): "{pergunta}"
Resultados do banco (JSON): {resultados}

Responda em portugues brasileiro de forma util e natural.
- Liste os objetos encontrados com suas localizacoes
- Para perguntas "como fazer" (ex: trocar chuveiro): monte um guia pratico
  listando o que o usuario JA TEM e o que precisa COMPRAR
- Se nao encontrou nada: sugira termos de busca alternativos
- Seja conciso e direto
"""


def chat(pergunta: str, db_conn) -> dict:
    """
    Retorna {"resposta": str, "sql": str, "resultados": list, "modelo": str}.
    db_conn = conexao SQLite da thread atual (usada SÓ para gravar histórico).

    O SQL gerado pelo LLM é executado numa conexão read-only separada —
    o engine SQLite recusa qualquer mutação mesmo se a validação textual falhar.
    """
    sql_gerado, resultados, modelo_usado = "", [], "ollama"

    if len(pergunta or "") > MAX_PERGUNTA_CHARS:
        return {
            "resposta": f"Pergunta muito longa (máx {MAX_PERGUNTA_CHARS} caracteres).",
            "sql": "", "resultados": [], "modelo": "rejeitado",
        }

    try:
        ro = conectar_readonly()
        try:
            schema = schema_para_prompt(ro)
            prompt_sql = PROMPT_SQL.format(schema=schema, pergunta=pergunta)
            sql_bruto, modelo_usado = chamar_texto(prompt_sql, tarefa=TarefaTexto.TEXT_TO_SQL)
            linhas_sql = [l for l in sql_bruto.splitlines()
                          if l.strip().lower() not in ("sql", "python", "")]
            sql_candidato = "\n".join(linhas_sql).strip()
            sql_gerado = garantir_limit(validar_select(sql_candidato))
            rows = ro.execute(sql_gerado).fetchall()
            resultados = [dict(r) for r in rows]
        finally:
            ro.close()
    except SQLInseguro as e:
        print(f"[Assistente] SQL bloqueado pelo validador: {e}")
        sql_gerado, resultados = "", []
    except Exception as e:
        print(f"[Assistente] ERRO SQL: {e}")
        sql_gerado, resultados = "", []

    try:
        resultados_str = json.dumps(resultados[:20], ensure_ascii=False)
        prompt_resp = PROMPT_RESPOSTA.format(pergunta=pergunta, resultados=resultados_str)
        resposta, modelo_usado = chamar_texto(prompt_resp, tarefa=TarefaTexto.RESPOSTA_CHAT,
                                              max_tokens=1024)
    except Exception as e:
        print(f"[Assistente] ERRO resposta: {e}")
        resposta = f"Encontrei {len(resultados)} resultado(s). Erro ao formular resposta."

    try:
        db_conn.execute(
            "INSERT INTO historico_chat (pergunta, resposta, modelo) VALUES (?,?,?)",
            (pergunta, resposta, modelo_usado)
        )
        db_conn.commit()
    except Exception as e:
        print(f"[Assistente] ERRO histórico: {e}")

    return {"resposta": resposta, "sql": sql_gerado, "resultados": resultados, "modelo": modelo_usado}
