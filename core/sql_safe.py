"""Validação e execução segura de SQL gerado por LLM.

Defesa em camadas (cada camada é redundante; mesmo se uma falhar as outras seguram):

  1. validar_select(sql)   — whitelist + regex bloqueia DDL/DML, comentários,
                             multi-statement, tokens perigosos (ATTACH, PRAGMA…).
  2. garantir_limit(sql)   — injeta LIMIT no fim se ausente.
  3. conectar_readonly()   — abre SQLite em mode=ro via URI; o próprio engine
                             recusa qualquer mutação, mesmo que a camada 1 falhe.

Camada extra do stdlib: sqlite3.Connection.execute() já recusa múltiplos
statements ('You can only execute one statement at a time').
"""
import re
import sqlite3
from pathlib import Path
from core.config import DB_PATH


MAX_PERGUNTA_CHARS = 1000
MAX_SQL_CHARS      = 2000
MAX_LINHAS         = 100


class SQLInseguro(ValueError):
    """SQL gerado por LLM falhou na validação."""


# Palavras-chave que jamais devem aparecer num SELECT de leitura.
_TOKENS_PROIBIDOS = frozenset({
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE",
    "REPLACE", "ATTACH", "DETACH", "PRAGMA", "VACUUM", "REINDEX",
    # Funções de extensão / FS
    "LOAD_EXTENSION",
})


def validar_select(sql: str) -> str:
    """Valida SQL gerado por LLM. Retorna SQL normalizado pronto pra execução.

    Levanta SQLInseguro se detectar qualquer vetor de risco.
    """
    if not sql or not sql.strip():
        raise SQLInseguro("SQL vazio")
    if len(sql) > MAX_SQL_CHARS:
        raise SQLInseguro(f"SQL excede {MAX_SQL_CHARS} chars")

    # Tira ; trailing (legítimo); ; no meio é ataque (multi-statement).
    s = sql.strip().rstrip(";").strip()
    if ";" in s:
        raise SQLInseguro("Múltiplos statements proibidos")

    # Comentários SQL escondem payload malicioso atrás de inocência aparente.
    if "--" in s or "/*" in s or "*/" in s:
        raise SQLInseguro("Comentários SQL proibidos")

    head = s.upper().lstrip()
    if not (head.startswith("SELECT") or head.startswith("WITH")):
        raise SQLInseguro("SQL deve começar com SELECT ou WITH")

    # Whitelist por palavra inteira: rejeita qualquer token de mutação/DDL.
    palavras = re.findall(r"\b[A-Za-z_][A-Za-z_0-9]*\b", s.upper())
    for p in palavras:
        if p in _TOKENS_PROIBIDOS:
            raise SQLInseguro(f"Token proibido: {p}")

    return s


def garantir_limit(sql: str, maximo: int = MAX_LINHAS) -> str:
    """Garante que o SQL termine com LIMIT N. Heurística simples:
    se há `LIMIT N` no fim do statement, respeita (cap em `maximo`);
    senão, anexa.

    Limitação conhecida: LIMIT em subquery interna não é detectado como
    sendo de subquery e pode pular a injeção do LIMIT externo. Aceitável
    porque a conexão read-only já impede dano e o cap é cortesia de UX.
    """
    s = sql.rstrip().rstrip(";").rstrip()
    m = re.search(r"\blimit\s+(\d+)\s*$", s, re.IGNORECASE)
    if m:
        n = int(m.group(1))
        if n > maximo:
            s = re.sub(r"\blimit\s+\d+\s*$",
                       f"LIMIT {maximo}", s, count=1, flags=re.IGNORECASE)
        return s
    return f"{s} LIMIT {maximo}"


def conectar_readonly(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Conexão SQLite em modo read-only via URI. O engine recusa qualquer escrita.

    Usar essa conexão para executar SQL gerado por LLM — nunca a get_db() comum.
    """
    caminho = Path(db_path) if db_path is not None else DB_PATH
    uri = f"file:{caminho}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


# Whitelist do que o LLM pode "ver" do schema. Esconde:
#   historico_chat (privacidade do usuário)
#   fotos_processadas / videos_processados (estado interno de pipeline)
#   sqlite_master e companhia (introspecção do engine)
_TABELAS_EXPOSTAS_AO_LLM = ("objetos", "localizacoes", "categorias")


def schema_para_prompt(conn: sqlite3.Connection) -> str:
    """Introspecta o schema das tabelas user-facing e retorna formato compacto
    pronto pra injetar no prompt de text-to-SQL.

    O LLM passa a "ver" o schema vivo do banco, não uma cópia hardcoded que
    apodrece a cada migração. Apenas tabelas da whitelist são expostas.
    """
    linhas = []
    for tabela in _TABELAS_EXPOSTAS_AO_LLM:
        cols = conn.execute(f"PRAGMA table_info({tabela})").fetchall()
        if not cols:
            continue
        nomes = [c[1] for c in cols]   # row[1] = name
        linhas.append(f"  {tabela}({', '.join(nomes)})")
    return "\n".join(linhas)
