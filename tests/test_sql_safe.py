"""Testes do módulo core.sql_safe — defesa em camadas para SQL gerado por LLM.

3 grupos:
  1. validar_select(): aceita queries de leitura, rejeita vetores de ataque.
  2. garantir_limit(): cap de linhas funciona em casos triviais.
  3. conectar_readonly(): SQLite recusa escrita mesmo com SQL malicioso.

A camada (3) é a defesa final — é o que importa se a validação textual (1)
for contornada por algum payload exótico.
"""
from __future__ import annotations

import sqlite3
import pytest

from core.sql_safe import (
    SQLInseguro, validar_select, garantir_limit, conectar_readonly,
    schema_para_prompt, MAX_SQL_CHARS, MAX_LINHAS,
)


# ---------------------------------------------------------------------------
# validar_select — caminho feliz
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sql", [
    "SELECT * FROM objetos",
    "select id, nome from objetos where confianca > 0.5",
    "SELECT o.nome, l.nome FROM objetos o JOIN localizacoes l ON o.localizacao_id=l.id",
    "WITH base AS (SELECT * FROM objetos) SELECT nome FROM base",
    "SELECT COUNT(*) FROM objetos GROUP BY categoria_id",
    "SELECT * FROM objetos WHERE palavras_chave LIKE '%|alicate|%'",
    "  SELECT 1  ",                               # whitespace tolerado
    "SELECT * FROM objetos;",                     # ; trailing OK
])
def test_validar_aceita_select_legitimo(sql):
    assert validar_select(sql)


# ---------------------------------------------------------------------------
# validar_select — vetores de ataque
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sql,motivo", [
    ("",                                         "vazio"),
    ("   \n  ",                                  "só espaço"),
    ("DROP TABLE objetos",                       "DDL direto"),
    ("DELETE FROM objetos",                      "DML direto"),
    ("UPDATE objetos SET nome='x'",              "UPDATE direto"),
    ("INSERT INTO objetos (nome) VALUES ('x')",  "INSERT direto"),
    ("ALTER TABLE objetos ADD COLUMN x TEXT",    "ALTER"),
    ("CREATE TABLE x (id INT)",                  "CREATE"),
    ("TRUNCATE TABLE objetos",                   "TRUNCATE"),
    ("REPLACE INTO objetos VALUES (1)",          "REPLACE"),
    ("PRAGMA writable_schema=1",                 "PRAGMA mutativo"),
    ("ATTACH DATABASE '/etc/passwd' AS x",       "ATTACH"),
    ("DETACH DATABASE main",                     "DETACH"),
    ("VACUUM",                                   "VACUUM"),
    ("REINDEX",                                  "REINDEX"),
    ("SELECT load_extension('/tmp/evil.so')",    "LOAD_EXTENSION"),
    ("SELECT 1; DROP TABLE objetos",             "multi-statement por ;"),
    ("SELECT * FROM objetos -- ' OR DROP",       "comentário inline --"),
    ("SELECT * FROM objetos /* DELETE */",       "comentário /* */"),
    ("EXPLAIN QUERY PLAN SELECT * FROM x",       "não começa com SELECT/WITH"),
    ("WITH x AS (DELETE FROM objetos RETURNING *) SELECT * FROM x",
                                                  "CTE com DELETE"),
])
def test_validar_bloqueia_ataque(sql, motivo):
    with pytest.raises(SQLInseguro):
        validar_select(sql)


def test_validar_bloqueia_sql_excessivamente_longo():
    longo = "SELECT " + "x," * (MAX_SQL_CHARS // 2) + " 1"
    with pytest.raises(SQLInseguro, match="excede"):
        validar_select(longo)


def test_validar_normaliza_e_remove_ponto_e_virgula_trailing():
    assert validar_select("SELECT 1;") == "SELECT 1"
    assert validar_select("  SELECT 1  ;  ") == "SELECT 1"


# ---------------------------------------------------------------------------
# garantir_limit
# ---------------------------------------------------------------------------

def test_limit_adicionado_quando_ausente():
    out = garantir_limit("SELECT * FROM objetos", maximo=50)
    assert out.upper().endswith("LIMIT 50")


def test_limit_existente_abaixo_do_cap_e_preservado():
    out = garantir_limit("SELECT * FROM objetos LIMIT 10", maximo=50)
    assert out == "SELECT * FROM objetos LIMIT 10"


def test_limit_excessivo_e_reduzido():
    out = garantir_limit("SELECT * FROM objetos LIMIT 9999", maximo=50)
    assert "LIMIT 50" in out.upper()
    assert "9999" not in out


def test_limit_case_insensitive_e_normalizado_para_maiusculo():
    out = garantir_limit("select * from x limit 5", maximo=50)
    assert out.upper().count("LIMIT") == 1


def test_limit_padrao_usa_max_linhas_global():
    out = garantir_limit("SELECT 1")
    assert f"LIMIT {MAX_LINHAS}" in out.upper()


# ---------------------------------------------------------------------------
# conectar_readonly — defesa final, com DB real
# ---------------------------------------------------------------------------

def test_readonly_permite_select(db_temp):
    ro = conectar_readonly(db_temp)
    try:
        rows = ro.execute("SELECT COUNT(*) AS c FROM categorias").fetchall()
        assert rows[0]["c"] > 0   # seed populou categorias
    finally:
        ro.close()


def test_readonly_recusa_insert_mesmo_com_sql_malformado(db_temp):
    """Defesa final: mesmo se o validador textual falhar, o engine recusa."""
    ro = conectar_readonly(db_temp)
    try:
        with pytest.raises(sqlite3.OperationalError, match=r"readonly"):
            ro.execute("INSERT INTO categorias (nome) VALUES ('hack')")
    finally:
        ro.close()


def test_readonly_recusa_drop_table(db_temp):
    ro = conectar_readonly(db_temp)
    try:
        with pytest.raises(sqlite3.OperationalError, match=r"readonly|read.?only"):
            ro.execute("DROP TABLE categorias")
    finally:
        ro.close()


# ---------------------------------------------------------------------------
# schema_para_prompt — introspecção dinâmica do schema
# ---------------------------------------------------------------------------

def test_schema_para_prompt_lista_tabelas_user_facing(db_temp):
    ro = conectar_readonly(db_temp)
    try:
        s = schema_para_prompt(ro)
    finally:
        ro.close()
    assert "objetos(" in s
    assert "localizacoes(" in s
    assert "categorias(" in s


def test_schema_para_prompt_e_dinamico_inclui_colunas_que_o_hardcoded_omitia(db_temp):
    """Antes do dinâmico, o prompt tinha schema hardcoded sem colunas como
    foto_original_path, recorte_path, icone_fonte, modelo_visao, etc.
    Agora o LLM vê o schema vivo."""
    ro = conectar_readonly(db_temp)
    try:
        s = schema_para_prompt(ro)
    finally:
        ro.close()
    for col in ("foto_original_path", "recorte_path", "icone_fonte",
                "modelo_visao", "revisado_pelo_usuario"):
        assert col in s, f"coluna '{col}' deveria estar no schema dinâmico"


def test_schema_para_prompt_nao_expoe_historico_chat(db_temp):
    """Privacidade: histórico de perguntas/respostas NÃO vai pro contexto do LLM."""
    ro = conectar_readonly(db_temp)
    try:
        s = schema_para_prompt(ro)
    finally:
        ro.close()
    assert "historico_chat" not in s


def test_schema_para_prompt_nao_expoe_estado_interno_de_pipeline(db_temp):
    """Tabelas de estado interno (fotos/videos processados, sqlite_master) ficam ocultas."""
    ro = conectar_readonly(db_temp)
    try:
        s = schema_para_prompt(ro)
    finally:
        ro.close()
    assert "fotos_processadas" not in s
    assert "videos_processados" not in s
    assert "sqlite_master" not in s


# ---------------------------------------------------------------------------
# ATTACH read-only
# ---------------------------------------------------------------------------

def test_attach_e_responsabilidade_do_validador_textual(db_temp, tmp_path):
    """Documenta limite da camada read-only: ATTACH NÃO é bloqueado pelo
    engine (o DB anexado pode até ser gravado). A única defesa é o validador
    textual em validar_select() — já testado em test_validar_bloqueia_ataque.
    """
    outro = tmp_path / "outro.db"
    sqlite3.connect(outro).execute("CREATE TABLE t(x)")  # cria DB anexável

    ro = conectar_readonly(db_temp)
    try:
        # Confirma o comportamento do SQLite (engine permite ATTACH em RO):
        ro.execute(f"ATTACH DATABASE '{outro}' AS x")
        # E confirma que o validador textual rejeita ANTES de chegar aqui:
        with pytest.raises(SQLInseguro):
            validar_select(f"ATTACH DATABASE '{outro}' AS x")
    finally:
        ro.close()
