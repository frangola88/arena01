"""Testes de agents.assistente.chat — integra core.sql_safe ao fluxo do /chat.

Estratégia: mocka chamar_texto (LLM) em duas chamadas (text_to_sql + resposta).
DB SQLite isolado via db_temp; chamar_texto não toca rede; conexão read-only
usada pelo assistente é a mesma DB_PATH (apontado pelo db_temp).

O foco aqui é o handshake entre assistente e camada de segurança:
  - SQL malicioso gerado pelo LLM cai no validador → resposta sem dano
  - SQL legítimo flui pra read-only e retorna resultados
  - LIMIT é injetado mesmo se o LLM esquecer
  - Histórico é sempre gravado (auditoria)
  - Pergunta longa demais é rejeitada antes de chegar no LLM
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_objetos(db_temp, qtd: int = 3) -> int:
    """Insere `qtd` linhas em objetos via conexão de escrita normal. Devolve
    localizacao_id usada."""
    from core.database import get_db
    conn = get_db()
    try:
        cur = conn.execute(
            "INSERT INTO localizacoes (nome, tipo) VALUES (?, ?)",
            ("Cozinha", "armário"),
        )
        loc_id = cur.lastrowid
        for i in range(qtd):
            conn.execute(
                "INSERT INTO objetos (nome, localizacao_id) VALUES (?, ?)",
                (f"obj_{i}", loc_id),
            )
        conn.commit()
        return loc_id
    finally:
        conn.close()


@pytest.fixture
def mock_llm(mocker):
    """Mocka chamar_texto no namespace de agents.assistente. Cada teste
    customiza side_effect (uma resposta por chamada: 1ª=SQL, 2ª=resposta)."""
    return mocker.patch("agents.assistente.chamar_texto")


# ---------------------------------------------------------------------------
# Caminho feliz: SELECT legítimo flui pra read-only e retorna resultados
# ---------------------------------------------------------------------------

def test_sql_valido_executa_em_readonly_e_retorna_resultados(db_temp, mock_llm):
    _seed_objetos(db_temp, qtd=3)
    mock_llm.side_effect = [
        ("SELECT nome FROM objetos ORDER BY id", "ollama"),
        ("Encontrei 3 objetos.", "ollama"),
    ]

    from core.database import get_db
    from agents.assistente import chat
    conn = get_db()
    try:
        resp = chat("quais objetos eu tenho?", conn)
    finally:
        conn.close()

    assert resp["modelo"] == "ollama"
    assert "SELECT" in resp["sql"].upper()
    assert "LIMIT" in resp["sql"].upper()  # garantir_limit injetou
    assert len(resp["resultados"]) == 3
    assert resp["resposta"] == "Encontrei 3 objetos."


# ---------------------------------------------------------------------------
# SQL malicioso do LLM é bloqueado pelo validador
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sql_malicioso", [
    "DROP TABLE objetos",
    "DELETE FROM objetos",
    "INSERT INTO objetos (nome) VALUES ('hack')",
    "ATTACH DATABASE '/tmp/evil.db' AS x",
    "PRAGMA writable_schema=1",
    "SELECT 1; DROP TABLE objetos",
    "SELECT * FROM objetos -- injection here",
    "SELECT load_extension('/tmp/evil.so')",
])
def test_sql_malicioso_do_llm_e_bloqueado(db_temp, mock_llm, sql_malicioso):
    """LLM 'inventa' SQL perigoso → validador rejeita, resposta segue sem
    resultados, NENHUMA mutação no DB."""
    _seed_objetos(db_temp, qtd=2)
    mock_llm.side_effect = [
        (sql_malicioso, "ollama"),
        ("Não consegui processar a busca.", "ollama"),
    ]

    from core.database import get_db
    from agents.assistente import chat
    conn = get_db()
    try:
        resp = chat("pergunta inocente", conn)

        # SQL bloqueado: campos devem refletir falha controlada
        assert resp["sql"] == ""
        assert resp["resultados"] == []
        # Mas a resposta foi entregue (LLM gerou texto user-facing)
        assert resp["resposta"]

        # DB intacto: 2 objetos seguem lá
        n = conn.execute("SELECT COUNT(*) AS c FROM objetos").fetchone()
        assert n["c"] == 2
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Pergunta longa é rejeitada antes de chegar ao LLM
# ---------------------------------------------------------------------------

def test_pergunta_excessivamente_longa_e_rejeitada(db_temp, mock_llm):
    """Defesa de custo/abuso: pergunta acima do limite não chega no LLM."""
    from agents.assistente import chat
    from core.database import get_db
    pergunta_longa = "x" * 5000

    conn = get_db()
    try:
        resp = chat(pergunta_longa, conn)
    finally:
        conn.close()

    assert resp["modelo"] == "rejeitado"
    assert resp["sql"] == ""
    assert resp["resultados"] == []
    assert "longa" in resp["resposta"].lower()
    mock_llm.assert_not_called()


# ---------------------------------------------------------------------------
# LIMIT é injetado quando o LLM esquece
# ---------------------------------------------------------------------------

def test_limit_e_injetado_quando_llm_esquece(db_temp, mock_llm):
    _seed_objetos(db_temp, qtd=2)
    mock_llm.side_effect = [
        ("SELECT * FROM objetos", "ollama"),       # sem LIMIT
        ("ok", "ollama"),
    ]

    from core.database import get_db
    from agents.assistente import chat
    conn = get_db()
    try:
        resp = chat("liste tudo", conn)
    finally:
        conn.close()

    assert "LIMIT" in resp["sql"].upper()


# ---------------------------------------------------------------------------
# Histórico gravado em ambos os caminhos (sucesso e SQL bloqueado)
# ---------------------------------------------------------------------------

def test_historico_gravado_no_caminho_feliz(db_temp, mock_llm):
    _seed_objetos(db_temp, qtd=1)
    mock_llm.side_effect = [
        ("SELECT nome FROM objetos", "ollama"),
        ("achei 1", "ollama"),
    ]

    from core.database import get_db
    from agents.assistente import chat
    conn = get_db()
    try:
        chat("pergunta?", conn)
        h = conn.execute(
            "SELECT pergunta, resposta, modelo FROM historico_chat"
        ).fetchall()
        assert len(h) == 1
        assert h[0]["pergunta"] == "pergunta?"
        assert h[0]["resposta"] == "achei 1"
        assert h[0]["modelo"] == "ollama"
    finally:
        conn.close()


def test_historico_gravado_mesmo_quando_sql_e_bloqueado(db_temp, mock_llm):
    """Auditoria: tentativa de SQL malicioso fica registrada no histórico."""
    mock_llm.side_effect = [
        ("DROP TABLE objetos", "ollama"),
        ("Falha ao buscar.", "ollama"),
    ]

    from core.database import get_db
    from agents.assistente import chat
    conn = get_db()
    try:
        chat("hack tentativa", conn)
        h = conn.execute(
            "SELECT pergunta, resposta FROM historico_chat"
        ).fetchall()
        assert len(h) == 1
        assert h[0]["pergunta"] == "hack tentativa"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# LLM falha completamente (timeout, etc.) → assistente degrada graciosamente
# ---------------------------------------------------------------------------

def test_falha_total_do_llm_nao_quebra_assistente(db_temp, mock_llm):
    mock_llm.side_effect = RuntimeError("LLM offline")

    from core.database import get_db
    from agents.assistente import chat
    conn = get_db()
    try:
        resp = chat("pergunta", conn)
    finally:
        conn.close()

    assert resp["sql"] == ""
    assert resp["resultados"] == []
    # A 2ª chamada também falhou → resposta usa fallback de erro
    assert resp["resposta"]
