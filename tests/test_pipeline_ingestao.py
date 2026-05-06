"""Testes de pipeline/ingestao.processar_foto.

Estratégia: mocks dos 4 agentes em pipeline.ingestao + DB SQLite isolado
(fixture db_temp). Não toca rede, não chama LLM.

O que está coberto:
- Happy path: 2 objetos detectados → 2 INSERTs + UPDATE de status
- Resiliência: falha em 1 objeto não derruba os outros
- Erro crítico: falha no Agente 1 marca a foto como 'erro'
- Regressão P5: SELECT em categorias roda UMA vez, antes do loop
- Categoria desconhecida → cat_id NULL (fallback gracioso)
- Agente 4 recebe objeto_id real (cursor.lastrowid pós-INSERT)
- Foto sem objetos detectados termina como 'concluido' com 0
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Fixtures locais
# ---------------------------------------------------------------------------

@pytest.fixture
def localizacao_id(db_temp) -> int:
    from core.database import get_db
    conn = get_db()
    try:
        cur = conn.execute(
            "INSERT INTO localizacoes (nome, tipo) VALUES (?, ?)",
            ("Garagem", "caixa"),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


@pytest.fixture
def foto_pendente(db_temp, localizacao_id) -> int:
    """Cria uma linha em fotos_processadas com status='pendente' e devolve o id."""
    from core.database import get_db
    conn = get_db()
    try:
        cur = conn.execute(
            "INSERT INTO fotos_processadas (caminho, localizacao_id, status) "
            "VALUES (?, ?, 'pendente')",
            ("/fake/foto.jpg", localizacao_id),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def _enriquecido_default(nome: str = "objeto") -> dict:
    return {
        "nome": nome,
        "categoria_nome": "Ferramentas",
        "descricao": "descrição",
        "cor": "preto",
        "tamanho": "médio",
        "tamanho_estimado_cm": "20x10x5",
        "peso_estimado_g": 200,
        "material": "metal",
        "estado": "bom",
        "funcao": "uso geral",
        "palavras_chave": "|martelo|ferramenta|",
        "confianca": 0.9,
        "_modelo": "ollama",
    }


@pytest.fixture
def mock_agentes(mocker):
    """Mocka os 4 agentes em pipeline.ingestao com defaults razoáveis.

    Cada teste pode customizar via dict retornado, ex:
        mock_agentes['seg'].return_value = [...]
        mock_agentes['ana'].side_effect = [...]
    """
    seg = mocker.patch(
        "pipeline.ingestao.segmentar_foto",
        return_value=[
            {"nome": "martelo", "recorte_path": "/fake/martelo.jpg"},
            {"nome": "chave",   "recorte_path": "/fake/chave.jpg"},
        ],
    )
    ana = mocker.patch(
        "pipeline.ingestao.analisar_objeto",
        return_value={"nome": "objeto", "confianca": 0.9, "_modelo": "ollama"},
    )
    enr = mocker.patch(
        "pipeline.ingestao.enriquecer_objeto",
        side_effect=lambda analise, cats: _enriquecido_default(analise.get("nome", "objeto")),
    )
    ico = mocker.patch(
        "pipeline.ingestao.gerar_icone",
        return_value=("/fake/icone.png", "claude_desenho"),
    )
    return {"seg": seg, "ana": ana, "enr": enr, "ico": ico}


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_happy_path_dois_objetos_inseridos_e_status_concluido(
    db_temp, foto_pendente, localizacao_id, mock_agentes
):
    from core.database import get_db
    from pipeline.ingestao import processar_foto

    processar_foto("/fake/foto.jpg", localizacao_id, foto_pendente)

    conn = get_db()
    try:
        foto = conn.execute(
            "SELECT status, objetos_encontrados, erro_mensagem, iniciado_em, concluido_em "
            "FROM fotos_processadas WHERE id=?",
            (foto_pendente,),
        ).fetchone()
        assert foto["status"] == "concluido"
        assert foto["objetos_encontrados"] == 2
        assert foto["erro_mensagem"] is None
        assert foto["iniciado_em"] is not None
        assert foto["concluido_em"] is not None

        objetos = conn.execute(
            "SELECT nome, categoria_id, icone_path, icone_fonte, modelo_visao "
            "FROM objetos WHERE localizacao_id=? ORDER BY id",
            (localizacao_id,),
        ).fetchall()
        assert len(objetos) == 2
        for obj in objetos:
            assert obj["icone_path"] == "/fake/icone.png"
            assert obj["icone_fonte"] == "claude_desenho"
            assert obj["categoria_id"] is not None  # 'Ferramentas' existe no seed
            assert obj["modelo_visao"] == "ollama"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Resiliência: falha em 1 objeto não derruba os outros
# ---------------------------------------------------------------------------

def test_falha_em_um_objeto_nao_derruba_o_outro(
    db_temp, foto_pendente, localizacao_id, mock_agentes
):
    """Agente 2 quebra no segundo objeto. Resultado: 1 objeto inserido, status=concluido."""
    mock_agentes["ana"].side_effect = [
        {"nome": "ok",  "confianca": 0.9, "_modelo": "ollama"},
        RuntimeError("falha no agente 2"),
    ]

    from core.database import get_db
    from pipeline.ingestao import processar_foto
    processar_foto("/fake/foto.jpg", localizacao_id, foto_pendente)

    conn = get_db()
    try:
        foto = conn.execute(
            "SELECT status, objetos_encontrados FROM fotos_processadas WHERE id=?",
            (foto_pendente,),
        ).fetchone()
        assert foto["status"] == "concluido"   # erro por objeto NÃO marca foto como erro
        assert foto["objetos_encontrados"] == 1
        n = conn.execute("SELECT COUNT(*) AS c FROM objetos").fetchone()
        assert n["c"] == 1
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Erro crítico: Agente 1 falha → foto marcada como 'erro'
# ---------------------------------------------------------------------------

def test_falha_no_agente_1_marca_foto_como_erro(
    db_temp, foto_pendente, localizacao_id, mocker
):
    mocker.patch(
        "pipeline.ingestao.segmentar_foto",
        side_effect=RuntimeError("imagem corrompida"),
    )

    from core.database import get_db
    from pipeline.ingestao import processar_foto
    processar_foto("/fake/foto.jpg", localizacao_id, foto_pendente)

    conn = get_db()
    try:
        foto = conn.execute(
            "SELECT status, erro_mensagem, objetos_encontrados "
            "FROM fotos_processadas WHERE id=?",
            (foto_pendente,),
        ).fetchone()
        assert foto["status"] == "erro"
        assert "imagem corrompida" in (foto["erro_mensagem"] or "")
        assert foto["objetos_encontrados"] == 0
        n = conn.execute("SELECT COUNT(*) AS c FROM objetos").fetchone()
        assert n["c"] == 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Regressão P5: categorias buscadas UMA vez, antes do loop
# ---------------------------------------------------------------------------

def test_categorias_buscadas_uma_unica_vez_antes_do_loop(
    db_temp, foto_pendente, localizacao_id, mock_agentes, mocker
):
    """Antes da P5 a query de categorias rodava dentro do loop — N objetos =
    N consultas. Agora deve rodar exatamente 1x (a query SEM cláusula WHERE).

    sqlite3.Connection é um tipo C imutável (não dá pra usar mocker.spy nele),
    então envolvemos a conexão real num wrapper que registra cada SQL antes de
    delegar.
    """
    queries: list[str] = []

    from core.database import get_db as _real_get_db

    class ConnSpy:
        def __init__(self, real):
            self._real = real
        def execute(self, sql, *args, **kwargs):
            queries.append(sql)
            return self._real.execute(sql, *args, **kwargs)
        def __getattr__(self, name):
            return getattr(self._real, name)

    mocker.patch(
        "pipeline.ingestao.get_db",
        side_effect=lambda: ConnSpy(_real_get_db()),
    )
    mock_agentes["seg"].return_value = [
        {"nome": f"obj{i}", "recorte_path": f"/o{i}.jpg"} for i in range(5)
    ]

    from pipeline.ingestao import processar_foto
    processar_foto("/fake/foto.jpg", localizacao_id, foto_pendente)

    queries_categorias_completas = [
        q for q in queries
        if "from categorias" in q.lower() and "where" not in q.lower()
    ]
    assert len(queries_categorias_completas) == 1, (
        f"esperado 1 SELECT completo de 'categorias', encontrado "
        f"{len(queries_categorias_completas)}: {queries_categorias_completas}"
    )


# ---------------------------------------------------------------------------
# Categoria desconhecida → cat_id NULL (fallback gracioso)
# ---------------------------------------------------------------------------

def test_categoria_desconhecida_resulta_em_categoria_id_nulo(
    db_temp, foto_pendente, localizacao_id, mock_agentes
):
    mock_agentes["seg"].return_value = [{"nome": "x", "recorte_path": "/x.jpg"}]
    mock_agentes["enr"].side_effect = lambda *a, **kw: {
        **_enriquecido_default("misterioso"),
        "categoria_nome": "CategoriaQueNaoExiste",
    }

    from core.database import get_db
    from pipeline.ingestao import processar_foto
    processar_foto("/fake/foto.jpg", localizacao_id, foto_pendente)

    conn = get_db()
    try:
        obj = conn.execute("SELECT categoria_id, nome FROM objetos").fetchone()
        assert obj is not None
        assert obj["categoria_id"] is None
        assert obj["nome"] == "misterioso"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Agente 4 recebe objeto_id real (cursor.lastrowid pós-INSERT)
# ---------------------------------------------------------------------------

def test_agente_4_recebe_objeto_id_da_linha_recem_inserida(
    db_temp, foto_pendente, localizacao_id, mock_agentes
):
    """O Agente 4 só pode rodar após o INSERT — precisa do ID real do objeto
    para usar como nome de arquivo do ícone."""
    mock_agentes["seg"].return_value = [{"nome": "x", "recorte_path": "/x.jpg"}]

    from core.database import get_db
    from pipeline.ingestao import processar_foto
    processar_foto("/fake/foto.jpg", localizacao_id, foto_pendente)

    chamada = mock_agentes["ico"].call_args
    objeto_id_passado = chamada.kwargs["objeto_id"]
    assert objeto_id_passado is not None and objeto_id_passado > 0

    conn = get_db()
    try:
        obj = conn.execute("SELECT id FROM objetos").fetchone()
        assert obj["id"] == objeto_id_passado
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Foto sem objetos detectados → 'concluido' com 0
# ---------------------------------------------------------------------------

def test_zero_objetos_detectados_termina_concluido_com_zero(
    db_temp, foto_pendente, localizacao_id, mock_agentes
):
    mock_agentes["seg"].return_value = []

    from core.database import get_db
    from pipeline.ingestao import processar_foto
    processar_foto("/fake/foto.jpg", localizacao_id, foto_pendente)

    conn = get_db()
    try:
        foto = conn.execute(
            "SELECT status, objetos_encontrados, erro_mensagem "
            "FROM fotos_processadas WHERE id=?",
            (foto_pendente,),
        ).fetchone()
        assert foto["status"] == "concluido"
        assert foto["objetos_encontrados"] == 0
        assert foto["erro_mensagem"] is None
        n = conn.execute("SELECT COUNT(*) AS c FROM objetos").fetchone()
        assert n["c"] == 0
        # Agentes 2-4 nunca devem ser chamados se Agente 1 não detectou nada
        mock_agentes["ana"].assert_not_called()
        mock_agentes["enr"].assert_not_called()
        mock_agentes["ico"].assert_not_called()
    finally:
        conn.close()
