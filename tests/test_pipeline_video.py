"""Testes de pipeline/video.processar_video.

Estratégia: mocks de _extrair_keyframes (evita ffmpeg) + dos 4 agentes em
pipeline.video. DB SQLite isolado via fixture db_temp. Sem rede, sem LLM.

O que está coberto (foco em comportamento exclusivo do pipeline de vídeo):
- Happy path: N keyframes → M objetos detectados → dedup → INSERTs + status
- Dedup por nome normalizado: mesma chave em 2 frames mantém maior confiança
- Dedup preserva enriquecimento vencedor (frame_origem do quadro com maior conf)
- frames_extraidos e frames_processados refletem o trabalho real
- synthetic_id único por (video_db_id, frame) — sem colisões com fotos
- Erro crítico: 0 keyframes → status='erro'
- Resiliência: erro em 1 objeto não derruba os outros do mesmo frame
- Resiliência: erro em 1 frame não derruba os demais frames
- Categoria desconhecida → cat_id NULL
- Nome vazio é descartado da dedup
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
def video_pendente(db_temp, localizacao_id) -> int:
    """Linha em videos_processados com status='pendente' e devolve o id."""
    from core.database import get_db
    conn = get_db()
    try:
        cur = conn.execute(
            "INSERT INTO videos_processados (caminho, localizacao_id, status) "
            "VALUES (?, ?, 'pendente')",
            ("/fake/video.mp4", localizacao_id),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def _enriquecido_default(nome: str = "objeto", confianca: float = 0.9,
                          _modelo: str = "ollama") -> dict:
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
        "palavras_chave": "|x|",
        "confianca": confianca,
        "_modelo": _modelo,
    }


@pytest.fixture
def mock_agentes(mocker):
    """Mocks default razoáveis para os 4 agentes + extrator de keyframes.

    Default: 2 keyframes; cada um detecta 1 objeto distinto.
    Cada teste customiza via dict (mocker patches), ex:
        mock_agentes['kf'].return_value = [...]
        mock_agentes['seg'].side_effect = [...]
    """
    kf = mocker.patch(
        "pipeline.video._extrair_keyframes",
        return_value=["/fake/frame_001.jpg", "/fake/frame_002.jpg"],
    )
    seg = mocker.patch(
        "pipeline.video.segmentar_foto",
        side_effect=[
            [{"nome": "martelo", "recorte_path": "/fake/m1.jpg"}],
            [{"nome": "chave",   "recorte_path": "/fake/c1.jpg"}],
        ],
    )
    ana = mocker.patch(
        "pipeline.video.analisar_objeto",
        side_effect=lambda recorte, nome: {
            "nome": nome, "confianca": 0.9, "_modelo": "ollama"
        },
    )
    enr = mocker.patch(
        "pipeline.video.enriquecer_objeto",
        side_effect=lambda analise, cats: _enriquecido_default(
            nome=analise.get("nome", "objeto"),
            confianca=analise.get("confianca", 0.9),
            _modelo=analise.get("_modelo", "ollama"),
        ),
    )
    ico = mocker.patch(
        "pipeline.video.gerar_icone",
        return_value=("/fake/icone.png", "claude_desenho"),
    )
    return {"kf": kf, "seg": seg, "ana": ana, "enr": enr, "ico": ico}


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_happy_path_dois_frames_dois_objetos_unicos(
    db_temp, video_pendente, localizacao_id, mock_agentes
):
    from core.database import get_db
    from pipeline.video import processar_video

    processar_video("/fake/video.mp4", localizacao_id, video_pendente)

    conn = get_db()
    try:
        v = conn.execute(
            "SELECT status, objetos_encontrados, frames_extraidos, "
            "frames_processados, erro_mensagem, iniciado_em, concluido_em "
            "FROM videos_processados WHERE id=?",
            (video_pendente,),
        ).fetchone()
        assert v["status"] == "concluido"
        assert v["objetos_encontrados"] == 2
        assert v["frames_extraidos"] == 2
        assert v["frames_processados"] == 2
        assert v["erro_mensagem"] is None
        assert v["iniciado_em"] is not None
        assert v["concluido_em"] is not None

        objetos = conn.execute(
            "SELECT nome, icone_path, modelo_visao "
            "FROM objetos WHERE localizacao_id=? ORDER BY nome",
            (localizacao_id,),
        ).fetchall()
        assert [o["nome"] for o in objetos] == ["chave", "martelo"]
        for o in objetos:
            assert o["icone_path"] == "/fake/icone.png"
            assert o["modelo_visao"] == "ollama"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Dedup por nome — maior confiança vence
# ---------------------------------------------------------------------------

def test_dedup_mantem_deteccao_de_maior_confianca(
    db_temp, video_pendente, localizacao_id, mock_agentes
):
    """Mesmo objeto (martelo) aparece em 2 frames; segundo tem confiança maior.
    Resultado: 1 INSERT só, com a confiança do segundo frame."""
    mock_agentes["seg"].side_effect = [
        [{"nome": "martelo", "recorte_path": "/f1/martelo.jpg"}],
        [{"nome": "martelo", "recorte_path": "/f2/martelo.jpg"}],
    ]
    mock_agentes["ana"].side_effect = [
        {"nome": "martelo", "confianca": 0.60, "_modelo": "ollama"},
        {"nome": "martelo", "confianca": 0.95, "_modelo": "claude"},
    ]

    from core.database import get_db
    from pipeline.video import processar_video
    processar_video("/fake/video.mp4", localizacao_id, video_pendente)

    conn = get_db()
    try:
        objetos = conn.execute(
            "SELECT nome, confianca, foto_original_path, modelo_visao FROM objetos"
        ).fetchall()
        assert len(objetos) == 1
        obj = objetos[0]
        assert obj["nome"] == "martelo"
        assert obj["confianca"] == pytest.approx(0.95)
        # _frame_origem do enriquecido vencedor → coluna foto_original_path
        assert obj["foto_original_path"] == "/fake/frame_002.jpg"
        assert obj["modelo_visao"] == "claude"

        v = conn.execute(
            "SELECT objetos_encontrados FROM videos_processados WHERE id=?",
            (video_pendente,),
        ).fetchone()
        assert v["objetos_encontrados"] == 1
    finally:
        conn.close()


def test_dedup_mantem_primeiro_quando_segundo_tem_confianca_menor(
    db_temp, video_pendente, localizacao_id, mock_agentes
):
    """Inverso do teste anterior: garante que o critério é > (maior vence)
    e não 'último vence'."""
    mock_agentes["seg"].side_effect = [
        [{"nome": "Martelo", "recorte_path": "/f1.jpg"}],   # caixa diferente
        [{"nome": "martelo", "recorte_path": "/f2.jpg"}],
    ]
    mock_agentes["ana"].side_effect = [
        {"nome": "Martelo", "confianca": 0.95, "_modelo": "claude"},
        {"nome": "martelo", "confianca": 0.50, "_modelo": "ollama"},
    ]

    from core.database import get_db
    from pipeline.video import processar_video
    processar_video("/fake/video.mp4", localizacao_id, video_pendente)

    conn = get_db()
    try:
        objetos = conn.execute(
            "SELECT confianca, foto_original_path FROM objetos"
        ).fetchall()
        assert len(objetos) == 1
        # Normalização lower → mesma chave; maior confiança (0.95) prevalece
        assert objetos[0]["confianca"] == pytest.approx(0.95)
        assert objetos[0]["foto_original_path"] == "/fake/frame_001.jpg"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# synthetic_id (segmentar_foto recebe id derivado de video_db_id + idx)
# ---------------------------------------------------------------------------

def test_segmentador_recebe_synthetic_id_unico_por_frame(
    db_temp, video_pendente, localizacao_id, mock_agentes
):
    """synthetic_id = 100000 + video_db_id*100 + idx — evita colisão com
    foto_db_id real ao reutilizar segmentar_foto."""
    mock_agentes["kf"].return_value = [
        "/fake/f1.jpg", "/fake/f2.jpg", "/fake/f3.jpg"
    ]
    mock_agentes["seg"].side_effect = [[], [], []]

    from pipeline.video import processar_video
    processar_video("/fake/video.mp4", localizacao_id, video_pendente)

    chamadas = mock_agentes["seg"].call_args_list
    ids_passados = [c.args[1] for c in chamadas]
    esperados = [100000 + video_pendente * 100 + i for i in range(3)]
    assert ids_passados == esperados
    # Sanity: todos > 100000 (não colidem com IDs reais de fotos_processadas)
    assert all(i >= 100000 for i in ids_passados)


# ---------------------------------------------------------------------------
# Erro crítico: 0 keyframes
# ---------------------------------------------------------------------------

def test_zero_keyframes_marca_video_como_erro(
    db_temp, video_pendente, localizacao_id, mock_agentes
):
    mock_agentes["kf"].return_value = []

    from core.database import get_db
    from pipeline.video import processar_video
    processar_video("/fake/video.mp4", localizacao_id, video_pendente)

    conn = get_db()
    try:
        v = conn.execute(
            "SELECT status, erro_mensagem, frames_extraidos, objetos_encontrados "
            "FROM videos_processados WHERE id=?",
            (video_pendente,),
        ).fetchone()
        assert v["status"] == "erro"
        assert "keyframe" in (v["erro_mensagem"] or "").lower()
        assert v["frames_extraidos"] == 0
        assert v["objetos_encontrados"] == 0
        # Agentes 1-4 nunca foram chamados
        mock_agentes["seg"].assert_not_called()
        mock_agentes["ico"].assert_not_called()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Resiliência: erro em 1 objeto não derruba os outros do mesmo frame
# ---------------------------------------------------------------------------

def test_falha_em_um_objeto_nao_derruba_o_outro_no_mesmo_frame(
    db_temp, video_pendente, localizacao_id, mock_agentes
):
    mock_agentes["kf"].return_value = ["/fake/frame.jpg"]
    mock_agentes["seg"].side_effect = [[
        {"nome": "ok",    "recorte_path": "/ok.jpg"},
        {"nome": "ruim",  "recorte_path": "/r.jpg"},
        {"nome": "ok2",   "recorte_path": "/ok2.jpg"},
    ]]
    mock_agentes["ana"].side_effect = [
        {"nome": "ok",  "confianca": 0.9, "_modelo": "ollama"},
        RuntimeError("erro analisador"),
        {"nome": "ok2", "confianca": 0.8, "_modelo": "ollama"},
    ]

    from core.database import get_db
    from pipeline.video import processar_video
    processar_video("/fake/video.mp4", localizacao_id, video_pendente)

    conn = get_db()
    try:
        v = conn.execute(
            "SELECT status, objetos_encontrados FROM videos_processados WHERE id=?",
            (video_pendente,),
        ).fetchone()
        assert v["status"] == "concluido"   # erros por-objeto não viram crítico
        assert v["objetos_encontrados"] == 2
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Resiliência: erro em 1 frame não impede os demais
# ---------------------------------------------------------------------------

def test_erro_em_um_frame_nao_derruba_loop_de_frames(
    db_temp, video_pendente, localizacao_id, mock_agentes
):
    """Se segmentar_foto explode no frame 2, frames 1 e 3 ainda são processados
    e frames_processados=3 reflete o trabalho tentado."""
    mock_agentes["kf"].return_value = [
        "/fake/f1.jpg", "/fake/f2.jpg", "/fake/f3.jpg"
    ]
    mock_agentes["seg"].side_effect = [
        [{"nome": "a", "recorte_path": "/a.jpg"}],
        RuntimeError("frame 2 corrompido"),
        [{"nome": "b", "recorte_path": "/b.jpg"}],
    ]

    from core.database import get_db
    from pipeline.video import processar_video
    processar_video("/fake/video.mp4", localizacao_id, video_pendente)

    conn = get_db()
    try:
        v = conn.execute(
            "SELECT status, objetos_encontrados, frames_processados "
            "FROM videos_processados WHERE id=?",
            (video_pendente,),
        ).fetchone()
        assert v["status"] == "concluido"
        assert v["objetos_encontrados"] == 2  # 'a' e 'b' sobreviveram
        assert v["frames_processados"] == 3   # contador avança mesmo em erro
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Categoria desconhecida → cat_id NULL
# ---------------------------------------------------------------------------

def test_categoria_desconhecida_resulta_em_categoria_id_nulo(
    db_temp, video_pendente, localizacao_id, mock_agentes
):
    mock_agentes["kf"].return_value = ["/fake/f.jpg"]
    mock_agentes["seg"].side_effect = [[{"nome": "x", "recorte_path": "/x.jpg"}]]
    mock_agentes["enr"].side_effect = lambda analise, cats: {
        **_enriquecido_default("misterioso"),
        "categoria_nome": "CategoriaQueNaoExiste",
    }

    from core.database import get_db
    from pipeline.video import processar_video
    processar_video("/fake/video.mp4", localizacao_id, video_pendente)

    conn = get_db()
    try:
        obj = conn.execute("SELECT categoria_id, nome FROM objetos").fetchone()
        assert obj is not None
        assert obj["categoria_id"] is None
        assert obj["nome"] == "misterioso"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Objeto com nome vazio é descartado antes do INSERT
# ---------------------------------------------------------------------------

def test_objeto_com_nome_vazio_e_descartado(
    db_temp, video_pendente, localizacao_id, mock_agentes
):
    """_normalizar_nome('') = '' → 'if not chave: continue' descarta.
    Atenção: o código faz `enriquecido.get("nome") or obj["nome"]`, então
    para chave ficar vazia AMBOS precisam ser falsy."""
    mock_agentes["kf"].return_value = ["/fake/f.jpg"]
    mock_agentes["seg"].side_effect = [[
        {"nome": "",          "recorte_path": "/v.jpg"},
        {"nome": "valido",    "recorte_path": "/ok.jpg"},
    ]]
    mock_agentes["enr"].side_effect = [
        {**_enriquecido_default(""), "nome": ""},
        _enriquecido_default("valido"),
    ]

    from core.database import get_db
    from pipeline.video import processar_video
    processar_video("/fake/video.mp4", localizacao_id, video_pendente)

    conn = get_db()
    try:
        nomes = [r["nome"] for r in conn.execute("SELECT nome FROM objetos").fetchall()]
        assert nomes == ["valido"]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Agente 4 recebe objeto_id real do INSERT
# ---------------------------------------------------------------------------

def test_agente_4_recebe_objeto_id_da_linha_recem_inserida(
    db_temp, video_pendente, localizacao_id, mock_agentes
):
    mock_agentes["kf"].return_value = ["/fake/f.jpg"]
    mock_agentes["seg"].side_effect = [[{"nome": "x", "recorte_path": "/x.jpg"}]]

    from core.database import get_db
    from pipeline.video import processar_video
    processar_video("/fake/video.mp4", localizacao_id, video_pendente)

    chamada = mock_agentes["ico"].call_args
    objeto_id_passado = chamada.kwargs["objeto_id"]
    assert objeto_id_passado is not None and objeto_id_passado > 0
    conn = get_db()
    try:
        obj = conn.execute("SELECT id FROM objetos").fetchone()
        assert obj["id"] == objeto_id_passado
    finally:
        conn.close()
