"""Smoke tests dos routers FastAPI.

Cobre 1 happy path + 1 erro relevante por router. Não exercita LLM real:
- pipeline.ingestao.processar_foto mockado (BackgroundTasks chama dummy)
- agents.assistente.chat mockado (chat endpoint não bate em Ollama/Claude)
- ollama.list mockado (modelos endpoint não exige Ollama vivo)

A fixture db_temp do conftest aponta DB_PATH pro tmp_path antes do TestClient
ser instanciado — então o startup event (init_db) cria as tabelas no banco
isolado. O modo está irrelevante aqui (só exercitamos as rotas).
"""
from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Cliente HTTP com DB isolado
# ---------------------------------------------------------------------------

@pytest.fixture
def client(db_temp):
    """TestClient com lifespan ativo (dispara startup → init_db no DB de teste)."""
    from api.app import app
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# /api/modo  (definido direto em app.py)
# ---------------------------------------------------------------------------

def test_modo_retorna_descricao_e_modelos_atuais(client):
    r = client.get("/api/modo")
    assert r.status_code == 200
    body = r.json()
    assert "modo" in body and "descricao" in body
    assert "vision_model" in body and "text_model" in body
    assert isinstance(body["tem_api_key"], bool)


# ---------------------------------------------------------------------------
# /api/localizacoes
# ---------------------------------------------------------------------------

def test_localizacoes_post_e_get_basico(client):
    r = client.post("/api/localizacoes",
                    json={"nome": "Garagem", "tipo": "armário", "comodo": "Garagem"})
    assert r.status_code == 200, r.text
    loc = r.json()
    assert loc["id"] > 0 and loc["nome"] == "Garagem"

    r = client.get("/api/localizacoes")
    assert r.status_code == 200
    lista = r.json()
    assert any(item["id"] == loc["id"] for item in lista)
    item = next(i for i in lista if i["id"] == loc["id"])
    assert item["total_objetos"] == 0


def test_localizacoes_delete_com_objetos_responde_400(client, db_temp):
    """Não deve permitir excluir localização que tem objetos — proteção contra perda de dados."""
    r = client.post("/api/localizacoes", json={"nome": "Cozinha"})
    loc_id = r.json()["id"]

    # Insere objeto vinculado direto no DB
    from core.database import get_db
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO objetos (nome, localizacao_id) VALUES (?, ?)",
            ("faca", loc_id),
        )
        conn.commit()
    finally:
        conn.close()

    r = client.delete(f"/api/localizacoes/{loc_id}")
    assert r.status_code == 400
    assert "objeto" in r.json()["detail"].lower()


# ---------------------------------------------------------------------------
# /api/fotos
# ---------------------------------------------------------------------------

def test_fotos_ingerir_e_consultar_status(client, mocker):
    """Upload → registra foto pendente, dispara BackgroundTask (mockado)."""
    mock_proc = mocker.patch("api.routes.fotos.processar_foto")

    r = client.post("/api/localizacoes", json={"nome": "Sala"})
    loc_id = r.json()["id"]

    arquivo_jpeg = io.BytesIO(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01")
    r = client.post(
        "/api/fotos/ingerir",
        data={"localizacao_id": str(loc_id)},
        files={"arquivo": ("teste.jpg", arquivo_jpeg, "image/jpeg")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "pendente"
    foto_id = body["foto_id"]

    # Background task foi enfileirada com os args certos
    mock_proc.assert_called_once()
    args = mock_proc.call_args.args
    assert args[1] == loc_id and args[2] == foto_id

    # GET status — ainda 'pendente' ou já 'processando' (BackgroundTask roda após response)
    r = client.get(f"/api/fotos/{foto_id}/status")
    assert r.status_code == 200
    s = r.json()
    assert s["id"] == foto_id
    assert s["status"] in {"pendente", "processando", "concluido"}


def test_fotos_ingerir_extensao_invalida_responde_400(client, mocker):
    mocker.patch("api.routes.fotos.processar_foto")
    r = client.post("/api/localizacoes", json={"nome": "Quintal"})
    loc_id = r.json()["id"]

    arquivo = io.BytesIO(b"qualquer coisa")
    r = client.post(
        "/api/fotos/ingerir",
        data={"localizacao_id": str(loc_id)},
        files={"arquivo": ("doc.pdf", arquivo, "application/pdf")},
    )
    assert r.status_code == 400
    assert "extens" in r.json()["detail"].lower()


def test_fotos_status_inexistente_responde_404(client):
    r = client.get("/api/fotos/99999/status")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# /api/objetos
# ---------------------------------------------------------------------------

def test_objetos_listar_vazio_retorna_array_vazio(client):
    r = client.get("/api/objetos")
    assert r.status_code == 200
    assert r.json() == []


def test_objetos_obter_inexistente_responde_404(client):
    r = client.get("/api/objetos/999")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# /api/estatisticas
# ---------------------------------------------------------------------------

def test_estatisticas_estrutura_basica(client):
    r = client.get("/api/estatisticas")
    assert r.status_code == 200
    body = r.json()
    for chave in ("total_objetos", "total_localizacoes", "total_fotos",
                  "por_categoria", "por_comodo", "modo_ativo"):
        assert chave in body
    assert body["total_objetos"] == 0
    assert body["modo_ativo"]["modo"]  # string não-vazia


# ---------------------------------------------------------------------------
# /api/modelos
# ---------------------------------------------------------------------------

def test_modelos_listar_separa_visao_e_texto(client, mocker):
    """Separa por heurística no nome — qwen2.5vl é visão, llama3.2 é texto."""
    mocker.patch(
        "api.routes.modelos.ollama.list",
        return_value={"models": [
            {"name": "qwen2.5vl:7b"},
            {"name": "llava:13b"},
            {"name": "llama3.2:3b"},
        ]},
    )
    r = client.get("/api/modelos")
    assert r.status_code == 200
    body = r.json()
    assert "qwen2.5vl:7b" in body["vision_models"]
    assert "llava:13b" in body["vision_models"]
    assert "llama3.2:3b" in body["text_models"]
    assert "atual" in body and "vision_model" in body["atual"]


def test_modelos_responde_503_se_ollama_indisponivel(client, mocker):
    mocker.patch("api.routes.modelos.ollama.list",
                 side_effect=ConnectionError("ollama down"))
    r = client.get("/api/modelos")
    assert r.status_code == 503


# ---------------------------------------------------------------------------
# /api/chat
# ---------------------------------------------------------------------------

def test_chat_endpoint_delega_para_assistente_mockado(client, mocker):
    mock = mocker.patch(
        "api.routes.chat.assistente_chat",
        return_value={"resposta": "ok", "sql": "", "resultados": [], "modelo": "ollama"},
    )
    r = client.post("/api/chat", json={"pergunta": "quantos objetos tenho?"})
    assert r.status_code == 200
    assert r.json()["resposta"] == "ok"
    mock.assert_called_once()
    assert mock.call_args.args[0] == "quantos objetos tenho?"


def test_chat_historico_vazio_retorna_lista_vazia(client):
    r = client.get("/api/chat/historico")
    assert r.status_code == 200
    assert r.json() == []


# ---------------------------------------------------------------------------
# /api/videos
# ---------------------------------------------------------------------------

def test_videos_ingerir_e_consultar_status(client, mocker):
    """Upload → registra vídeo pendente, dispara BackgroundTask (mockado)."""
    mock_proc = mocker.patch("api.routes.videos.processar_video")

    r = client.post("/api/localizacoes", json={"nome": "Sala"})
    loc_id = r.json()["id"]

    arquivo_mp4 = io.BytesIO(b"\x00\x00\x00\x20ftypisom")
    r = client.post(
        "/api/videos/ingerir",
        data={"localizacao_id": str(loc_id)},
        files={"arquivo": ("teste.mp4", arquivo_mp4, "video/mp4")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "pendente"
    video_id = body["video_id"]

    # Background task foi enfileirada com os args certos
    mock_proc.assert_called_once()
    args = mock_proc.call_args.args
    assert args[1] == loc_id and args[2] == video_id

    # GET status — ainda 'pendente' ou já 'processando'
    r = client.get(f"/api/videos/{video_id}/status")
    assert r.status_code == 200
    s = r.json()
    assert s["id"] == video_id
    assert s["status"] in {"pendente", "processando", "concluido", "erro"}


def test_videos_ingerir_extensao_invalida_responde_400(client, mocker):
    mocker.patch("api.routes.videos.processar_video")
    r = client.post("/api/localizacoes", json={"nome": "Quintal"})
    loc_id = r.json()["id"]

    arquivo = io.BytesIO(b"qualquer coisa")
    r = client.post(
        "/api/videos/ingerir",
        data={"localizacao_id": str(loc_id)},
        files={"arquivo": ("doc.pdf", arquivo, "application/pdf")},
    )
    assert r.status_code == 400
    assert "extens" in r.json()["detail"].lower()


def test_videos_ingerir_localizacao_inexistente_responde_404(client, mocker):
    mocker.patch("api.routes.videos.processar_video")
    arquivo = io.BytesIO(b"\x00\x00\x00\x20ftypisom")
    r = client.post(
        "/api/videos/ingerir",
        data={"localizacao_id": "99999"},
        files={"arquivo": ("teste.mp4", arquivo, "video/mp4")},
    )
    assert r.status_code == 404
    assert "localiz" in r.json()["detail"].lower()


def test_videos_status_inexistente_responde_404(client):
    r = client.get("/api/videos/99999/status")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# /api/categorias
# ---------------------------------------------------------------------------

def test_categorias_listar_retorna_lista(client):
    r = client.get("/api/categorias")
    assert r.status_code == 200
    cats = r.json()
    assert isinstance(cats, list)
    assert len(cats) > 0
    primeira = cats[0]
    assert "nome" in primeira and "total_objetos" in primeira


# ---------------------------------------------------------------------------
# /api/objetos batch operations
# ---------------------------------------------------------------------------

def test_objetos_batch_delete_e_move(client, db_temp):
    from core.database import get_db
    conn = get_db()
    try:
        conn.execute("INSERT INTO localizacoes (id, nome) VALUES (10, 'Loc A'), (20, 'Loc B')")
        conn.execute("INSERT INTO objetos (id, nome, localizacao_id) VALUES (101, 'Item 1', 10), (102, 'Item 2', 10)")
        conn.commit()
    finally:
        conn.close()

    r_move = client.post("/api/objetos/batch-move", json={"ids": [101], "localizacao_id": 20})
    assert r_move.status_code == 200
    assert r_move.json()["movidos"] == 1

    r_del = client.post("/api/objetos/batch-delete", json={"ids": [102]})
    assert r_del.status_code == 200
    assert r_del.json()["deletados"] == 1

    r_empty = client.post("/api/objetos/batch-delete", json={"ids": []})
    assert r_empty.status_code == 200
    assert r_empty.json()["deletados"] == 0


# ---------------------------------------------------------------------------
# /api/fotos/progresso, /cancelar e /reprocessar
# ---------------------------------------------------------------------------

def test_fotos_progresso_cancelar_e_reprocessar(client, db_temp, mocker, tmp_path):
    mock_proc = mocker.patch("api.routes.fotos.processar_foto")
    fake_img = tmp_path / "fake.jpg"
    fake_img.write_bytes(b"dummy image data")

    from core.database import get_db
    conn = get_db()
    try:
        conn.execute("INSERT INTO localizacoes (id, nome) VALUES (1, 'Sala')")
        cursor = conn.execute(
            "INSERT INTO fotos_processadas (caminho, localizacao_id, status) VALUES (?,?,?)",
            (str(fake_img), 1, "processando")
        )
        conn.commit()
        foto_id = cursor.lastrowid
    finally:
        conn.close()

    r_prog = client.get(f"/api/fotos/{foto_id}/progresso")
    assert r_prog.status_code == 200
    assert r_prog.json()["id"] == foto_id

    r_canc = client.post(f"/api/fotos/{foto_id}/cancelar")
    assert r_canc.status_code == 200
    assert r_canc.json()["foto_id"] == foto_id

    r_reproc = client.post(f"/api/fotos/{foto_id}/reprocessar")
    assert r_reproc.status_code == 200
    assert r_reproc.json()["ok"] is True
    mock_proc.assert_called_once()

