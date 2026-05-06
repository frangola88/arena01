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
