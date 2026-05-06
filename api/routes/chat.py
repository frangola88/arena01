"""Rotas de chat: POST /chat e GET /chat/historico."""
from fastapi import APIRouter
from core.database import get_db
from agents.assistente import chat as assistente_chat
from api.schemas import ChatRequest

router = APIRouter()


@router.post("/chat")
def chat(req: ChatRequest):
    conn = get_db()
    try:
        return assistente_chat(req.pergunta, conn)
    finally:
        conn.close()


@router.get("/chat/historico")
def historico():
    conn = get_db()
    try:
        rows = conn.execute("""
            SELECT id, pergunta, resposta, modelo, criado_em
            FROM historico_chat
            ORDER BY id DESC
            LIMIT 20
        """).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
