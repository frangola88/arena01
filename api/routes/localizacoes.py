"""Rotas de localizações: POST/GET/DELETE."""
from fastapi import APIRouter, HTTPException
from core.database import get_db
from api.schemas import LocalizacaoCreate

router = APIRouter()


@router.post("/localizacoes")
def criar_localizacao(loc: LocalizacaoCreate):
    conn = get_db()
    try:
        cursor = conn.execute(
            "INSERT INTO localizacoes (nome, tipo, comodo, descricao) VALUES (?,?,?,?)",
            (loc.nome, loc.tipo, loc.comodo, loc.descricao),
        )
        conn.commit()
        return {"id": cursor.lastrowid, "nome": loc.nome, "tipo": loc.tipo,
                "comodo": loc.comodo, "descricao": loc.descricao}
    finally:
        conn.close()


@router.get("/localizacoes")
def listar_localizacoes():
    conn = get_db()
    try:
        rows = conn.execute("""
            SELECT l.id, l.nome, l.tipo, l.comodo, l.descricao, l.criado_em,
                   COUNT(o.id) AS total_objetos
            FROM localizacoes l
            LEFT JOIN objetos o ON o.localizacao_id = l.id
            GROUP BY l.id
            ORDER BY l.nome
        """).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


@router.delete("/localizacoes/{loc_id}")
def deletar_localizacao(loc_id: int):
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM objetos WHERE localizacao_id=?", (loc_id,)
        ).fetchone()
        if row["n"] > 0:
            raise HTTPException(
                status_code=400,
                detail=f"Localização possui {row['n']} objeto(s). Mova ou remova-os antes."
            )
        cursor = conn.execute("DELETE FROM localizacoes WHERE id=?", (loc_id,))
        conn.commit()
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Localização não encontrada")
        return {"ok": True, "deletado_id": loc_id}
    finally:
        conn.close()
