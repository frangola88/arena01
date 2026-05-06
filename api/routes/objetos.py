"""Rotas de objetos: listagem com filtros + GET/PUT/DELETE individual."""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from core.database import get_db
from api.schemas import ObjetoUpdate

router = APIRouter()


@router.get("/objetos")
def listar_objetos(
    busca:        Optional[str] = Query(None),
    categoria:    Optional[int] = Query(None),
    localizacao:  Optional[int] = Query(None),
    estado:       Optional[str] = Query(None),
    limite:       int = Query(200, ge=1, le=1000),
):
    conn = get_db()
    try:
        sql = """
            SELECT o.*, c.nome AS categoria_nome, c.grupo AS categoria_grupo,
                   c.icone AS categoria_icone, l.nome AS localizacao_nome,
                   l.comodo AS localizacao_comodo
            FROM objetos o
            LEFT JOIN categorias  c ON o.categoria_id   = c.id
            LEFT JOIN localizacoes l ON o.localizacao_id = l.id
            WHERE 1=1
        """
        params: list = []
        if busca:
            sql += " AND (o.nome LIKE ? OR o.palavras_chave LIKE ?)"
            params.extend([f"%{busca}%", f"%|{busca.lower()}|%"])
        if categoria is not None:
            sql += " AND o.categoria_id = ?"
            params.append(categoria)
        if localizacao is not None:
            sql += " AND o.localizacao_id = ?"
            params.append(localizacao)
        if estado:
            sql += " AND o.estado = ?"
            params.append(estado)
        sql += " ORDER BY o.criado_em DESC LIMIT ?"
        params.append(limite)
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


@router.get("/objetos/{obj_id}")
def obter_objeto(obj_id: int):
    conn = get_db()
    try:
        row = conn.execute("""
            SELECT o.*, c.nome AS categoria_nome, c.grupo AS categoria_grupo,
                   c.icone AS categoria_icone, l.nome AS localizacao_nome,
                   l.comodo AS localizacao_comodo
            FROM objetos o
            LEFT JOIN categorias  c ON o.categoria_id   = c.id
            LEFT JOIN localizacoes l ON o.localizacao_id = l.id
            WHERE o.id = ?
        """, (obj_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Objeto não encontrado")
        return dict(row)
    finally:
        conn.close()


@router.put("/objetos/{obj_id}")
def atualizar_objeto(obj_id: int, dados: ObjetoUpdate):
    conn = get_db()
    try:
        existente = conn.execute("SELECT id FROM objetos WHERE id=?", (obj_id,)).fetchone()
        if not existente:
            raise HTTPException(status_code=404, detail="Objeto não encontrado")

        campos = {k: v for k, v in dados.model_dump(exclude_unset=True).items() if v is not None}
        if not campos:
            return {"ok": True, "atualizados": 0}

        set_clause = ", ".join(f"{k}=?" for k in campos)
        params = list(campos.values()) + [obj_id]
        # marca como revisado pelo usuário
        sql = f"UPDATE objetos SET {set_clause}, revisado_pelo_usuario=1 WHERE id=?"
        conn.execute(sql, params)
        conn.commit()
        return {"ok": True, "atualizados": len(campos)}
    finally:
        conn.close()


@router.delete("/objetos/{obj_id}")
def deletar_objeto(obj_id: int):
    conn = get_db()
    try:
        cursor = conn.execute("DELETE FROM objetos WHERE id=?", (obj_id,))
        conn.commit()
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Objeto não encontrado")
        return {"ok": True, "deletado_id": obj_id}
    finally:
        conn.close()
