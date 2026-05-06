"""Estatísticas agregadas + modo ativo."""
from fastapi import APIRouter
from core.database import get_db
from core.roteador import descricao_modo, _modo, _tem_api

router = APIRouter()


@router.get("/estatisticas")
def estatisticas():
    conn = get_db()
    try:
        total = conn.execute("SELECT COUNT(*) AS n FROM objetos").fetchone()["n"]
        total_localizacoes = conn.execute("SELECT COUNT(*) AS n FROM localizacoes").fetchone()["n"]
        total_fotos = conn.execute("SELECT COUNT(*) AS n FROM fotos_processadas").fetchone()["n"]

        por_categoria = conn.execute("""
            SELECT c.nome AS categoria, c.icone AS icone, c.grupo AS grupo,
                   COUNT(o.id) AS total
            FROM categorias c
            LEFT JOIN objetos o ON o.categoria_id = c.id
            GROUP BY c.id
            HAVING total > 0
            ORDER BY total DESC
            LIMIT 10
        """).fetchall()

        por_comodo = conn.execute("""
            SELECT COALESCE(NULLIF(l.comodo, ''), 'Sem cômodo') AS comodo,
                   COUNT(o.id) AS total
            FROM localizacoes l
            LEFT JOIN objetos o ON o.localizacao_id = l.id
            GROUP BY comodo
            HAVING total > 0
            ORDER BY total DESC
        """).fetchall()

        return {
            "total_objetos": total,
            "total_localizacoes": total_localizacoes,
            "total_fotos": total_fotos,
            "por_categoria": [dict(r) for r in por_categoria],
            "por_comodo": [dict(r) for r in por_comodo],
            "modo_ativo": {
                "modo": str(_modo()),
                "descricao": descricao_modo(),
                "tem_api_key": _tem_api(),
            },
        }
    finally:
        conn.close()
