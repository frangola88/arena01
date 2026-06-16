"""
Rotas administrativas: reset do banco, exportação, manutenção.

⚠️ Endpoints destrutivos exigem `confirmacao: "ZERAR"` no body para evitar
clique acidental.
"""
import logging
import shutil
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from core.database import get_db, init_db
from core.config import DB_PATH, FOTOS_ORIGINAIS_DIR, RECORTES_DIR

_log = logging.getLogger("casaiq.admin")
router = APIRouter()


class ResetBody(BaseModel):
    confirmacao: str  # Deve ser exatamente "ZERAR"


def _limpar_dir(caminho: Path) -> int:
    """Limpa arquivos de um diretório, mantendo o diretório."""
    if not caminho.exists():
        return 0
    count = 0
    for arq in caminho.iterdir():
        try:
            if arq.is_file():
                arq.unlink()
                count += 1
            elif arq.is_dir():
                shutil.rmtree(arq)
                count += 1
        except Exception as e:
            _log.warning("erro_limpar_item", extra={"item": str(arq), "erro": str(e)})
    return count


@router.get("/admin/estatisticas")
def estatisticas_banco():
    """Retorna contagens — útil para o frontend mostrar o que será apagado."""
    conn = get_db()
    try:
        n_obj = conn.execute("SELECT COUNT(*) AS c FROM objetos").fetchone()["c"]
        n_fotos = conn.execute("SELECT COUNT(*) AS c FROM fotos_processadas").fetchone()["c"]
        n_loc = conn.execute("SELECT COUNT(*) AS c FROM localizacoes").fetchone()["c"]
        n_cat = conn.execute("SELECT COUNT(*) AS c FROM categorias").fetchone()["c"]
        return {
            "objetos": n_obj,
            "fotos_processadas": n_fotos,
            "localizacoes": n_loc,
            "categorias": n_cat,
        }
    finally:
        conn.close()


@router.post("/admin/resetar-banco")
def resetar_banco(body: ResetBody):
    """
    Zera completamente o banco e arquivos relacionados.
    
    Requer body: {"confirmacao": "ZERAR"}
    
    Apaga:
    - Banco SQLite (recriado vazio)
    - Fotos originais, recortes e ícones
    
    Mantém:
    - Categorias (seed inicial)
    - Estrutura de diretórios
    """
    if body.confirmacao != "ZERAR":
        raise HTTPException(
            status_code=400,
            detail='Confirmação inválida. Body deve ter {"confirmacao": "ZERAR"}.'
        )
    
    _log.warning("reset_banco_iniciado")
    
    # 1. Remover banco SQLite (e arquivos WAL/SHM)
    arquivos_db_removidos = []
    if DB_PATH.exists():
        DB_PATH.unlink()
        arquivos_db_removidos.append(DB_PATH.name)
    for sufixo in ["-wal", "-shm"]:
        aux = Path(str(DB_PATH) + sufixo)
        if aux.exists():
            aux.unlink()
            arquivos_db_removidos.append(aux.name)
    
    # 2. Limpar diretórios
    n_fotos = _limpar_dir(FOTOS_ORIGINAIS_DIR)
    n_recortes = _limpar_dir(RECORTES_DIR)
    
    icones_dir = DB_PATH.parent / "storage" / "icones"
    if not icones_dir.exists():
        icones_dir = Path.cwd() / "storage" / "icones"
    n_icones = _limpar_dir(icones_dir)
    
    # 3. Recriar estrutura
    try:
        init_db()
    except Exception as e:
        _log.error("erro_recriar_banco", extra={"erro": str(e)})
        raise HTTPException(status_code=500, detail=f"Erro ao recriar banco: {e}")
    
    _log.warning("reset_banco_concluido", extra={
        "fotos_removidas": n_fotos,
        "recortes_removidos": n_recortes,
        "icones_removidos": n_icones,
    })
    
    return {
        "ok": True,
        "mensagem": "Banco zerado com sucesso. Recarregue a página.",
        "removido": {
            "arquivos_banco": arquivos_db_removidos,
            "fotos_originais": n_fotos,
            "recortes": n_recortes,
            "icones": n_icones,
        },
    }
