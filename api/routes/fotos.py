"""Rotas de fotos: upload + status. POST dispara BackgroundTask."""
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from datetime import datetime
from pathlib import Path
import shutil
from core.database import get_db
from core.config import FOTOS_ORIGINAIS_DIR
from pipeline.ingestao import processar_foto

router = APIRouter()


@router.post("/fotos/ingerir")
async def ingerir_foto(
    background_tasks: BackgroundTasks,
    localizacao_id: int = Form(...),
    arquivo: UploadFile = File(...),
):
    # Validar localização
    conn = get_db()
    try:
        loc = conn.execute("SELECT id FROM localizacoes WHERE id=?", (localizacao_id,)).fetchone()
        if not loc:
            raise HTTPException(status_code=404, detail="Localização não encontrada")

        # Salvar arquivo com nome único
        ext = Path(arquivo.filename or "foto.jpg").suffix.lower() or ".jpg"
        if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
            raise HTTPException(status_code=400, detail=f"Extensão não suportada: {ext}")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        nome_arq = f"{ts}{ext}"
        caminho = FOTOS_ORIGINAIS_DIR / nome_arq
        with open(caminho, "wb") as f:
            shutil.copyfileobj(arquivo.file, f)

        # Registrar foto no banco
        cursor = conn.execute(
            "INSERT INTO fotos_processadas (caminho, localizacao_id, status) VALUES (?,?,?)",
            (str(caminho), localizacao_id, "pendente"),
        )
        conn.commit()
        foto_db_id = cursor.lastrowid
    finally:
        conn.close()

    # Disparar processamento em background (thread separada)
    background_tasks.add_task(processar_foto, str(caminho), localizacao_id, foto_db_id)
    return {"foto_id": foto_db_id, "status": "pendente", "caminho": str(caminho)}


@router.get("/fotos/{foto_id}/status")
def status_foto(foto_id: int):
    conn = get_db()
    try:
        row = conn.execute("""
            SELECT id, caminho, localizacao_id, status, objetos_encontrados,
                   erro_mensagem, iniciado_em, concluido_em, criado_em
            FROM fotos_processadas WHERE id=?
        """, (foto_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Foto não encontrada")
        info = dict(row)

        # Se concluído, anexa lista resumida de objetos
        if info["status"] == "concluido":
            objs = conn.execute("""
                SELECT id, nome, categoria_id, icone_path, icone_fonte, confianca
                FROM objetos WHERE foto_original_path=?
                ORDER BY id
            """, (info["caminho"],)).fetchall()
            info["objetos"] = [dict(o) for o in objs]
        return info
    finally:
        conn.close()
