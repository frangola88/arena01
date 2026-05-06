"""Rotas de vídeo: upload + status. POST dispara BackgroundTask."""
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from datetime import datetime
from pathlib import Path
import shutil
from core.database import get_db
from core.config import VIDEOS_ORIGINAIS_DIR, KEYFRAMES_DIR
from pipeline.video import processar_video

router = APIRouter()

EXTENSOES_VIDEO = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}


@router.post("/videos/ingerir")
async def ingerir_video(
    background_tasks: BackgroundTasks,
    localizacao_id: int = Form(...),
    arquivo: UploadFile = File(...),
):
    conn = get_db()
    try:
        loc = conn.execute("SELECT id FROM localizacoes WHERE id=?", (localizacao_id,)).fetchone()
        if not loc:
            raise HTTPException(status_code=404, detail="Localização não encontrada")

        ext = Path(arquivo.filename or "video.mp4").suffix.lower() or ".mp4"
        if ext not in EXTENSOES_VIDEO:
            raise HTTPException(status_code=400, detail=f"Extensão não suportada: {ext}")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        nome_arq = f"{ts}{ext}"
        caminho = VIDEOS_ORIGINAIS_DIR / nome_arq
        with open(caminho, "wb") as f:
            shutil.copyfileobj(arquivo.file, f)

        cursor = conn.execute(
            "INSERT INTO videos_processados (caminho, localizacao_id, status) VALUES (?,?,?)",
            (str(caminho), localizacao_id, "pendente"),
        )
        conn.commit()
        video_db_id = cursor.lastrowid
    finally:
        conn.close()

    background_tasks.add_task(processar_video, str(caminho), localizacao_id, video_db_id)
    return {"video_id": video_db_id, "status": "pendente", "caminho": str(caminho)}


@router.get("/videos/{video_id}/status")
def status_video(video_id: int):
    conn = get_db()
    try:
        row = conn.execute("""
            SELECT id, caminho, localizacao_id, status,
                   frames_extraidos, frames_processados, objetos_encontrados,
                   erro_mensagem, iniciado_em, concluido_em, criado_em
            FROM videos_processados WHERE id=?
        """, (video_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Vídeo não encontrado")
        info = dict(row)

        if info["status"] == "concluido":
            # Liga objetos por path do keyframe (KEYFRAMES_DIR/video_{id}/frame_*.jpg)
            prefixo = str(KEYFRAMES_DIR / f"video_{video_id}") + "%"
            objs = conn.execute("""
                SELECT id, nome, categoria_id, icone_path, icone_fonte, confianca
                FROM objetos
                WHERE foto_original_path LIKE ?
                ORDER BY id
            """, (prefixo,)).fetchall()
            info["objetos"] = [dict(o) for o in objs]
        return info
    finally:
        conn.close()
