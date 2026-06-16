"""Rotas de fotos: upload + status. POST dispara BackgroundTask."""
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from datetime import datetime
from pathlib import Path
import logging
import shutil
import json
import time
from PIL import Image, ImageOps
from core.database import get_db
from core.config import FOTOS_ORIGINAIS_DIR
from pipeline.ingestao import processar_foto

router = APIRouter()
_log = logging.getLogger("casaiq.api.fotos")


def _corrigir_orientacao_e_salvar(caminho_origem: Path) -> None:
    """
    Aplica rotação EXIF fisicamente ao arquivo da foto.

    IMPORTANTE: preserva o FORMATO do arquivo original baseado na extensão.
    Salvar JPEG em arquivo .png faz o Claude API rejeitar com erro 400
    ("media type mismatch"), o que quebra todo o pipeline de visão global.
    """
    try:
        img = Image.open(caminho_origem)
        img_corrigida = ImageOps.exif_transpose(img)
        if img_corrigida.mode != "RGB":
            img_corrigida = img_corrigida.convert("RGB")

        # Escolhe formato baseado na extensão real do arquivo
        ext = caminho_origem.suffix.lower()
        if ext == ".png":
            img_corrigida.save(caminho_origem, "PNG", optimize=True)
        elif ext == ".webp":
            img_corrigida.save(caminho_origem, "WEBP", quality=92)
        else:  # .jpg / .jpeg / outros
            img_corrigida.save(caminho_origem, "JPEG", quality=92, optimize=True)
        _log.info("orientacao_corrigida",
                  extra={"caminho": str(caminho_origem), "formato": ext})
    except Exception as e:
        _log.warning("erro_corrigir_orientacao",
                     extra={"caminho": str(caminho_origem), "erro": str(e)})


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

        # ─── CRÍTICO: Aplicar rotação EXIF ANTES de processar ───
        # Fotos de celular vêm com tag EXIF "Orientation" que indica 90/180/270°.
        # Sem aplicar, o VLM vê a foto girada → classifica errado
        # (mesa vista de lado = "régua de madeira").
        _corrigir_orientacao_e_salvar(caminho)

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


@router.get("/fotos/{foto_id}/debug")
def obter_debug_segmentacao(foto_id: int):
    """
    Retorna a imagem de debug da segmentação OpenCV (bordas + contornos coloridos).
    Útil para ver o que o sistema detectou e por que classificou errado.
    """
    from fastapi.responses import FileResponse

    conn = get_db()
    try:
        foto = conn.execute(
            "SELECT caminho FROM fotos_processadas WHERE id = ?", (foto_id,)
        ).fetchone()
        if not foto:
            raise HTTPException(status_code=404, detail="Foto não encontrada")

        caminho_foto = Path(foto["caminho"])
        debug_path = caminho_foto.with_name(caminho_foto.stem + "_debug.jpg")

        if not debug_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Imagem de debug ainda não foi gerada."
            )

        return FileResponse(str(debug_path), media_type="image/jpeg")
    finally:
        conn.close()


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


@router.get("/fotos/{foto_id}/progresso")
def obter_progresso_foto(foto_id: int):
    """Retorna status detalhado do processamento com progresso e ETA."""
    conn = get_db()
    try:
        foto = conn.execute(
            "SELECT * FROM fotos_processadas WHERE id = ?", (foto_id,)
        ).fetchone()
        
        if not foto:
            raise HTTPException(status_code=404, detail="Foto não encontrada")
        
        # Parsear campo progresso (JSON) — trata dados legados em texto plano
        try:
            progresso_json = json.loads(foto["progresso"] or "{}")
        except (json.JSONDecodeError, TypeError):
            # Dados legados em texto plano — retorna estrutura padrão
            progresso_json = {
                "etapa": "processando",
                "descricao": foto["progresso"] or "Processando...",
                "objetos_processados": foto["objetos_encontrados"] or 0,
                "objetos_totais": 1
            }
        
        # ─── CÁLCULO DO TEMPO (sem fuso horário) ───
        # Usamos `inicio_epoch` (time.time() = UTC em segundos) salvo pelo pipeline.
        # Comparado com time.time() atual → diferença correta independente de TZ.
        tempo_decorrido = 0.0
        tempo_estimado_total = None
        tempo_restante = None  # None = ainda não temos ETA confiável
        
        objetos_processados = progresso_json.get("objetos_processados", 0)
        objetos_totais = progresso_json.get("objetos_totais", 1)
        inicio_epoch = progresso_json.get("inicio_epoch")
        
        # Garantir consistência: objetos_totais nunca menor que processados
        if objetos_totais < objetos_processados:
            objetos_totais = max(objetos_processados, 1)
        
        if inicio_epoch:
            tempo_decorrido = max(0.0, time.time() - float(inicio_epoch))
            # Sanity check: clamp em 2h
            if tempo_decorrido > 7200:
                tempo_decorrido = 0
            
            # ETA confiável SÓ se já processou pelo menos 2 objetos
            # e os totais já foram determinados (não estamos mais segmentando)
            etapa = progresso_json.get("etapa", "")
            etapa_estavel = etapa not in ("iniciando", "segmentando", "")
            
            if etapa_estavel and objetos_processados >= 2 and objetos_totais > objetos_processados:
                tempo_por_objeto = tempo_decorrido / objetos_processados
                tempo_estimado_total = tempo_por_objeto * objetos_totais
                tempo_restante = max(0, tempo_estimado_total - tempo_decorrido)
                # Clamp: nunca prometer mais que 30 minutos
                if tempo_restante > 1800:
                    tempo_restante = None
        
        percentual = int((objetos_processados / objetos_totais) * 100) if objetos_totais > 0 else 0
        
        return {
            "id": foto["id"],
            "tipo": "foto",
            "nome_arquivo": Path(foto["caminho"]).name,
            "etapa_atual": progresso_json.get("etapa", "iniciando"),
            "descricao_etapa": progresso_json.get("descricao", "Processando..."),
            "percentual": percentual,
            "tempo_decorrido_s": tempo_decorrido,
            "tempo_estimado_total_s": tempo_estimado_total,
            "tempo_restante_s": tempo_restante,
            "objetos_processados": objetos_processados,
            "objetos_totais": objetos_totais,
            "status": foto["status"],
            "erro_mensagem": foto["erro_mensagem"]
        }
    finally:
        conn.close()


@router.post("/fotos/{foto_id}/cancelar")
def cancelar_processamento_foto(foto_id: int):
    """Cancela o processamento de uma foto em andamento."""
    conn = get_db()
    try:
        foto = conn.execute(
            "SELECT status FROM fotos_processadas WHERE id = ?", (foto_id,)
        ).fetchone()
        
        if not foto:
            raise HTTPException(status_code=404, detail="Foto não encontrada")
        
        if foto["status"] not in ["pendente", "processando"]:
            raise HTTPException(
                status_code=400,
                detail=f"Não pode cancelar foto com status '{foto['status']}'"
            )
        
        # Marcar como cancelado (erro)
        conn.execute(
            """UPDATE fotos_processadas
               SET status = ?, erro_mensagem = ?, concluido_em = CURRENT_TIMESTAMP
               WHERE id = ?""",
            ("erro", "Cancelado pelo usuário", foto_id)
        )
        conn.commit()
        
        return {"mensagem": "Processamento cancelado", "foto_id": foto_id}
    finally:
        conn.close()
