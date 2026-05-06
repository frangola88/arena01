"""
Pipeline de Ingestão de Vídeo:

  1. Extrai keyframes via ffmpeg (CASAIQ_VIDEO_FPS, max CASAIQ_VIDEO_MAX_FRAMES).
  2. Para cada keyframe:
       a. Agente 1 (segmentador) -> lista de recortes
       b. Agente 2 (analisador)  -> análise visual por objeto
       c. Agente 3 (enriquecedor)-> metadados normalizados
  3. Deduplica objetos por nome normalizado (maior confiança vence).
  4. Para cada objeto único: INSERT + Agente 4 (ícone).

Thread safety: nova conexão SQLite dentro da função (BackgroundTask = thread).
"""
import re
import subprocess
import traceback
from datetime import datetime
from pathlib import Path

from core.config import (KEYFRAMES_DIR, CASAIQ_VIDEO_FPS,
                         CASAIQ_VIDEO_MAX_FRAMES)
from core.database import get_db
from agents.agent_1_segmentador import segmentar_foto
from agents.agent_2_analisador   import analisar_objeto
from agents.agent_3_enriquecedor import enriquecer_objeto
from agents.agent_4_icone        import gerar_icone


def _extrair_keyframes(caminho_video: str, video_id: int) -> list[str]:
    """Extrai keyframes via ffmpeg. Retorna lista ordenada de caminhos JPG."""
    saida_dir = KEYFRAMES_DIR / f"video_{video_id}"
    saida_dir.mkdir(parents=True, exist_ok=True)
    padrao = saida_dir / "frame_%03d.jpg"

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", caminho_video,
        "-vf", f"fps={CASAIQ_VIDEO_FPS},scale='min(1024,iw)':-2",
        "-frames:v", str(CASAIQ_VIDEO_MAX_FRAMES),
        "-q:v", "3",
        str(padrao),
    ]
    print(f"[Vídeo] ffmpeg fps={CASAIQ_VIDEO_FPS} max={CASAIQ_VIDEO_MAX_FRAMES}")
    subprocess.run(cmd, check=True, timeout=300)
    return sorted(str(p) for p in saida_dir.glob("frame_*.jpg"))


def _normalizar_nome(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def processar_video(caminho_video: str, localizacao_id: int, video_db_id: int) -> None:
    conn = get_db()
    try:
        conn.execute(
            "UPDATE videos_processados SET status='processando', iniciado_em=? WHERE id=?",
            (datetime.now(), video_db_id)
        )
        conn.commit()

        keyframes = _extrair_keyframes(caminho_video, video_db_id)
        if not keyframes:
            raise RuntimeError("Nenhum keyframe extraído (vídeo vazio ou corrompido)")
        conn.execute(
            "UPDATE videos_processados SET frames_extraidos=? WHERE id=?",
            (len(keyframes), video_db_id)
        )
        conn.commit()
        print(f"[Vídeo] {len(keyframes)} keyframe(s) extraído(s)")

        rows = conn.execute("SELECT nome, grupo, icone FROM categorias").fetchall()
        categorias_nomes  = [r["nome"] for r in rows]
        categorias_lookup = {r["nome"]: {"grupo": r["grupo"], "icone": r["icone"]} for r in rows}

        # Detecções acumuladas por nome normalizado (mantém maior confiança)
        deteccoes: dict[str, dict] = {}
        for idx, frame_path in enumerate(keyframes):
            print(f"[Vídeo] Frame {idx + 1}/{len(keyframes)}")
            try:
                synthetic_id = 100000 + video_db_id * 100 + idx
                objetos = segmentar_foto(frame_path, synthetic_id)
                for obj in objetos:
                    try:
                        analise     = analisar_objeto(obj["recorte_path"], obj["nome"])
                        enriquecido = enriquecer_objeto(analise, categorias_nomes)
                        chave = _normalizar_nome(enriquecido.get("nome") or obj["nome"])
                        if not chave:
                            continue
                        enriquecido["_recorte_path"] = obj["recorte_path"]
                        enriquecido["_frame_origem"] = frame_path
                        nova_conf = float(enriquecido.get("confianca") or 0.0)
                        anterior  = deteccoes.get(chave)
                        if anterior is None or nova_conf > float(anterior.get("confianca") or 0.0):
                            deteccoes[chave] = enriquecido
                    except Exception as e:
                        print(f"[Vídeo] erro objeto '{obj.get('nome')}': {e}")
            except Exception as e:
                print(f"[Vídeo] erro frame {idx + 1}: {e}")
                traceback.print_exc()
            conn.execute(
                "UPDATE videos_processados SET frames_processados=? WHERE id=?",
                (idx + 1, video_db_id)
            )
            conn.commit()

        inseridos = 0
        for chave, enr in deteccoes.items():
            try:
                cat_nome = enr.get("categoria_nome", "Outros")
                cat_row  = conn.execute("SELECT id FROM categorias WHERE nome=?", (cat_nome,)).fetchone()
                cat_id   = cat_row["id"] if cat_row else None
                cat_info = categorias_lookup.get(cat_nome, {"grupo": "Geral", "icone": "\U0001F4E6"})

                cursor = conn.execute("""
                    INSERT INTO objetos (
                        nome, descricao, categoria_id, localizacao_id,
                        cor, tamanho, tamanho_estimado_cm, peso_estimado_g,
                        material, estado, funcao, palavras_chave,
                        foto_original_path, recorte_path, icone_path,
                        icone_fonte, confianca, modelo_visao
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    enr.get("nome", chave),
                    enr.get("descricao", ""),
                    cat_id, localizacao_id,
                    enr.get("cor", ""),
                    enr.get("tamanho", ""),
                    enr.get("tamanho_estimado_cm", ""),
                    enr.get("peso_estimado_g"),
                    enr.get("material", ""),
                    enr.get("estado", "bom"),
                    enr.get("funcao", ""),
                    enr.get("palavras_chave", ""),
                    enr.get("_frame_origem", caminho_video),
                    enr.get("_recorte_path", ""),
                    "", "",
                    enr.get("confianca", 0.0),
                    enr.get("_modelo", ""),
                ))
                conn.commit()
                objeto_id = cursor.lastrowid

                icone_path, icone_fonte = gerar_icone(
                    recorte_path   = enr.get("_recorte_path", ""),
                    nome           = enr.get("nome", chave),
                    categoria_nome = cat_nome,
                    icone_emoji    = cat_info["icone"],
                    grupo          = cat_info["grupo"],
                    confianca      = enr.get("confianca", 0.0),
                    objeto_id      = objeto_id,
                )
                conn.execute(
                    "UPDATE objetos SET icone_path=?, icone_fonte=? WHERE id=?",
                    (icone_path, icone_fonte, objeto_id)
                )
                conn.commit()
                inseridos += 1
            except Exception as e:
                print(f"[Vídeo] erro inserindo '{chave}': {e}")
                traceback.print_exc()

        conn.execute("""
            UPDATE videos_processados
            SET status='concluido', objetos_encontrados=?, concluido_em=?
            WHERE id=?
        """, (inseridos, datetime.now(), video_db_id))
        conn.commit()
        print(f"[Vídeo] Concluído: {inseridos} objeto(s) único(s) "
              f"de {len(deteccoes)} detecção(ões) em {len(keyframes)} frame(s)")

    except Exception as e:
        print(f"[Vídeo] ERRO CRÍTICO: {e}")
        traceback.print_exc()
        try:
            conn.execute(
                "UPDATE videos_processados SET status='erro', erro_mensagem=? WHERE id=?",
                (str(e), video_db_id)
            )
            conn.commit()
        except Exception:
            pass
    finally:
        conn.close()
