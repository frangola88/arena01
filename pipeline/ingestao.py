"""
Pipeline de Ingestão — ordem correta de operações:

  Para cada objeto:
    1. Agente 2  -> analise visual (local; Claude se confiança baixa)
    2. Agente 3  -> enriquecimento (sempre local)
    3. INSERT    -> objeto_id = cursor.lastrowid  <- ID real antes de chamar Agente 4
    4. Agente 4  -> gerar_icone(objeto_id=ID_real)
    5. UPDATE    -> icone_path + icone_fonte no banco

Thread safety: cria conexão SQLite DENTRO desta função (BackgroundTask = thread separada).
"""
import logging
from datetime import datetime
from core.database import get_db
from agents.agent_1_segmentador import segmentar_foto
from agents.agent_2_analisador   import analisar_objeto
from agents.agent_3_enriquecedor import enriquecer_objeto
from agents.agent_4_icone        import gerar_icone

_log = logging.getLogger("casaiq.pipeline.foto")


def processar_foto(caminho_foto: str, localizacao_id: int, foto_db_id: int) -> None:
    conn = get_db()   # nova conexão nesta thread
    inseridos = 0

    def progresso(msg: str) -> None:
        try:
            conn.execute("UPDATE fotos_processadas SET progresso=? WHERE id=?", (msg, foto_db_id))
            conn.commit()
        except Exception:
            pass

    try:
        conn.execute(
            "UPDATE fotos_processadas SET status='processando', iniciado_em=? WHERE id=?",
            (datetime.now(), foto_db_id)
        )
        conn.commit()

        # Buscar categorias do banco ANTES do loop (P5 corrigido)
        rows = conn.execute("SELECT nome, grupo, icone FROM categorias").fetchall()
        categorias_nomes  = [r["nome"] for r in rows]
        categorias_lookup = {r["nome"]: {"grupo": r["grupo"], "icone": r["icone"]} for r in rows}

        progresso("Agente 1: segmentando objetos da foto…")
        objetos_detectados = segmentar_foto(caminho_foto, foto_db_id)
        total = len(objetos_detectados)
        _log.info("objetos_detectados", extra={"total": total, "foto_id": foto_db_id})
        progresso(f"Agente 1: {total} objeto(s) detectado(s)")

        for idx, obj in enumerate(objetos_detectados, start=1):
            try:
                nome_sugerido = obj["nome"]
                recorte_path  = obj["recorte_path"]

                analise     = analisar_objeto(recorte_path, nome_sugerido)
                enriquecido = enriquecer_objeto(analise, categorias_nomes)

                cat_nome = enriquecido.get("categoria_nome", "Outros")
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
                    enriquecido.get("nome", nome_sugerido),
                    enriquecido.get("descricao", ""),
                    cat_id, localizacao_id,
                    enriquecido.get("cor", ""),
                    enriquecido.get("tamanho", ""),
                    enriquecido.get("tamanho_estimado_cm", ""),
                    enriquecido.get("peso_estimado_g"),
                    enriquecido.get("material", ""),
                    enriquecido.get("estado", "bom"),
                    enriquecido.get("funcao", ""),
                    enriquecido.get("palavras_chave", ""),
                    caminho_foto, recorte_path,
                    "", "",   # icone_path e icone_fonte preenchidos após Agente 4
                    enriquecido.get("confianca", 0.0),
                    enriquecido.get("_modelo", ""),
                ))
                conn.commit()
                objeto_id = cursor.lastrowid   # ID real — INSERT já feito

                icone_path, icone_fonte = gerar_icone(
                    recorte_path   = recorte_path,
                    nome           = enriquecido.get("nome", nome_sugerido),
                    categoria_nome = cat_nome,
                    icone_emoji    = cat_info["icone"],
                    grupo          = cat_info["grupo"],
                    confianca      = enriquecido.get("confianca", 0.0),
                    objeto_id      = objeto_id,
                )
                conn.execute(
                    "UPDATE objetos SET icone_path=?, icone_fonte=? WHERE id=?",
                    (icone_path, icone_fonte, objeto_id)
                )
                conn.commit()
                inseridos += 1

            except Exception as e:
                _log.warning("erro_objeto", extra={
                    "nome": obj.get("nome"), "erro": str(e), "foto_id": foto_db_id,
                }, exc_info=True)

        conn.execute("""
            UPDATE fotos_processadas
            SET status='concluido', objetos_encontrados=?, concluido_em=?
            WHERE id=?
        """, (inseridos, datetime.now(), foto_db_id))
        conn.commit()
        _log.info("foto_concluida", extra={"inseridos": inseridos, "foto_id": foto_db_id})

    except Exception as e:
        _log.error("erro_critico", extra={
            "erro": str(e), "foto_id": foto_db_id,
        }, exc_info=True)
        try:
            conn.execute(
                "UPDATE fotos_processadas SET status='erro', erro_mensagem=? WHERE id=?",
                (str(e), foto_db_id)
            )
            conn.commit()
        except Exception:
            pass
    finally:
        conn.close()
