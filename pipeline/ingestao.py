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
import json
import time
from datetime import datetime
from core.database import get_db
from agents.agent_1_segmentador import segmentar_foto
from agents.agent_2_analisador   import analisar_objeto
from agents.agent_3_enriquecedor import enriquecer_objeto
from agents.agent_4_icone        import gerar_icone
from core.deduplicacao import buscar_objeto_similar, extrair_nucleo_nome, limpar_nome_objeto
from core.estimador_peso import estimar_peso
from core.visao_global import obter_dados_objeto_por_nome

# Limiar mínimo de confiança para REUTILIZAR a análise rica da skill
# (e pular o Agent 2). Abaixo disso, o Agent 2 roda como antes.
LIMIAR_CONFIANCA_SKILL = 0.70

_log = logging.getLogger("casaiq.pipeline.foto")


def processar_foto(caminho_foto: str, localizacao_id: int, foto_db_id: int) -> None:
    conn = get_db()   # nova conexão nesta thread
    inseridos = 0
    
    # Marca o início em EPOCH (segundos UTC desde 1970) — sem fuso, sempre consistente
    inicio_epoch = time.time()

    def progresso(etapa: str, descricao: str = "", objetos_processados: int = 0, objetos_totais: int = 1) -> None:
        try:
            progress_data = {
                "etapa": etapa,
                "descricao": descricao,
                "objetos_processados": objetos_processados,
                "objetos_totais": objetos_totais,
                "inicio_epoch": inicio_epoch,  # ⏱️ Tempo de referência absoluto
            }
            conn.execute("UPDATE fotos_processadas SET progresso=? WHERE id=?", (json.dumps(progress_data), foto_db_id))
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

        progresso("segmentando", "Segmentando objetos da foto…", 0, 1)
        objetos_detectados = segmentar_foto(caminho_foto, foto_db_id)
        total = len(objetos_detectados)
        _log.info("objetos_detectados", extra={"total": total, "foto_id": foto_db_id})
        progresso("segmentando", f"{total} objeto(s) detectado(s)", total, total)

        for idx, obj in enumerate(objetos_detectados, start=1):
            try:
                nome_sugerido = obj["nome"]
                recorte_path  = obj["recorte_path"]

                # ─── FAST PATH: tentar reutilizar dados ricos da skill ───
                # Se a skill (visão global) já analisou esse nome com confiança >= 0.70,
                # pulamos o Agent 2 (que faria chamada Claude redundante).
                dados_skill = obter_dados_objeto_por_nome(caminho_foto, nome_sugerido)

                if dados_skill and dados_skill.get("confianca", 0) >= LIMIAR_CONFIANCA_SKILL:
                    progresso("analisando", f"Reaproveitando análise rica {idx}/{total}",
                              idx - 1, total)
                    # Monta `analise` no formato esperado pelo Agent 3
                    analise = {
                        "nome": dados_skill["nome"],
                        "descricao": dados_skill.get("descricao", ""),
                        "cor": dados_skill.get("cor", ""),
                        "tamanho": dados_skill.get("tamanho", ""),
                        "tamanho_estimado_cm": dados_skill.get("tamanho_estimado_cm", ""),
                        "peso_estimado_g": dados_skill.get("peso_estimado_g"),
                        "material": dados_skill.get("material", ""),
                        "estado": dados_skill.get("estado", "bom"),
                        "funcao": dados_skill.get("funcao", ""),
                        "palavras_chave": dados_skill.get("palavras_chave", []),
                        "confianca": dados_skill.get("confianca", 0.9),
                        "_modelo": "claude_skill_inventario",
                    }
                    _log.info("agent2_pulado_usando_skill", extra={
                        "nome": nome_sugerido,
                        "confianca": dados_skill["confianca"],
                        "foto_id": foto_db_id,
                    })
                else:
                    # SLOW PATH: pede análise por recorte ao Agent 2 (fallback)
                    progresso("analisando", f"Analisando objeto {idx}/{total}",
                              idx - 1, total)
                    analise = analisar_objeto(recorte_path, nome_sugerido)

                progresso("enriquecendo", f"Enriquecendo objeto {idx}/{total}", idx, total)
                enriquecido = enriquecer_objeto(analise, categorias_nomes)

                # Se a skill ja deu uma categoria valida, usar diretamente
                if dados_skill and dados_skill.get("categoria_sugerida"):
                    enriquecido["categoria_nome"] = dados_skill["categoria_sugerida"]

                # ─── LIMPAR NOME: remove cores/modificadores e corrige mojibake ───
                # Ex: "chave de fenda phillips com cabo vermelho" → "chave de fenda"
                # Ex: "rãgua" (mojibake) → "régua"
                nome_original = enriquecido.get("nome", nome_sugerido)
                nome_limpo = limpar_nome_objeto(nome_original)
                if nome_limpo != nome_original:
                    _log.info("nome_limpo", extra={
                        "original": nome_original, "limpo": nome_limpo, "foto_id": foto_db_id,
                    })
                enriquecido["nome"] = nome_limpo

                cat_nome = enriquecido.get("categoria_nome", "Outros")
                cat_row  = conn.execute("SELECT id FROM categorias WHERE nome=?", (cat_nome,)).fetchone()
                cat_id   = cat_row["id"] if cat_row else None
                cat_info = categorias_lookup.get(cat_nome, {"grupo": "Geral", "icone": "\U0001F4E6"})

                # ─── DEDUPLICAÇÃO: Verificar se objeto similar já existe ───
                # Limiar 0.70 + comparação por NÚCLEO (sem cor/tamanho/modificadores)
                obj_similar = buscar_objeto_similar(
                    {
                        "nome": enriquecido.get("nome", nome_sugerido),
                        "tamanho": enriquecido.get("tamanho", ""),
                        "categoria_id": cat_id,
                        "localizacao_id": localizacao_id,
                    },
                    localizacao_id=localizacao_id,
                    categoria_id=cat_id,
                    limiar_similaridade=0.70,
                    excluir_foto_path=caminho_foto,  # objetos da mesma foto nunca são dupes
                )
                
                if obj_similar:
                    _log.info("objeto_duplicado_pulado", extra={
                        "novo": enriquecido.get("nome"),
                        "existente": obj_similar.get("nome"),
                        "obj_id": obj_similar.get("id"),
                        "foto_id": foto_db_id,
                    })
                    continue  # Pula este objeto — não insere duplicado
                
                # ─── ESTIMATIVA DE PESO: Calcular aproximadamente ───
                peso_estimado = estimar_peso(
                    categoria_nome=cat_nome,
                    tamanho=enriquecido.get("tamanho", ""),
                    material=enriquecido.get("material", ""),
                    tamanho_estimado_cm=enriquecido.get("tamanho_estimado_cm", "")
                )
                enriquecido["peso_estimado_g"] = peso_estimado

                import json as _json
                cores_json = _json.dumps(
                    enriquecido.get("cores_dominantes", []),
                    ensure_ascii=False
                )
                # geometria_vetor: GeoJSON Feature do polígono casado ao objeto
                # (vem do agent_1 via casar_poligono_a_bbox); "" se não casou.
                geometria_vetor = obj.get("geometria_vetor", "")
                cursor = conn.execute("""
                    INSERT INTO objetos (
                        nome, descricao, categoria_id, localizacao_id,
                        cor, tamanho, tamanho_estimado_cm, peso_estimado_g,
                        material, estado, funcao, palavras_chave,
                        foto_original_path, recorte_path, icone_path,
                        icone_fonte, confianca, modelo_visao, cores_json,
                        geometria_vetor
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
                    cores_json,
                    geometria_vetor,
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
                progresso("enriquecendo", f"Catalogado {idx}/{total}: {enriquecido.get('nome', nome_sugerido)}", idx, total)

            except Exception as e:
                _log.warning("erro_objeto", extra={
                    "nome": obj.get("nome"), "erro": str(e), "foto_id": foto_db_id,
                }, exc_info=True)

        progresso("finalizando", "Finalizando processamento", total, total)
        conn.execute("""
            UPDATE fotos_processadas
            SET status='concluido', objetos_encontrados=?, concluido_em=?
            WHERE id=?
        """, (inseridos, datetime.now(), foto_db_id))
        conn.commit()
        progresso("concluído", f"Processamento concluído! {inseridos} objeto(s)", total, total)
        _log.info("foto_concluida", extra={"inseridos": inseridos, "foto_id": foto_db_id})

    except Exception as e:
        _log.error("erro_critico", extra={
            "erro": str(e), "foto_id": foto_db_id,
        }, exc_info=True)
        progresso("erro", str(e), 0, 1)
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
