#!/usr/bin/env python3
"""
Pipeline Completo CasaIQ — Etapas 1-4 em Todas as Fotos Mestras

Fluxo:
  Foto Mestre → Agent 1 (segmentação) → Agent 2 (análise) → Agent 3 (enriquecimento)
             → Agent 4 (ícones) → SQLite

Uso:
  cd ~/projetos/casaiq
  conda run -n casaiq python run_full_pipeline_all_fotos.py
"""
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Config
BASE = Path.home() / "projetos" / "casaiq"
FOTOS_DIR = BASE / "storage" / "fotos_originais"
REPORTS_DIR = BASE / "pipeline_reports"
REPORTS_DIR.mkdir(exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("pipeline_full")

# Imports locais
sys.path.insert(0, str(BASE))
from core.database import get_db
from agents.agent_1_segmentador import segmentar_foto
from agents.agent_2_analisador import analisar_objeto
from agents.agent_3_enriquecedor import enriquecer_objeto
from agents.agent_4_icone import gerar_icone


def executar_pipeline_completo(caminho_foto: str, foto_id: int) -> dict:
    """
    Executa o pipeline completo (agentes 1-4) para uma foto.

    Retorna dict com resultado ou erro.
    """
    foto_nome = Path(caminho_foto).name
    resultado = {
        "foto_id": foto_id,
        "arquivo": foto_nome,
        "timestamp": datetime.now().isoformat(),
        "etapas": {},
        "itens_criados": 0,
        "erros": [],
    }

    try:
        # ─── AGENT 1: Segmentação ───────────────────────────────────────────
        log.info("Agent 1 (segmentação): iniciando para %s", foto_nome)
        t0 = time.time()
        try:
            objetos = segmentar_foto(caminho_foto, foto_id=foto_id)
            tempo_a1 = time.time() - t0
            resultado["etapas"]["agent_1"] = {
                "status": "ok",
                "n_objetos": len(objetos),
                "tempo_s": tempo_a1,
            }
            log.info("Agent 1 OK: %d objetos em %.1fs", len(objetos), tempo_a1)
        except Exception as e:
            log.exception("Agent 1 falhou")
            resultado["etapas"]["agent_1"] = {"status": "erro", "erro": str(e)}
            resultado["erros"].append(f"Agent 1: {str(e)}")
            return resultado

        if not objetos:
            log.warning("Agent 1: nenhum objeto detectado")
            resultado["itens_criados"] = 0
            return resultado

        # ─── AGENT 2: Análise ───────────────────────────────────────────────
        db = get_db()
        n_items_criados = 0

        for idx, obj in enumerate(objetos, start=1):
            try:
                log.info("Agent 2 (análise): objeto %d de %d", idx, len(objetos))
                t0 = time.time()
                analise = analisar_objeto(obj.get("recorte_path"), obj.get("nome"))
                tempo_a2 = time.time() - t0

                # ─── AGENT 3: Enriquecimento ─────────────────────────────
                log.info("Agent 3 (enriquecimento): objeto %d", idx)
                t0 = time.time()
                enriquecido = enriquecer_objeto(analise, [])
                tempo_a3 = time.time() - t0

                # ─── AGENT 4: Ícone ──────────────────────────────────────
                log.info("Agent 4 (ícone): objeto %d", idx)
                t0 = time.time()
                caminho_icone = gerar_icone(
                    obj.get("recorte_path"),
                    obj.get("nome"),
                    enriquecido.get("categoria", "outro")
                )
                tempo_a4 = time.time() - t0

                # ─── Salvar no banco ──────────────────────────────────────
                item = {
                    "foto_id": foto_id,
                    "nome": enriquecido.get("nome"),
                    "categoria": enriquecido.get("categoria"),
                    "preco_estimado": enriquecido.get("preco_estimado"),
                    "condicao": enriquecido.get("condicao"),
                    "descricao": enriquecido.get("descricao"),
                    "icone_path": str(caminho_icone) if caminho_icone else None,
                    "recorte_path": obj.get("recorte_path"),
                }

                sql = """
                    INSERT INTO inventario_itens (
                        foto_id, nome, categoria, preco_estimado,
                        condicao, descricao, icone_path, recorte_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """
                db.execute(sql, tuple(item.values()))
                db.commit()
                n_items_criados += 1

                log.info("Item %d ✓ (A2: %.1fs, A3: %.1fs, A4: %.1fs)",
                        idx, tempo_a2, tempo_a3, tempo_a4)

            except Exception as e:
                log.exception("Erro processando objeto %d", idx)
                resultado["erros"].append(f"Objeto {idx}: {str(e)}")
                continue

        resultado["itens_criados"] = n_items_criados
        resultado["etapas"]["agentes_2_3_4"] = {
            "status": "ok",
            "n_items_criados": n_items_criados,
        }

    except Exception as e:
        log.exception("Pipeline falhou para %s", foto_nome)
        resultado["erros"].append(str(e))

    return resultado


def main():
    log.info("=" * 80)
    log.info("Pipeline Completo — Iniciando para todas as fotos mestras")
    log.info("=" * 80)

    # Seleciona fotos (ignora _debug)
    todas_fotos = sorted([
        f for f in FOTOS_DIR.glob("*.jpg")
        if "_debug" not in f.name
    ])

    if not todas_fotos:
        log.error("Nenhuma foto encontrada em %s", FOTOS_DIR)
        return

    log.info("Fotos encontradas: %d", len(todas_fotos))

    resultados = {
        "timestamp": datetime.now().isoformat(),
        "n_fotos_total": len(todas_fotos),
        "fotos": [],
        "resumo": {},
    }

    t_inicio = time.time()
    n_sucesso = 0
    n_erro = 0
    n_items_total = 0

    for idx, caminho_foto in enumerate(todas_fotos, start=1):
        log.info("\n[%d/%d] %s", idx, len(todas_fotos), caminho_foto.name)
        log.info("-" * 80)

        try:
            resultado = executar_pipeline_completo(str(caminho_foto), foto_id=idx)
            resultados["fotos"].append(resultado)

            if resultado["erros"]:
                n_erro += 1
                log.warning("Foto com erros: %d erro(s)", len(resultado["erros"]))
            else:
                n_sucesso += 1
                n_items_total += resultado["itens_criados"]
                log.info("✓ Foto completa: %d itens criados", resultado["itens_criados"])

        except Exception as e:
            log.exception("Falha ao processar foto %s", caminho_foto.name)
            resultados["fotos"].append({
                "foto_id": idx,
                "arquivo": caminho_foto.name,
                "erro": str(e),
            })
            n_erro += 1

    # Resumo final
    tempo_total = time.time() - t_inicio
    resultados["resumo"] = {
        "fotos_sucesso": n_sucesso,
        "fotos_erro": n_erro,
        "items_criados": n_items_total,
        "tempo_total_s": tempo_total,
        "tempo_medio_foto_s": tempo_total / len(todas_fotos) if todas_fotos else 0,
    }

    # Salva relatório
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    relatorio_json = REPORTS_DIR / f"pipeline_full_{timestamp_str}.json"

    with open(relatorio_json, "w") as f:
        json.dump(resultados, f, indent=2)

    log.info("\n" + "=" * 80)
    log.info("PIPELINE COMPLETO — RESUMO FINAL")
    log.info("=" * 80)
    log.info("Fotos processadas: %d", len(todas_fotos))
    log.info("  ✓ Sucesso: %d", n_sucesso)
    log.info("  ✗ Erro: %d", n_erro)
    log.info("Itens criados: %d", n_items_total)
    log.info("Tempo total: %.1f min (%.1fs/foto)", tempo_total / 60, tempo_total / len(todas_fotos))
    log.info("Relatório: %s", relatorio_json)
    log.info("=" * 80)

    print("\n" + "=" * 80)
    print("✓ PIPELINE COMPLETO FINALIZADO")
    print("=" * 80)
    print(f"Fotos: {n_sucesso}/{len(todas_fotos)} sucesso, {n_erro} erro")
    print(f"Itens criados: {n_items_total}")
    print(f"Tempo: {tempo_total/60:.1f} min ({tempo_total/len(todas_fotos):.1f}s/foto)")
    print(f"Relatório: {relatorio_json}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
