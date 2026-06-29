#!/usr/bin/env python3
"""
Pipeline Simples — Roda agent_1 em todas as fotos mestras.

O Agent 1 já coordena os demais agentes internamente.
"""
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

BASE = Path.home() / "projetos" / "casaiq"
FOTOS_DIR = BASE / "storage" / "fotos_originais"
REPORTS_DIR = BASE / "pipeline_reports"
REPORTS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("pipeline_simple")

sys.path.insert(0, str(BASE))
from agents.agent_1_segmentador import segmentar_foto

def main():
    todas_fotos = sorted([f for f in FOTOS_DIR.glob("*.jpg") if "_debug" not in f.name])

    if not todas_fotos:
        log.error("Nenhuma foto encontrada")
        return

    log.info("=" * 80)
    log.info("PIPELINE SIMPLES — Agent 1 em %d fotos", len(todas_fotos))
    log.info("=" * 80)

    resultados = {
        "timestamp": datetime.now().isoformat(),
        "fotos_total": len(todas_fotos),
        "fotos_sucesso": 0,
        "fotos_erro": 0,
        "objetos_total": 0,
        "detalhes": []
    }

    t_inicio = time.time()

    for idx, foto in enumerate(todas_fotos, 1):
        log.info("[%d/%d] %s", idx, len(todas_fotos), foto.name)
        try:
            objetos = segmentar_foto(str(foto), foto_id=idx)
            n_obj = len(objetos)
            resultados["objetos_total"] += n_obj
            resultados["fotos_sucesso"] += 1
            resultados["detalhes"].append({
                "arquivo": foto.name,
                "status": "ok",
                "n_objetos": n_obj
            })
            log.info("✓ %d objetos", n_obj)
        except Exception as e:
            resultados["fotos_erro"] += 1
            resultados["detalhes"].append({
                "arquivo": foto.name,
                "status": "erro",
                "erro": str(e)[:100]
            })
            log.exception("✗ Erro")

    tempo = time.time() - t_inicio
    resultados["tempo_total_s"] = tempo
    resultados["tempo_medio_s"] = tempo / len(todas_fotos)

    # Salva
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rel = REPORTS_DIR / f"pipeline_simple_{ts}.json"
    with open(rel, "w") as f:
        json.dump(resultados, f, indent=2)

    log.info("\n" + "=" * 80)
    log.info("RESULTADO FINAL")
    log.info("=" * 80)
    log.info("Fotos: %d/%d sucesso", resultados["fotos_sucesso"], len(todas_fotos))
    log.info("Objetos: %d total", resultados["objetos_total"])
    log.info("Tempo: %.1f min (%.1fs/foto)", tempo / 60, tempo / len(todas_fotos))
    log.info("Relatório: %s", rel)
    log.info("=" * 80)

    print("\n✓ PIPELINE FINALIZADO\n")

if __name__ == "__main__":
    main()
