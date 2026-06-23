#!/usr/bin/env python3
"""
Test Pipeline Validation — Rodar pipeline completo em fotos mestras reais
e gerar relatório de confiança do score_qualidade.

Uso:
  cd ~/projetos/casaiq
  conda run -n casaiq python test_pipeline_validation.py
"""
import json
import sys
import logging
from pathlib import Path
from datetime import datetime
import numpy as np

# Config
BASE = Path.home() / "projetos" / "casaiq"
FOTOS_DIR = BASE / "storage" / "fotos_originais"
REPORT_DIR = BASE / "validation_reports"
REPORT_DIR.mkdir(exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("validation")

# Imports locais
sys.path.insert(0, str(BASE))
from agents.agent_1_segmentador import segmentar_foto


def run_validation(n_fotos=4):
    """Roda pipeline em n_fotos e coleta scores."""

    # Seleciona fotos (ignora _debug)
    todas_fotos = sorted([
        f for f in FOTOS_DIR.glob("*.jpg")
        if "_debug" not in f.name
    ])

    if not todas_fotos:
        log.error("Nenhuma foto encontrada em %s", FOTOS_DIR)
        return

    fotos_selecionadas = todas_fotos[:n_fotos]
    log.info("Processando %d fotos de %d disponíveis", len(fotos_selecionadas), len(todas_fotos))

    resultados_globais = {
        "timestamp": datetime.now().isoformat(),
        "n_fotos": len(fotos_selecionadas),
        "fotos": [],
        "estatisticas_globais": {},
    }

    scores_por_foto = []
    flags_contador = {}

    for idx, caminho_foto in enumerate(fotos_selecionadas, start=1):
        foto_id = idx
        log.info("=" * 70)
        log.info("[%d/%d] Processando: %s", idx, len(fotos_selecionadas), caminho_foto.name)
        log.info("=" * 70)

        try:
            resultado_objetos = segmentar_foto(str(caminho_foto), foto_id=foto_id)

            # Extrai scores de qualidade
            scores_foto = []
            objetos_com_flags = []

            for obj in resultado_objetos:
                score = obj.get("_score_qualidade", None)
                flags = obj.get("_flags_qualidade", [])

                if score is not None:
                    scores_foto.append(score)
                    objetos_com_flags.append({
                        "nome": obj.get("nome"),
                        "score": score,
                        "flags": flags,
                        "suspeita_bg": obj.get("_suspeita_bg", False),
                    })

                    # Conta flags globalmente
                    for flag in flags:
                        flags_contador[flag] = flags_contador.get(flag, 0) + 1

            # Resumo da foto
            foto_resumo = {
                "arquivo": caminho_foto.name,
                "n_objetos": len(resultado_objetos),
                "n_com_score": len(scores_foto),
                "objetos": objetos_com_flags,
            }

            if scores_foto:
                foto_resumo.update({
                    "score_medio": float(np.mean(scores_foto)),
                    "score_min": float(np.min(scores_foto)),
                    "score_max": float(np.max(scores_foto)),
                    "score_std": float(np.std(scores_foto)),
                })
                scores_por_foto.extend(scores_foto)
                log.info("Foto resumida: %d objetos, score_medio=%.3f ± %.3f [%.3f–%.3f]",
                        foto_resumo["n_objetos"],
                        foto_resumo["score_medio"],
                        foto_resumo["score_std"],
                        foto_resumo["score_min"],
                        foto_resumo["score_max"])
            else:
                log.warning("Foto: nenhum objeto com score calculado")

            resultados_globais["fotos"].append(foto_resumo)

        except Exception as e:
            log.exception("Erro ao processar %s: %s", caminho_foto.name, e)
            resultados_globais["fotos"].append({
                "arquivo": caminho_foto.name,
                "erro": str(e),
            })

    # Estatísticas globais
    if scores_por_foto:
        resultados_globais["estatisticas_globais"] = {
            "n_scores_coletados": len(scores_por_foto),
            "score_medio_global": float(np.mean(scores_por_foto)),
            "score_min_global": float(np.min(scores_por_foto)),
            "score_max_global": float(np.max(scores_por_foto)),
            "score_std_global": float(np.std(scores_por_foto)),
            "percentil_25": float(np.percentile(scores_por_foto, 25)),
            "percentil_50": float(np.percentile(scores_por_foto, 50)),
            "percentil_75": float(np.percentile(scores_por_foto, 75)),
            "flags_encontradas": sorted(flags_contador.items(), key=lambda x: -x[1]),
        }

    # Salva relatório
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    relatorio_json = REPORT_DIR / f"validation_{timestamp_str}.json"
    relatorio_txt = REPORT_DIR / f"validation_{timestamp_str}.txt"

    with open(relatorio_json, "w") as f:
        json.dump(resultados_globais, f, indent=2)
    log.info("Relatório JSON salvo: %s", relatorio_json)

    # Relatório em texto legível
    with open(relatorio_txt, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("RELATÓRIO DE VALIDAÇÃO DO PIPELINE — score_qualidade\n")
        f.write(f"Data/Hora: {datetime.now().isoformat()}\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"RESUMO GLOBAL\n")
        f.write(f"  Fotos processadas: {len(fotos_selecionadas)}\n")

        if scores_por_foto:
            stats = resultados_globais["estatisticas_globais"]
            f.write(f"  Scores coletados: {stats['n_scores_coletados']}\n")
            f.write(f"  Score médio: {stats['score_medio_global']:.3f}\n")
            f.write(f"  Intervalo: [{stats['score_min_global']:.3f}, {stats['score_max_global']:.3f}]\n")
            f.write(f"  Desvio padrão: {stats['score_std_global']:.3f}\n")
            f.write(f"  P25: {stats['percentil_25']:.3f}, P50: {stats['percentil_50']:.3f}, P75: {stats['percentil_75']:.3f}\n\n")

            f.write("FLAGS ENCONTRADAS (frequência):\n")
            for flag, count in stats["flags_encontradas"]:
                f.write(f"  - {flag}: {count}\n")
            f.write("\n")

        f.write("DETALHES POR FOTO\n")
        f.write("-" * 80 + "\n")
        for foto in resultados_globais["fotos"]:
            f.write(f"\n{foto['arquivo']}\n")
            if "erro" in foto:
                f.write(f"  ERRO: {foto['erro']}\n")
            else:
                f.write(f"  Objetos: {foto['n_objetos']}\n")
                if "score_medio" in foto:
                    f.write(f"  Score médio: {foto['score_medio']:.3f} ± {foto['score_std']:.3f}\n")
                    f.write(f"  Intervalo: [{foto['score_min']:.3f}, {foto['score_max']:.3f}]\n")
                    f.write(f"\n  Objetos detalhados:\n")
                    for obj in foto["objetos"]:
                        f.write(f"    • {obj['nome']}: score={obj['score']:.3f}")
                        if obj["suspeita_bg"]:
                            f.write(" [SUSPEITA_BG]")
                        if obj["flags"]:
                            f.write(f" flags={','.join(obj['flags'])}")
                        f.write("\n")

    log.info("Relatório TXT salvo: %s", relatorio_txt)

    # Exibe resumo no console
    print("\n" + "=" * 80)
    print("RESUMO FINAL")
    print("=" * 80)
    if scores_por_foto:
        stats = resultados_globais["estatisticas_globais"]
        print(f"\n✓ Pipeline rodou com sucesso em {len(fotos_selecionadas)} fotos")
        print(f"✓ {stats['n_scores_coletados']} scores coletados")
        print(f"\nScore médio global: {stats['score_medio_global']:.3f}")
        print(f"  Intervalo: [{stats['score_min_global']:.3f}, {stats['score_max_global']:.3f}]")
        print(f"  Desvio padrão: {stats['score_std_global']:.3f}")
        print(f"  Distribuição: P25={stats['percentil_25']:.3f}, P50={stats['percentil_50']:.3f}, P75={stats['percentil_75']:.3f}")

        print(f"\nTop flags encontradas:")
        for flag, count in stats["flags_encontradas"][:5]:
            print(f"  - {flag}: {count}")
    else:
        print("⚠ Nenhum score foi coletado")

    print("\nRelatorios:")
    print(f"  JSON: {relatorio_json}")
    print(f"  TXT:  {relatorio_txt}")
    print()


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    run_validation(n_fotos=n)
