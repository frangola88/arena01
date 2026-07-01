"""
core/exportar_poligonos_claude.py — Empacota a saída de alpha_pipeline para
handoff manual ao Claude Web: uma figura anotada (alpha + contornos) e um
JSON com os polígonos em coordenadas normalizadas [0,1], no mesmo formato
bbox {x1,y1,x2,y2} usado em vetorizador_raster._calcular_iou_vs_claude.
"""

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.alpha_pipeline import processar_imagem


def _normalizar_poligonos(poligonos: list, grid: int) -> list[dict]:
    """Converte cada polígono (coords em pixels de grid) para formato
    normalizado [0,1]: bbox, centroid, area_pct, n_vertices."""
    normalizados = []
    for i, p in enumerate(poligonos):
        geom = p["geometry"]
        minx, miny, maxx, maxy = geom.bounds
        cx, cy = p["centroid"]
        normalizados.append({
            "id": i + 1,
            "bbox": {
                "x1": round(minx / grid, 4),
                "y1": round(miny / grid, 4),
                "x2": round(maxx / grid, 4),
                "y2": round(maxy / grid, 4),
            },
            "centroid": {
                "x": round(cx / grid, 4),
                "y": round(cy / grid, 4),
            },
            "area_pct": round(100.0 * p["area"] / (grid * grid), 4),
            "n_vertices": p["n_vertices"],
        })
    return normalizados


def _desenhar_figura(resultado: dict, out_path: str) -> None:
    """Salva alpha_grid + contornos dos polígonos simplificados em PNG."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(resultado["alpha_grid"], cmap="viridis", vmin=0, vmax=1)

    cmap = plt.cm.tab10
    for i, p in enumerate(resultado["poligonos"]):
        geom = p["geometry"]
        color = cmap(i % 10)
        polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        for poly in polys:
            xs, ys = poly.exterior.xy
            ax.plot(xs, ys, color=color, linewidth=2)
            for interior in poly.interiors:
                ixs, iys = interior.xy
                ax.plot(ixs, iys, color=color, linewidth=1.2, linestyle="--")
            cx, cy = poly.centroid.x, poly.centroid.y
            ax.text(cx, cy, str(i + 1), color="white", fontsize=11,
                     fontweight="bold", ha="center", va="center")

    ax.set_title(f"{resultado['n_poligonos']} polígono(s)  "
                 f"cobertura={resultado['cobertura_pct']:.1f}%")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def gerar_pacote_claude(img_path: str, out_dir: str, *, grid: int = 280,
                         threshold: float = 0.5, morph_kernel: int = 5,
                         simplify_tol: float = 2.0) -> dict:
    """Roda o pipeline alpha completo e exporta o material para handoff manual
    ao Claude Web.

    Grava em `out_dir`:
      poligonos.png  — alpha U2Net + contornos numerados
      poligonos.json — polígonos normalizados [0,1] + metadados do pipeline

    Retorna dict com 'png_path', 'json_path', 'poligonos' (normalizados) e
    'resumo' (n_poligonos, cobertura_pct, tempo_ms).
    """
    from pathlib import Path
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    resultado = processar_imagem(
        img_path, grid=grid, threshold=threshold,
        morph_kernel=morph_kernel, simplify_tol=simplify_tol,
    )

    poligonos_norm = _normalizar_poligonos(resultado["poligonos"], grid)

    png_path = str(Path(out_dir) / "poligonos.png")
    json_path = str(Path(out_dir) / "poligonos.json")

    _desenhar_figura(resultado, png_path)

    payload = {
        "imagem_origem": img_path,
        "grid": grid,
        "gerado_em": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_poligonos": resultado["n_poligonos"],
        "cobertura_pct": round(resultado["cobertura_pct"], 4),
        "tempo_ms": resultado["tempo_ms"],
        "poligonos": poligonos_norm,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return {
        "png_path": png_path,
        "json_path": json_path,
        "poligonos": poligonos_norm,
        "resumo": {
            "n_poligonos": resultado["n_poligonos"],
            "cobertura_pct": round(resultado["cobertura_pct"], 4),
            "tempo_ms": resultado["tempo_ms"],
        },
    }
