"""
alpha_pipeline.py — Pipeline completo: Alpha U2Net → morph close → vetorizar_superficie → simplify
"""

import json
import time
import numpy as np
import cv2
from PIL import Image
from shapely.geometry import shape

from core.vetorizador_raster import vetorizar_superficie

_ORT_SESSION = None
_MODEL_PATH = "/home/cuco/.u2net/u2net.onnx"


def _get_session():
    global _ORT_SESSION
    if _ORT_SESSION is None:
        import onnxruntime as ort
        _ORT_SESSION = ort.InferenceSession(_MODEL_PATH, providers=["CPUExecutionProvider"])
    return _ORT_SESSION


def _u2net_infer(img_path: str, grid: int, threshold: float) -> tuple[np.ndarray, float]:
    """Retorna (alpha_grid float32 [0,1] grid×grid, elapsed_ms)."""
    t0 = time.perf_counter()

    img = Image.open(img_path).convert("RGB")
    img_320 = img.resize((320, 320), Image.BILINEAR)
    arr = np.array(img_320, dtype=np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)[np.newaxis, :]  # (1,3,320,320)

    session = _get_session()
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: arr})

    pred = outputs[0][0, 0]  # (320,320)
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

    # resize para grid×grid
    alpha_grid = cv2.resize(pred, (grid, grid), interpolation=cv2.INTER_LINEAR)
    alpha_grid = np.clip(alpha_grid, 0.0, 1.0).astype(np.float32)

    elapsed = (time.perf_counter() - t0) * 1000
    return alpha_grid, elapsed


def _morph_close(alpha: np.ndarray, kernel_size: int) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    uint8 = (alpha * 255).astype(np.uint8)
    closed = cv2.morphologyEx(uint8, cv2.MORPH_CLOSE, kernel)
    result = closed.astype(np.float32) / 255.0
    elapsed = (time.perf_counter() - t0) * 1000
    return result, elapsed


def _vetorizar(sup: np.ndarray) -> tuple[list, float]:
    t0 = time.perf_counter()
    res = vetorizar_superficie(sup)
    features = json.loads(res["geojson"])["features"]
    elapsed = (time.perf_counter() - t0) * 1000
    return features, elapsed


def _simplify_collect(features: list, tol: float) -> tuple[list, float]:
    t0 = time.perf_counter()
    polys = []
    for feat in features:
        geom = shape(feat["geometry"])
        geom = geom.simplify(tol, preserve_topology=True)
        if geom.is_empty or not geom.is_valid:
            continue
        area = geom.area
        cx, cy = geom.centroid.x, geom.centroid.y
        n_verts = len(list(geom.geoms[0].exterior.coords)) if geom.geom_type == "MultiPolygon" else (
            len(list(geom.exterior.coords)) if geom.geom_type == "Polygon" else 0
        )
        polys.append({"geometry": geom, "area": area, "n_vertices": n_verts, "centroid": (cx, cy)})

    polys.sort(key=lambda p: p["area"], reverse=True)
    polys = polys[:10]
    elapsed = (time.perf_counter() - t0) * 1000
    return polys, elapsed


def processar_imagem(img_path: str, grid: int = 280, threshold: float = 0.5,
                     morph_kernel: int = 5, simplify_tol: float = 2.0) -> dict:
    """
    Pipeline completo: imagem → polígonos vetoriais via alpha U2Net.

    Retorna dict com:
      'alpha_grid'    : np.ndarray (grid×grid) superficie alpha [0,1]
      'sup_morph'     : np.ndarray (grid×grid) após morph close
      'poligonos'     : list[dict] cada item: {'geometry': shapely_geom, 'area': float,
                         'n_vertices': int, 'centroid': (x,y)}
      'n_poligonos'   : int
      'area_total'    : float (px²)
      'cobertura_pct' : float (% do grid)
      'tempo_ms'      : dict {'u2net': float, 'morph': float, 'vetorizar': float,
                              'simplify': float, 'total': float}
    """
    t_total = time.perf_counter()

    alpha_grid, t_u2net = _u2net_infer(img_path, grid, threshold)
    sup_morph, t_morph = _morph_close(alpha_grid, morph_kernel)
    features, t_vet = _vetorizar(sup_morph)
    polys, t_simp = _simplify_collect(features, simplify_tol)

    area_total = sum(p["area"] for p in polys)
    cobertura_pct = (area_total / (grid * grid)) * 100.0
    t_total_ms = (time.perf_counter() - t_total) * 1000

    return {
        "alpha_grid":    alpha_grid,
        "sup_morph":     sup_morph,
        "poligonos":     polys,
        "n_poligonos":   len(polys),
        "area_total":    area_total,
        "cobertura_pct": cobertura_pct,
        "tempo_ms": {
            "u2net":    round(t_u2net, 1),
            "morph":    round(t_morph, 1),
            "vetorizar": round(t_vet, 1),
            "simplify": round(t_simp, 1),
            "total":    round(t_total_ms, 1),
        },
    }
