"""Benchmark completo — todas as fotos, todas as abordagens."""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.gazetteer import GazetteerMatcher

g = GazetteerMatcher(
    emb_path=Path(__file__).parent.parent/"embeddings.npy",
    faiss_path=Path(__file__).parent.parent/"data/gazetteer/gazetteer.faiss",
    map_path=Path(__file__).parent.parent/"data/gazetteer/index_mapping.jsonl",
)
def area(b): return (b["x2"]-b["x1"])*(b["y2"]-b["y1"])*100

CASOS = [
    # ── foto 1: chaves de fenda (cena densa) ─────────────────────────────────
    {"label":"Philips cluster (vermelho+prata+bege)",
     "foto":"/home/cuco/Downloads/IMG_20260324_153910599.jpg",
     "bbox":{"x1":0.50,"y1":0.00,"x2":0.98,"y2":0.20},
     "gt_area": 9.6,  # ideal = Claude (correto neste caso)
     "cores":[
         {"hex":"#C0392B","area_pct":45},{"hex":"#C0C0C0","area_pct":40},
         {"hex":"#F5CBA7","area_pct":15}]},
    {"label":"Cabo Laranja (laranja único)",
     "foto":"/home/cuco/Downloads/IMG_20260324_153910599.jpg",
     "bbox":{"x1":0.22,"y1":0.12,"x2":0.80,"y2":0.30},
     "gt_area": 6.6,
     "cores":[
         {"hex":"#E06428","area_pct":50},{"hex":"#C0C0C0","area_pct":35},
         {"hex":"#8B4513","area_pct":15}]},
    {"label":"Chave Amarela Worker",
     "foto":"/home/cuco/Downloads/IMG_20260324_153910599.jpg",
     "bbox":{"x1":0.28,"y1":0.68,"x2":0.75,"y2":0.98},
     "gt_area": 12.0,
     "cores":[
         {"hex":"#F4D03F","area_pct":40},{"hex":"#1A1A1A","area_pct":35},
         {"hex":"#C0C0C0","area_pct":25}]},
    # ── foto 2: dois alicates amarelos (caso difícil) ─────────────────────────
    {"label":"Alicate Bico Curvo (amarelo/preto)",
     "foto":"/home/cuco/Downloads/IMG_20260602_185719031.jpg",
     "bbox":{"x1":0.06,"y1":0.18,"x2":0.30,"y2":0.62},
     "gt_area": 9.0,  # esperado: ~10% (objeto completo, sem expansão)
     "cores":[
         {"hex":"#E8C84A","area_pct":45},{"hex":"#1A1A1A","area_pct":35},
         {"hex":"#909090","area_pct":20}]},
    {"label":"Alicate Corte Diagonal (amarelo esc.)",
     "foto":"/home/cuco/Downloads/IMG_20260602_185719031.jpg",
     "bbox":{"x1":0.28,"y1":0.14,"x2":0.58,"y2":0.65},
     "gt_area": 7.5,
     "cores":[
         {"hex":"#D4A017","area_pct":50},{"hex":"#1A1A1A","area_pct":35},
         {"hex":"#808080","area_pct":15}]},
    {"label":"Desencapador LAOA (verde único)",
     "foto":"/home/cuco/Downloads/IMG_20260602_185719031.jpg",
     "bbox":{"x1":0.54,"y1":0.04,"x2":0.96,"y2":0.62},
     "gt_area": 10.0,
     "cores":[
         {"hex":"#2E7D32","area_pct":50},{"hex":"#1A1A1A","area_pct":25},
         {"hex":"#C0C0C0","area_pct":25}]},
    # ── foto 3: duas chaves sobrepostas ──────────────────────────────────────
    {"label":"CRAFTSMAN Azul (azul único)",
     "foto":"/home/cuco/Downloads/duaschaves.png",
     "bbox":{"x1":0.04,"y1":0.04,"x2":0.52,"y2":0.78},
     "gt_area": 35.0,
     "cores":[
         {"hex":"#4A90D9","area_pct":55},{"hex":"#CC2200","area_pct":20},
         {"hex":"#C0C0C0","area_pct":25}]},
    {"label":"Chave Preta/Vermelha",
     "foto":"/home/cuco/Downloads/duaschaves.png",
     "bbox":{"x1":0.38,"y1":0.04,"x2":0.88,"y2":0.82},
     "gt_area": 36.0,
     "cores":[
         {"hex":"#1A1A1A","area_pct":60},{"hex":"#CC2200","area_pct":25},
         {"hex":"#C0C0C0","area_pct":15}]},
]

print(f"\n{'Caso':<40} {'Claude':>7} {'S/cor':>7} {'C/cor':>7} {'Δ':>7} {'GT':>6}")
print("─"*80)
delta_total = 0
for c in CASOS:
    r0  = g.refinar_bbox(c["foto"], c["bbox"])
    r_c = g.refinar_bbox(c["foto"], c["bbox"], cores=c["cores"])
    ac  = area(c["bbox"])
    a0  = area(r0)
    aco = area(r_c)
    gt  = c["gt_area"]
    delta = aco - gt
    delta_total += abs(delta)
    ok = "✅" if abs(aco-gt) < abs(a0-gt) else ("🟡" if abs(aco-gt)==abs(a0-gt) else "❌")
    print(f"  {c['label'][:38]:<38} {ac:>6.1f}% {a0:>6.1f}% {aco:>6.1f}% {delta:>+6.1f}% {ok}")

print("─"*80)
print(f"\nδ total |com_cor - gt|: {delta_total:.1f}%")
