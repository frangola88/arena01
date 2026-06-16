"""Benchmark com centroide: compara sem_cor, com_cor, com_cor+centroide."""
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
    {"label":"Philips cluster",
     "foto":"/home/cuco/Downloads/IMG_20260324_153910599.jpg",
     "bbox":{"x1":0.50,"y1":0.00,"x2":0.98,"y2":0.20},
     "gt":9.6,
     "centroide":{"cx":0.74,"cy":0.10},
     "cores":[{"hex":"#C0392B","area_pct":45},{"hex":"#C0C0C0","area_pct":40},{"hex":"#F5CBA7","area_pct":15}]},
    {"label":"Cabo Laranja",
     "foto":"/home/cuco/Downloads/IMG_20260324_153910599.jpg",
     "bbox":{"x1":0.22,"y1":0.12,"x2":0.80,"y2":0.30},
     "gt":6.6,
     "centroide":{"cx":0.38,"cy":0.20},
     "cores":[{"hex":"#E06428","area_pct":50},{"hex":"#C0C0C0","area_pct":35},{"hex":"#8B4513","area_pct":15}]},
    {"label":"Chave Amarela",
     "foto":"/home/cuco/Downloads/IMG_20260324_153910599.jpg",
     "bbox":{"x1":0.28,"y1":0.68,"x2":0.75,"y2":0.98},
     "gt":12.0,
     "centroide":{"cx":0.50,"cy":0.83},
     "cores":[{"hex":"#F4D03F","area_pct":40},{"hex":"#1A1A1A","area_pct":35},{"hex":"#C0C0C0","area_pct":25}]},
    {"label":"Bico Curvo",
     "foto":"/home/cuco/Downloads/IMG_20260602_185719031.jpg",
     "bbox":{"x1":0.06,"y1":0.18,"x2":0.30,"y2":0.62},
     "gt":9.0,
     "centroide":{"cx":0.18,"cy":0.40},
     "cores":[{"hex":"#E8C84A","area_pct":45},{"hex":"#1A1A1A","area_pct":35},{"hex":"#909090","area_pct":20}]},
    {"label":"Corte Diagonal",
     "foto":"/home/cuco/Downloads/IMG_20260602_185719031.jpg",
     "bbox":{"x1":0.28,"y1":0.14,"x2":0.58,"y2":0.65},
     "gt":7.5,
     "centroide":{"cx":0.43,"cy":0.39},
     "cores":[{"hex":"#D4A017","area_pct":50},{"hex":"#1A1A1A","area_pct":35},{"hex":"#808080","area_pct":15}]},
    {"label":"LAOA",
     "foto":"/home/cuco/Downloads/IMG_20260602_185719031.jpg",
     "bbox":{"x1":0.54,"y1":0.04,"x2":0.96,"y2":0.62},
     "gt":10.0,
     "centroide":{"cx":0.75,"cy":0.33},
     "cores":[{"hex":"#2E7D32","area_pct":50},{"hex":"#1A1A1A","area_pct":25},{"hex":"#C0C0C0","area_pct":25}]},
    {"label":"CRAFTSMAN Azul",
     "foto":"/home/cuco/Downloads/duaschaves.png",
     "bbox":{"x1":0.04,"y1":0.04,"x2":0.52,"y2":0.78},
     "gt":35.0,
     "centroide":{"cx":0.20,"cy":0.41},
     "cores":[{"hex":"#4A90D9","area_pct":55},{"hex":"#CC2200","area_pct":20},{"hex":"#C0C0C0","area_pct":25}]},
    {"label":"Preta/Vermelha",
     "foto":"/home/cuco/Downloads/duaschaves.png",
     "bbox":{"x1":0.38,"y1":0.04,"x2":0.88,"y2":0.82},
     "gt":36.0,
     "centroide":{"cx":0.63,"cy":0.43},
     "cores":[{"hex":"#1A1A1A","area_pct":60},{"hex":"#CC2200","area_pct":25},{"hex":"#C0C0C0","area_pct":15}]},
]

print(f"\n{'Caso':<22} {'S/cor':>6} {'C/cor':>6} {'C/cor+C':>8}  GT")
print("─"*60)
deltas = {"s":[], "c":[], "cc":[]}
for caso in CASOS:
    r0  = g.refinar_bbox(caso["foto"], caso["bbox"])
    rc  = g.refinar_bbox(caso["foto"], caso["bbox"], cores=caso["cores"])
    rcc = g.refinar_bbox(caso["foto"], caso["bbox"],
                         cores=caso["cores"], centroide=caso["centroide"])
    a0, ac, acc, gt = area(r0), area(rc), area(rcc), caso["gt"]
    d0 = abs(a0-gt); dc = abs(ac-gt); dcc = abs(acc-gt)
    deltas["s"].append(d0); deltas["c"].append(dc); deltas["cc"].append(dcc)
    best = min(d0, dc, dcc)
    def mk(a, d):
        m = "✅" if d == best and d < 3 else ("🟡" if d == best else "  ")
        return f"{a:5.1f}%{m}"
    print(f"  {caso['label']:<20} {mk(a0,d0)} {mk(ac,dc)} {mk(acc,dcc)}  {gt:.1f}%")

print("─"*60)
print(f"  {'δ total':<20} {sum(deltas['s']):5.1f}  {sum(deltas['c']):5.1f}  {sum(deltas['cc']):7.1f}")
