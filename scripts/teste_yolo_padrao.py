"""Teste com YOLOv8 PADRÃO (80 classes COCO) - serve para validar setup."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO

foto = "/home/cuco/projetos/casaiq/storage/fotos_originais/20260513_235112_376761.png"
for p in Path(foto).parent.glob("*.png"):
    if "_debug" not in p.name and "_anotada" not in p.name:
        foto = str(p); break

print(f"Foto: {foto}")
print("Carregando YOLOv8 nano (COCO)...")
model = YOLO("yolov8n.pt")

print("Detectando com threshold 0.05...")
r = model.predict(foto, conf=0.05, verbose=False)[0]
print(f"\nDetecções: {len(r.boxes) if r.boxes is not None else 0}")
if r.boxes is not None:
    for i in range(len(r.boxes)):
        cls = r.names[int(r.boxes.cls[i])]
        conf = float(r.boxes.conf[i])
        bbox = r.boxes.xyxy[i].tolist()
        print(f"  {cls:20s} conf={conf:.3f} bbox={bbox}")

# Tenta também com threshold default 0.25
print("\n--- Threshold 0.25 ---")
r = model.predict(foto, conf=0.25, verbose=False)[0]
print(f"Detecções: {len(r.boxes) if r.boxes is not None else 0}")
if r.boxes is not None:
    for i in range(len(r.boxes)):
        cls = r.names[int(r.boxes.cls[i])]
        conf = float(r.boxes.conf[i])
        print(f"  {cls:20s} conf={conf:.3f}")
