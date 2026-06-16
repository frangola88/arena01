"""Teste 2: YOLO World L (large) + threshold baixíssimo + imgsz=640."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLOWorld

foto = "/home/cuco/projetos/casaiq/storage/fotos_originais/20260513_235112_376761.png"
for p in Path(foto).parent.glob("*.png"):
    if "_debug" not in p.name and "_anotada" not in p.name:
        foto = str(p); break

print(f"Foto: {foto}")

# Modelo LARGE (mais capacidade)
print("Carregando yolov8l-world.pt (large, ~140MB)...")
model = YOLOWorld("yolov8l-world.pt")

# Classes mais simples + algumas COCO para validar
classes = [
    "screwdriver", "tool", "hand tool",
    "ruler", "wooden ruler",
    "scissors", "knife", "bottle",  # COCO known
]
model.set_classes(classes)
print(f"Classes: {classes}\n")

# Threshold MUITO baixo + imgsz fixo
print("Predict imgsz=640, conf=0.001...")
r = model.predict(foto, conf=0.001, imgsz=640, verbose=False)[0]
print(f"Detecções: {len(r.boxes) if r.boxes is not None else 0}\n")

if r.boxes is not None:
    for i in range(len(r.boxes)):
        cls = classes[int(r.boxes.cls[i])]
        conf = float(r.boxes.conf[i])
        bbox = r.boxes.xyxy[i].tolist()
        print(f"  {cls:25s} conf={conf:.4f}  bbox={[int(b) for b in bbox]}")

# Desenhar
pil = Image.open(foto)
pil = ImageOps.exif_transpose(pil)
img = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
cores = [(0,255,0), (0,165,255), (255,0,255), (0,255,255), (255,0,0)]
if r.boxes is not None:
    for i in range(len(r.boxes)):
        x1, y1, x2, y2 = [int(v) for v in r.boxes.xyxy[i].tolist()]
        cls = classes[int(r.boxes.cls[i])]
        conf = float(r.boxes.conf[i])
        cv2.rectangle(img, (x1,y1), (x2,y2), cores[i % 5], 3)
        cv2.putText(img, f"{cls} {conf:.2f}", (x1+5, y1+25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, cores[i % 5], 2)
cv2.imwrite("/tmp/teste_yolo_world_v2.jpg", img)
print(f"\n✅ Salvo em /tmp/teste_yolo_world_v2.jpg")
