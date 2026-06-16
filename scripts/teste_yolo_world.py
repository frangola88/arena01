"""
Teste empírico de YOLO World com classes customizadas.

Define classes dinamicamente ('chave de fenda', 'régua') e detecta.
Compara com bboxes que o Claude API havia dado.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLOWorld


def desenhar_bboxes(caminho_foto: str, deteccoes: list, saida: str) -> None:
    """Desenha bboxes do YOLO na foto."""
    pil = Image.open(caminho_foto)
    pil = ImageOps.exif_transpose(pil)
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    cores = [
        (0, 255, 0),    # verde
        (0, 165, 255),  # laranja
        (255, 0, 255),  # magenta
        (255, 255, 0),  # ciano
        (0, 255, 255),  # amarelo
    ]

    for i, det in enumerate(deteccoes):
        x1, y1, x2, y2 = det["bbox"]
        cor = cores[i % len(cores)]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), cor, 4)

        label = f"#{i+1} {det['classe']} {det['conf']:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (int(x1), int(y1)),
                      (int(x1) + lw + 10, int(y1) + lh + 12), cor, -1)
        cv2.putText(img, label, (int(x1) + 5, int(y1) + lh + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.imwrite(saida, img)


def main():
    foto = "/home/cuco/projetos/casaiq/storage/fotos_originais/20260513_235112_376761.png"
    if not Path(foto).exists():
        # Procura qualquer foto válida
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            for p in Path(foto).parent.glob(ext):
                if "_debug" not in p.name and "_anotada" not in p.name:
                    foto = str(p)
                    break

    print(f"=== TESTANDO YOLO WORLD em {foto} ===\n")

    # Carregar YOLO World (vai baixar pesos na primeira vez ~25MB)
    print("Carregando modelo yolov8s-world.pt...")
    model = YOLOWorld("yolov8s-world.pt")

    # Definir classes customizadas (em inglês — YOLO World é treinado em inglês)
    classes = [
        "screwdriver",        # chave de fenda
        "phillips screwdriver",
        "flathead screwdriver",
        "ruler",              # régua
        "wooden ruler",
        "measuring stick",
    ]
    model.set_classes(classes)
    print(f"Classes definidas: {classes}\n")

    # Detectar
    print("Detectando objetos...")
    resultados = model.predict(foto, conf=0.05, verbose=False)
    r = resultados[0]

    deteccoes = []
    if r.boxes is not None:
        for i in range(len(r.boxes)):
            cls_idx = int(r.boxes.cls[i])
            conf = float(r.boxes.conf[i])
            x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
            deteccoes.append({
                "classe": classes[cls_idx],
                "conf": conf,
                "bbox": (x1, y1, x2, y2),
            })

    print(f"\n{len(deteccoes)} detecções encontradas:")
    for i, d in enumerate(deteccoes):
        x1, y1, x2, y2 = d["bbox"]
        print(f"  #{i+1} {d['classe']:30s} conf={d['conf']:.3f}  "
              f"bbox=({x1:.0f},{y1:.0f})→({x2:.0f},{y2:.0f})")

    saida = "/tmp/teste_yolo_world.jpg"
    desenhar_bboxes(foto, deteccoes, saida)
    print(f"\n✅ Imagem salva: {saida}")


if __name__ == "__main__":
    main()
