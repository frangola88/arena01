"""
Teste empírico: a skill devolve bboxes precisas?

Pede a análise rica da foto (com bboxes) e desenha o resultado.
Imagem de saída: /tmp/teste_bbox_skill.jpg
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from PIL import Image, ImageOps
from core.visao_global import analisar_foto_completa


def desenhar_bboxes(caminho_foto: str, objetos: list[dict], saida: str) -> None:
    """Desenha cada bbox numerado + nome em cima da foto."""
    pil = Image.open(caminho_foto)
    pil = ImageOps.exif_transpose(pil)
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]

    cores = [
        (0, 255, 0),    # verde
        (0, 165, 255),  # laranja
        (255, 0, 255),  # magenta
        (255, 255, 0),  # ciano
        (0, 255, 255),  # amarelo
    ]

    for i, obj in enumerate(objetos):
        bbox = obj.get("bbox_normalizada")
        if not bbox:
            print(f"  ⚠ Objeto {i+1} ({obj['nome']}) sem bbox")
            continue

        x1 = int(bbox["x1"] * w)
        y1 = int(bbox["y1"] * h)
        x2 = int(bbox["x2"] * w)
        y2 = int(bbox["y2"] * h)

        cor = cores[i % len(cores)]
        cv2.rectangle(img, (x1, y1), (x2, y2), cor, 4)

        # Label
        label = f"#{i+1} {obj['nome']}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (x1, y1), (x1 + lw + 10, y1 + lh + 12), cor, -1)
        cv2.putText(img, label, (x1 + 5, y1 + lh + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Confiança no canto inferior do bbox
        conf = obj.get("confianca", 0)
        cv2.putText(img, f"{conf:.2f}", (x1 + 5, y2 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)

        print(f"  #{i+1} {obj['nome']:30s} bbox=({x1},{y1})→({x2},{y2})  "
              f"conf={conf:.2f}  marca={obj.get('marca', '—')}")

    cv2.imwrite(saida, img)


def main():
    foto = "/home/cuco/projetos/casaiq/storage/fotos_originais/20260513_235112_376761.png"
    if not Path(foto).exists():
        # Procura qualquer .png ou .jpg em fotos_originais
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            for p in Path(foto).parent.glob(ext):
                if "_debug" not in p.name and "_anotada" not in p.name:
                    foto = str(p)
                    break

    print(f"=== TESTANDO SKILL COM BBOX em {foto} ===\n")
    analise = analisar_foto_completa(foto)
    objetos = analise.get("objetos", [])

    print(f"\nContexto: {analise.get('contexto_da_cena', '—')}")
    print(f"Fundo: {analise.get('fundo', '—')}")
    print(f"\nObjetos identificados ({len(objetos)}):")
    for o in objetos:
        print(f"  • {o['nome']:30s} conf={o.get('confianca',0):.2f}  "
              f"bbox={'✓' if o.get('bbox_normalizada') else '✗'}")

    saida = "/tmp/teste_bbox_skill.jpg"
    print(f"\nDesenhando bboxes em: {saida}\n")
    desenhar_bboxes(foto, objetos, saida)

    print(f"\n✅ Pronto. Abra {saida} para validar visualmente.")


if __name__ == "__main__":
    main()
