"""Genera los ejemplos de inferencia del informe con el checkpoint limpio.

Elige casos del split de test por criterio explicito (un acierto compacto, una
imagen multi-instancia y el peor taxon) en vez de tomar los primeros que salgan,
para que las figuras del informe sean representativas y no cosmeticas.
"""
import csv
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
TEST = ROOT / "datasets" / "clean" / "test" / "images"
OUT = ROOT / "informe" / "imagenes"
PESOS = ROOT / "checkpoints" / "yolo26s_clean_best.pt"
CONF, IOU = 0.30, 0.60   # punto de operacion documentado del pipeline


def inventario_test():
    por_img = defaultdict(list)
    for r in csv.DictReader(open(ROOT / "reports" / "paper" / "bbox_instances.csv")):
        if r["split"] == "test":
            por_img[r["image"]].append(r)
    return por_img


def elegir(por_img):
    """Un caso facil, uno multi-instancia y el peor taxon (Chironomidae)."""
    multi = max(por_img.items(), key=lambda kv: len(kv[1]))
    facil = next(kv for kv in sorted(por_img.items())
                 if len(kv[1]) == 1 and kv[1][0]["family"] == "Belostomatidae")
    dificil = next(kv for kv in sorted(por_img.items())
                   if kv[1][0]["family"] == "Chironomidae")
    return {"ejemplo_simple": facil, "ejemplo_multiple": multi, "ejemplo_dificil": dificil}


def main():
    modelo = YOLO(str(PESOS))
    por_img = inventario_test()
    for nombre, (stem, objs) in elegir(por_img).items():
        # bbox_instances guarda el stem con la extension aplanada a _JPG
        candidatos = list(TEST.glob(stem + ".*")) or list(TEST.glob(stem.rsplit("_", 1)[0] + ".*"))
        if not candidatos:
            print(f"AVISO  sin archivo para {stem}, se omite {nombre}")
            continue
        img = candidatos[0]
        res = modelo.predict(str(img), conf=CONF, iou=IOU, verbose=False)[0]
        res.save(filename=str(OUT / f"{nombre}.png"))
        etiquetas = ", ".join(sorted({o["family"] for o in objs}))
        dets = [f"{modelo.names[int(c)]} {p:.2f}"
                for c, p in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist(), strict=False)]
        print(f"{nombre:18} {img.name}")
        print(f"{'':18}   verdad: {len(objs)} obj ({etiquetas})")
        print(f"{'':18}   predicho: {len(dets)} obj ({', '.join(dets) or 'ninguno'})")


if __name__ == "__main__":
    main()
