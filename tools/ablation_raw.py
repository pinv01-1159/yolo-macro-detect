"""Copia instrumentada de tools/background_ablation.py.

Misma logica de brazos (crop_only con 15% de margen, background_only con gris
114, conf=0.3, criterio de distancia de centros), pero guarda TODAS las
detecciones crudas por instancia para poder recalcular metricas, criterios
alternativos, linea base de azar e IC agrupados por especimen offline.

Agrega un brazo de referencia "full": imagen original sin tocar (una prediccion
por imagen), para comparar crop vs contexto completo sobre las mismas
instancias.

Uso: python ablation_raw.py <model_path> <out_json> [--limit N] [--device cpu]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import background_ablation as BA  # noqa: E402  (reusa load_instances/crop/mask)


def dets(res):
    out = []
    for b in res.boxes:
        out.append(
            {
                "cls": int(b.cls[0]),
                "conf": float(b.conf[0]),
                "xyxy": [float(v) for v in b.xyxy[0].tolist()],
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("out_json")
    ap.add_argument("--data-dir", type=Path, default=REPO / "datasets/clean/test")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    insts = BA.load_instances(a.data_dir)
    if a.limit:
        insts = insts[: a.limit]
    model = YOLO(a.model_path)
    conf = BA.CONF_THRESHOLD
    full_cache = {}
    recs = []
    t0 = time.time()
    for i, inst in enumerate(insts):
        img = cv2.imread(str(inst["image_path"]))
        key = str(inst["image_path"])
        if key not in full_cache:
            full_cache[key] = dets(model.predict(img, verbose=False, conf=conf, device=a.device)[0])
        crop = BA.crop_only(img, inst["box"], inst["img_w"], inst["img_h"])
        crop_d = (
            []
            if crop.size == 0
            else dets(model.predict(crop, verbose=False, conf=conf, device=a.device)[0])
        )
        masked = BA.background_only(img, inst["box"])
        bg_d = dets(model.predict(masked, verbose=False, conf=conf, device=a.device)[0])
        recs.append(
            {
                "image": inst["image_path"].stem,
                "class_id": inst["class_id"],
                "box": list(inst["box"]),
                "img_w": inst["img_w"],
                "img_h": inst["img_h"],
                "area_frac": inst["area_frac"],
                "crop_shape": list(crop.shape[:2]),
                "crop": crop_d,
                "bg": bg_d,
                "full": full_cache[key],
            }
        )
        if i % 25 == 0:
            print(f"{i}/{len(insts)} {time.time() - t0:.1f}s", flush=True)
    Path(a.out_json).write_text(
        json.dumps(
            {
                "model": a.model_path,
                "conf": conf,
                "crop_padding": BA.CROP_PADDING,
                "n": len(recs),
                "seconds": time.time() - t0,
                "records": recs,
            }
        )
    )
    print("DONE", time.time() - t0)


if __name__ == "__main__":
    main()
