"""
Ablacion de fondo/contexto: mide cuanto depende un modelo entrenado del
fondo/sesion en vez de la morfologia real del animal (ver
docs/leakage_analysis.md seccion 3 -- confusor de sesion).

Dos experimentos por cada instancia ground-truth del test set:

  A) "solo bicho": recorte ajustado a la caja (margen 15%), sin contexto de
     fondo/sesion. Se corre el modelo sobre el recorte y se compara la clase
     predicha con mayor confianza contra la clase real. Si el modelo sigue
     acertando, es evidencia de que aprendio morfologia real.

  B) "solo fondo": en la imagen COMPLETA, se tapa la caja del bicho (relleno
     gris neutro) dejando el resto de la imagen intacta, y se corre deteccion
     completa. Si el modelo predice la clase correcta en esa zona SIN que
     haya bicho, es evidencia directa de atajo por fondo/sesion.

Se compara contra el recall "con contexto completo" que ya esta en
per_class_metrics.csv (deteccion normal, imagen sin modificar).

Uso:
    python tools/background_ablation.py <experiment_name> <model_path> \
        [--data-dir datasets/clean/test] [--device cuda:0]
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

CLASS_NAMES = [
    "Ampullariidae", "Ancylidae", "Belostomatidae", "Ceratopogonidae", "Chironomidae",
    "Coenagrionidae", "Dytiscidae", "Gerridae", "Glossiphoniidae", "Hirudinidae",
    "Hydrophilidae", "Hyriidae", "Libellulidae", "Miridae", "Noteridae", "Notonectidae",
    "Physidae", "Planorbidae", "Psychodidae",
]
CROP_PADDING = 0.15  # 15% del tamano de la caja, de margen a cada lado
CONF_THRESHOLD = 0.3  # mismo umbral que el eval oficial del pipeline


def load_instances(data_dir: Path) -> list[dict]:
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"
    instances = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        image_path = images_dir / f"{label_path.stem}.jpg"
        if not image_path.exists():
            continue
        img = cv2.imread(str(image_path))
        h, w = img.shape[:2]
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:5])
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            instances.append({
                "image_path": image_path,
                "class_id": cls_id,
                "box": (x1, y1, x2, y2),
                "img_w": w,
                "img_h": h,
            })
    return instances


def crop_only(img, box, img_w, img_h):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    pad_x, pad_y = bw * CROP_PADDING, bh * CROP_PADDING
    cx1 = max(0, int(x1 - pad_x))
    cy1 = max(0, int(y1 - pad_y))
    cx2 = min(img_w, int(x2 + pad_x))
    cy2 = min(img_h, int(y2 + pad_y))
    return img[cy1:cy2, cx1:cx2]


def background_only(img, box):
    x1, y1, x2, y2 = box
    masked = img.copy()
    masked[int(y1):int(y2), int(x1):int(x2)] = 114  # gris neutro, mismo valor de padding que usa Ultralytics en letterbox
    return masked


def box_center_dist(pred_box, gt_box):
    px = (pred_box[0] + pred_box[2]) / 2
    py = (pred_box[1] + pred_box[3]) / 2
    gx = (gt_box[0] + gt_box[2]) / 2
    gy = (gt_box[1] + gt_box[3]) / 2
    return ((px - gx) ** 2 + (py - gy) ** 2) ** 0.5


def run_experiment(model: YOLO, instances: list[dict], device: str) -> dict:
    crop_correct = 0
    crop_nodet = 0
    crop_wrong = 0
    bg_predicts_true_class = 0  # deteccion con clase correcta DONDE debia estar el bicho tapado
    bg_any_detection_at_box = 0  # cualquier deteccion (clase que sea) en esa zona

    per_class_crop = {c: {"correct": 0, "total": 0} for c in CLASS_NAMES}
    per_class_bg = {c: {"shortcut_hits": 0, "total": 0} for c in CLASS_NAMES}

    for inst in instances:
        img = cv2.imread(str(inst["image_path"]))
        box = inst["box"]
        true_name = CLASS_NAMES[inst["class_id"]]

        # A) solo bicho
        crop = crop_only(img, box, inst["img_w"], inst["img_h"])
        per_class_crop[true_name]["total"] += 1
        if crop.size == 0:
            crop_nodet += 1
        else:
            res = model.predict(crop, verbose=False, conf=CONF_THRESHOLD, device=device)[0]
            if len(res.boxes) == 0:
                crop_nodet += 1
            else:
                best = res.boxes[res.boxes.conf.argmax()]
                pred_name = CLASS_NAMES[int(best.cls[0])]
                if pred_name == true_name:
                    crop_correct += 1
                    per_class_crop[true_name]["correct"] += 1
                else:
                    crop_wrong += 1

        # B) solo fondo (bicho tapado)
        masked = background_only(img, box)
        per_class_bg[true_name]["total"] += 1
        res = model.predict(masked, verbose=False, conf=CONF_THRESHOLD, device=device)[0]
        if len(res.boxes) > 0:
            # cualquier deteccion cuyo centro caiga cerca de donde estaba el bicho tapado
            for b in res.boxes:
                pb = b.xyxy[0].tolist()
                if box_center_dist(pb, box) < max(box[2] - box[0], box[3] - box[1]):
                    bg_any_detection_at_box += 1
                    pred_name = CLASS_NAMES[int(b.cls[0])]
                    if pred_name == true_name:
                        bg_predicts_true_class += 1
                        per_class_bg[true_name]["shortcut_hits"] += 1
                    break

    n = len(instances)
    return {
        "n_instances": n,
        "crop_only": {
            "accuracy": crop_correct / n,
            "no_detection_rate": crop_nodet / n,
            "wrong_class_rate": crop_wrong / n,
        },
        "background_only": {
            "shortcut_rate": bg_predicts_true_class / n,  # clase correcta pese a no haber bicho
            "any_detection_rate": bg_any_detection_at_box / n,  # el modelo alucina algo ahi, sea la clase que sea
        },
        "per_class_crop_accuracy": {
            c: (v["correct"] / v["total"] if v["total"] else None) for c, v in per_class_crop.items()
        },
        "per_class_background_shortcut_rate": {
            c: (v["shortcut_hits"] / v["total"] if v["total"] else None) for c, v in per_class_bg.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name")
    parser.add_argument("model_path")
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/clean/test"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    instances = load_instances(args.data_dir)
    print(f"{len(instances)} instancias de test cargadas")

    model = YOLO(args.model_path)
    model.to(args.device)
    result = run_experiment(model, instances, args.device)
    print(json.dumps({k: v for k, v in result.items() if not k.startswith("per_class")}, indent=2))

    out_dir = Path("results") / args.experiment_name / "background_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "ablation.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    with open(out_dir / "per_class.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "crop_only_accuracy", "background_only_shortcut_rate"])
        for c in CLASS_NAMES:
            writer.writerow([
                c,
                result["per_class_crop_accuracy"][c],
                result["per_class_background_shortcut_rate"][c],
            ])

    print("ABLATION_DONE")


if __name__ == "__main__":
    main()
