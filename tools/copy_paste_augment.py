"""
Augmentación copy-paste "Same Y" contra el confusor de sesión.

Problema que ataca (ver docs/leakage_analysis.md sección 3): 12 de 19
familias se fotografiaron en una sola sesión, así que el fondo/iluminación
de laboratorio queda correlacionado con la clase. build_clean_split.py
arregla la fuga por ráfaga, pero no arregla esto -- es estructural del
dataset, no del split.

Técnica (Gao et al. 2023, "Out-of-Domain Robustness via Targeted
Augmentations", arXiv:2302.11861 -- "Copy-Paste (Same Y)"): para cada
instancia de train, recortar el bicho de su caja y pegarlo sobre el fondo
de OTRA imagen de train (de otro grupo/ráfaga), conservando la misma
etiqueta. Esto randomiza el fondo espurio sin tocar la morfología real.

A diferencia del paper original (cámaras trampa, donde el fondo también
lleva señal legítima de hábitat), acá el fondo es una bandeja de
laboratorio sin señal ecológica real -- así que el donante de fondo se
elige de cualquier clase, no solo de la misma, para maximizar la ruptura
de la correlación espuria en las clases de sesión única (que no tienen
ninguna otra sesión propia de la que tomar un fondo distinto).

Genera UN dataset nuevo (no modifica datasets/clean/): copia valid/ y
test/ tal cual (nunca se aumentan, solo train/), y agrega a train/ una
imagen sintética por cada instancia real (aprox. duplica el tamaño de
train, mismo balance de clases que el original).

Uso:
    python tools/copy_paste_augment.py datasets/clean datasets/clean_cpaug
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import yaml

PAD_COLOR = 114  # mismo gris neutro que usa Ultralytics para padding/letterbox


def read_yolo_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls_id, cx, cy, w, h = int(parts[0]), *map(float, parts[1:5])
        boxes.append((cls_id, cx, cy, w, h))
    return boxes


def norm_to_pixels(box: tuple[float, float, float, float], img_w: int, img_h: int) -> tuple[int, int, int, int]:
    cx, cy, w, h = box
    x1 = max(0, int((cx - w / 2) * img_w))
    y1 = max(0, int((cy - h / 2) * img_h))
    x2 = min(img_w, int((cx + w / 2) * img_w))
    y2 = min(img_h, int((cy + h / 2) * img_h))
    return x1, y1, x2, y2


def erase_boxes(img, boxes: list[tuple[int, float, float, float, float]]):
    h, w = img.shape[:2]
    out = img.copy()
    for _, cx, cy, bw, bh in boxes:
        x1, y1, x2, y2 = norm_to_pixels((cx, cy, bw, bh), w, h)
        out[y1:y2, x1:x2] = PAD_COLOR
    return out


def copy_train(src_dir: Path, dst_dir: Path) -> None:
    import shutil
    for split in ("train", "valid", "test"):
        (dst_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (dst_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        src_split = src_dir / split
        if split == "train":
            # train se copia + se le agregan sinteticas despues
            for f in (src_split / "images").glob("*"):
                shutil.copy(f, dst_dir / split / "images" / f.name)
            for f in (src_split / "labels").glob("*.txt"):
                shutil.copy(f, dst_dir / split / "labels" / f.name)
        else:
            # valid/test NUNCA se tocan: son la medicion honesta
            shutil.copytree(src_split / "images", dst_dir / split / "images", dirs_exist_ok=True)
            shutil.copytree(src_split / "labels", dst_dir / split / "labels", dirs_exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--multiplier", type=int, default=1,
        help=(
            "variantes sinteticas por instancia real, cada una con un donante "
            "distinto. Con multiplier=1 las sinteticas quedan en minoria frente "
            "a las reales sin tocar, y el atajo de fondo sigue disponible en la "
            "mayoria de los datos (verificado empiricamente: no bajo la tasa de "
            "atajo). Con multiplier>=3 las sinteticas pasan a ser mayoria."
        ),
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    split_report = json.loads((args.src / "split_report.json").read_text())
    group_of_image = split_report["group_of_image"]

    meta = yaml.safe_load((args.src / "data.yaml").read_text())
    names = meta["names"]

    print("Copiando dataset base...")
    copy_train(args.src, args.dst)

    train_images_dir = args.src / "train" / "images"
    train_labels_dir = args.src / "train" / "labels"
    out_images_dir = args.dst / "train" / "images"
    out_labels_dir = args.dst / "train" / "labels"

    image_files = sorted(train_images_dir.glob("*.jpg"))
    # pool de imagenes donantes: cualquier imagen de train, con su grupo
    donor_pool = [(f, group_of_image.get(f.stem)) for f in image_files]

    per_class_generated: dict[str, int] = {n: 0 for n in names}
    skipped_degenerate = 0
    n_generated = 0

    for img_path in image_files:
        label_path = train_labels_dir / f"{img_path.stem}.txt"
        boxes = read_yolo_labels(label_path)
        if not boxes:
            continue
        source_group = group_of_image.get(img_path.stem)
        source_img = cv2.imread(str(img_path))
        if source_img is None:
            continue
        sh, sw = source_img.shape[:2]

        for inst_idx, box in enumerate(boxes):
            cls_id, cx, cy, bw, bh = box
            x1, y1, x2, y2 = norm_to_pixels((cx, cy, bw, bh), sw, sh)
            if x2 <= x1 or y2 <= y1:
                skipped_degenerate += 1
                continue
            crop = source_img[y1:y2, x1:x2]

            used_donors: set = set()
            for variant in range(args.multiplier):
                # elegir donante de OTRO grupo (sesion/rafaga distinta), sin repetir
                # donante entre variantes de la misma instancia
                donor_path = donor_group = None
                for _ in range(20):
                    cand_path, cand_group = rng.choice(donor_pool)
                    if cand_group != source_group and cand_path not in used_donors:
                        donor_path, donor_group = cand_path, cand_group
                        break
                if donor_path is None:
                    continue  # no se encontro donante valido, saltar esta variante
                used_donors.add(donor_path)

                donor_img = cv2.imread(str(donor_path))
                if donor_img is None:
                    continue
                dh, dw = donor_img.shape[:2]
                if (dh, dw) != (sh, sw):
                    donor_img = cv2.resize(donor_img, (sw, sh))

                donor_boxes = read_yolo_labels(train_labels_dir / f"{donor_path.stem}.txt")
                synthetic = erase_boxes(donor_img, donor_boxes)
                synthetic[y1:y2, x1:x2] = crop  # mismas coords relativas: mismo box normalizado sirve tal cual

                out_name = f"cpaug_{img_path.stem}_{inst_idx}_v{variant}"
                cv2.imwrite(str(out_images_dir / f"{out_name}.jpg"), synthetic)
                (out_labels_dir / f"{out_name}.txt").write_text(f"{cls_id} {cx} {cy} {bw} {bh}\n")

                per_class_generated[names[cls_id]] += 1
                n_generated += 1

    (args.dst / "data.yaml").write_text(
        yaml.dump(
            {
                "path": str(args.dst.resolve()),
                "train": "train/images",
                "val": "valid/images",
                "test": "test/images",
                "nc": len(names),
                "names": names,
            },
            sort_keys=False,
        )
    )

    print(f"\n{n_generated} imágenes sintéticas generadas ({skipped_degenerate} cajas degeneradas saltadas)")
    print("por clase:")
    for n in names:
        print(f"  {n}: +{per_class_generated[n]}")
    print(f"\ntrain total ahora: {len(list(out_images_dir.glob('*.jpg')))} imágenes")


if __name__ == "__main__":
    main()
