"""
Auditoría estadística de un dataset YOLO: distribución de clases, geometría
de bounding boxes, integridad de anotaciones y fugas de datos entre splits.
"""

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from utils.logger import setup_logger

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
SPLIT_FOLDERS = ("train", "valid", "test")
SMALL_AREA_PX = 32 * 32
MEDIUM_AREA_PX = 96 * 96


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(sum(values) / len(values)), "min": float(min(values)), "max": float(max(values))}


class DatasetReport:
    """Genera un reporte de auditoría estadística para un dataset YOLO."""

    def __init__(self, data_yaml_path: str | Path):
        self.data_yaml_path = Path(data_yaml_path)
        self.dataset_path = self.data_yaml_path.parent
        self.logger = setup_logger("dataset_report")

    def generate(self, output_dir: str | Path = "results/dataset_report") -> dict[str, Any]:
        with open(self.data_yaml_path, encoding="utf-8") as f:
            data_yaml = yaml.safe_load(f)

        class_names = data_yaml.get("names", [])
        num_classes = data_yaml.get("nc", len(class_names))

        report: dict[str, Any] = {
            "dataset_path": str(self.dataset_path),
            "num_classes": num_classes,
            "class_names": class_names,
            "splits": {},
            "integrity_issues": {
                "malformed_boxes": [],
                "class_id_out_of_range": [],
                "images_without_labels": [],
                "labels_without_images": [],
            },
        }

        all_class_counts: dict[str, int] = defaultdict(int)
        image_hashes: dict[str, list[str]] = defaultdict(list)

        for split in SPLIT_FOLDERS:
            images_dir = self.dataset_path / split / "images"
            labels_dir = self.dataset_path / split / "labels"
            if not images_dir.exists():
                continue

            split_report, split_class_counts = self._audit_split(
                split, images_dir, labels_dir, num_classes, class_names, report["integrity_issues"]
            )
            report["splits"][split] = split_report
            for name, count in split_class_counts.items():
                all_class_counts[name] += count

            for image_path in sorted(images_dir.iterdir()):
                if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    digest = hashlib.md5(image_path.read_bytes()).hexdigest()
                    image_hashes[digest].append(f"{split}/images/{image_path.name}")

        report["class_imbalance_ratio"] = self._imbalance_ratio(all_class_counts)
        report["duplicate_leakage"] = self._find_cross_split_duplicates(image_hashes)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self._write_json(report, output_path / "report.json")
        self._write_markdown(report, output_path / "report.md")
        self._write_charts(report, output_path)

        self.logger.info(f"✅ Reporte de dataset generado en: {output_path}")
        return report

    def _audit_split(self, split, images_dir, labels_dir, num_classes, class_names, issues):
        class_counts: dict[str, int] = defaultdict(int)
        objects_per_image: list[int] = []
        bbox_aspect_ratios: list[float] = []
        size_buckets = {"small": 0, "medium": 0, "large": 0}

        image_files = {p.stem: p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}
        label_files = {p.stem: p for p in labels_dir.glob("*.txt")} if labels_dir.exists() else {}

        for stem, image_path in image_files.items():
            if stem not in label_files:
                issues["images_without_labels"].append(f"{split}/images/{image_path.name}")

        for stem, label_path in label_files.items():
            if stem not in image_files:
                issues["labels_without_images"].append(f"{split}/labels/{label_path.name}")
                continue

            image_path = image_files[stem]
            width_px = height_px = None
            try:
                from PIL import Image

                with Image.open(image_path) as img:
                    width_px, height_px = img.size
            except Exception as e:
                self.logger.warning(f"No se pudo leer dimensiones de {image_path}: {e}")

            n_objects = 0
            for line_no, raw_line in enumerate(label_path.read_text().splitlines(), start=1):
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    issues["malformed_boxes"].append(
                        {"split": split, "file": str(label_path), "line": line_no, "reason": "expected 5 fields"}
                    )
                    continue
                try:
                    class_id = int(parts[0])
                    cx, cy, w, h = (float(v) for v in parts[1:])
                except ValueError:
                    issues["malformed_boxes"].append(
                        {"split": split, "file": str(label_path), "line": line_no, "reason": "non-numeric values"}
                    )
                    continue

                if not (0 <= class_id < num_classes):
                    issues["class_id_out_of_range"].append(
                        {"split": split, "file": str(label_path), "line": line_no, "class_id": class_id}
                    )
                    continue

                if w <= 0 or h <= 0 or not (0 <= cx <= 1) or not (0 <= cy <= 1):
                    issues["malformed_boxes"].append(
                        {
                            "split": split,
                            "file": str(label_path),
                            "line": line_no,
                            "reason": "coordinates out of [0,1] or non-positive size",
                        }
                    )
                    continue

                class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                class_counts[class_name] += 1
                n_objects += 1
                bbox_aspect_ratios.append(w / h)

                if width_px and height_px:
                    area_px = (w * width_px) * (h * height_px)
                    if area_px < SMALL_AREA_PX:
                        size_buckets["small"] += 1
                    elif area_px < MEDIUM_AREA_PX:
                        size_buckets["medium"] += 1
                    else:
                        size_buckets["large"] += 1

            objects_per_image.append(n_objects)

        split_report = {
            "num_images": len(image_files),
            "num_labels": len(label_files),
            "num_instances": sum(class_counts.values()),
            "class_counts": dict(class_counts),
            "objects_per_image": _stats(objects_per_image),
            "bbox_size_buckets": size_buckets,
            "bbox_aspect_ratio": _stats(bbox_aspect_ratios),
        }
        return split_report, class_counts

    def _imbalance_ratio(self, class_counts: dict[str, int]) -> float:
        counts = [c for c in class_counts.values() if c > 0]
        if not counts:
            return 0.0
        return round(max(counts) / min(counts), 2)

    def _find_cross_split_duplicates(self, image_hashes: dict[str, list[str]]) -> list[dict[str, Any]]:
        leaks = []
        for digest, paths in image_hashes.items():
            splits_involved = {p.split("/")[0] for p in paths}
            if len(splits_involved) > 1:
                leaks.append({"md5": digest, "splits": sorted(splits_involved), "files": paths})
        return leaks

    def _write_json(self, report: dict[str, Any], path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    def _write_markdown(self, report: dict[str, Any], path: Path) -> None:
        lines = [
            f"# Reporte de Dataset: {report['dataset_path']}",
            "",
            f"- Clases: {report['num_classes']}",
            f"- Ratio de desbalance (clase mayor/menor): {report['class_imbalance_ratio']}",
            f"- Duplicados exactos entre splits: {len(report['duplicate_leakage'])}",
            "",
            "## Por split",
            "",
        ]
        for split, data in report["splits"].items():
            lines.append(f"### {split}")
            lines.append(
                f"- Imágenes: {data['num_images']} | Labels: {data['num_labels']} | "
                f"Instancias: {data['num_instances']}"
            )
            lines.append(f"- Distribución de clases: {data['class_counts']}")
            lines.append(f"- Objetos por imagen: {data['objects_per_image']}")
            lines.append(f"- Tamaño de bbox (buckets COCO): {data['bbox_size_buckets']}")
            lines.append("")

        issues = report["integrity_issues"]
        lines.append("## Integridad de anotaciones")
        lines.append(f"- Boxes malformados: {len(issues['malformed_boxes'])}")
        lines.append(f"- class_id fuera de rango: {len(issues['class_id_out_of_range'])}")
        lines.append(f"- Imágenes sin label: {len(issues['images_without_labels'])}")
        lines.append(f"- Labels sin imagen: {len(issues['labels_without_images'])}")

        if report["duplicate_leakage"]:
            lines.append("")
            lines.append("## ⚠️ Posible fuga de datos (duplicados exactos entre splits)")
            for leak in report["duplicate_leakage"]:
                lines.append(f"- {leak['splits']}: {leak['files']}")

        path.write_text("\n".join(lines), encoding="utf-8")

    def _write_charts(self, report: dict[str, Any], output_dir: Path) -> None:
        all_classes = report["class_names"]
        totals: dict[str, int] = defaultdict(int)
        for data in report["splits"].values():
            for name, count in data["class_counts"].items():
                totals[name] += count

        if not totals:
            return

        names = [n for n in all_classes if n in totals] or list(totals.keys())
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(names, [totals[n] for n in names])
        ax.set_ylabel("Instancias")
        ax.set_title("Distribución de clases (todos los splits)")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(output_dir / "class_distribution.png")
        plt.close(fig)
