"""
Reporte de evaluación de un modelo YOLO: métricas por clase, intervalos de
confianza por bootstrap, barrido de umbral de confianza y benchmark de latencia.
"""

import csv
import json
import shutil
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

from utils.logger import setup_logger

CONFIDENCE_SWEEP = (0.1, 0.3, 0.5, 0.7, 0.9)
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 42
CURVE_FILES = (
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "BoxPR_curve.png",
    "BoxF1_curve.png",
    "BoxP_curve.png",
    "BoxR_curve.png",
)


class ModelReport:
    """Genera un reporte de evaluación estadística para un modelo YOLO entrenado."""

    def __init__(self, model_path: str | Path, data_yaml_path: str | Path):
        self.model_path = Path(model_path)
        self.data_yaml_path = Path(data_yaml_path)
        self.logger = setup_logger("model_report")
        self._last_model: Any = None

    def generate(self,
                 metrics: Any = None,
                 output_dir: str | Path = "results/model_report") -> dict[str, Any]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        model = YOLO(str(self.model_path))
        self._last_model = model

        if metrics is None:
            metrics = model.val(data=str(self.data_yaml_path), split="test", plots=True, verbose=False)

        self._copy_curve_plots(metrics, output_path)

        report: dict[str, Any] = {
            "model_path": str(self.model_path),
            "data_yaml_path": str(self.data_yaml_path),
            "overall": {
                "map50": float(metrics.box.map50),
                "map50_95": float(metrics.box.map),
                "precision": float(metrics.box.mp),
                "recall": float(metrics.box.mr),
            },
            "per_class": self._per_class_table(metrics, model.names),
        }

        image_metrics = getattr(metrics.box, "image_metrics", None)
        if image_metrics:
            report["bootstrap_ci_95"] = self._bootstrap_ci(image_metrics)

        report["confidence_sweep"] = self._confidence_sweep(model)
        report["latency_ms"] = self._latency_benchmark(model)

        self._write_per_class_csv(report["per_class"], output_path / "per_class_metrics.csv")
        self._write_confidence_sweep_csv(report["confidence_sweep"], output_path / "confidence_sweep.csv")
        self._write_json(report, output_path / "report.json")
        self._write_markdown(report, output_path / "report.md")
        self._write_confidence_sweep_chart(report["confidence_sweep"], output_path)

        self.logger.info(f"✅ Reporte de modelo generado en: {output_path}")
        return report

    def _per_class_table(self, metrics: Any, names: dict[int, str]) -> list[dict[str, Any]]:
        box = metrics.box
        table = []
        for i, class_index in enumerate(box.ap_class_index):
            table.append({
                "class": names.get(int(class_index), str(class_index)),
                "precision": float(box.p[i]),
                "recall": float(box.r[i]),
                "f1": float(box.f1[i]),
                "ap50": float(box.ap50[i]),
                "ap50_95": float(box.ap[i]),
            })
        return table

    def _bootstrap_ci(self, image_metrics: dict[str, dict[str, float]]) -> dict[str, Any]:
        precisions = np.array([m["precision"] for m in image_metrics.values()])
        recalls = np.array([m["recall"] for m in image_metrics.values()])
        f1s = np.array([m["f1"] for m in image_metrics.values()])

        rng = np.random.default_rng(BOOTSTRAP_SEED)
        n = len(precisions)
        boot_p, boot_r, boot_f1 = [], [], []
        for _ in range(BOOTSTRAP_RESAMPLES):
            idx = rng.integers(0, n, size=n)
            boot_p.append(precisions[idx].mean())
            boot_r.append(recalls[idx].mean())
            boot_f1.append(f1s[idx].mean())

        def _ci(values: list[float]) -> dict[str, float]:
            arr = np.array(values)
            return {
                "mean": float(arr.mean()),
                "ci_low": float(np.percentile(arr, 2.5)),
                "ci_high": float(np.percentile(arr, 97.5)),
            }

        return {"precision": _ci(boot_p), "recall": _ci(boot_r), "f1": _ci(boot_f1), "n_images": n}

    def _confidence_sweep(self, model: Any) -> list[dict[str, float]]:
        sweep = []
        for conf in CONFIDENCE_SWEEP:
            metrics = model.val(
                data=str(self.data_yaml_path), split="test", conf=conf, plots=False, verbose=False
            )
            precision = float(metrics.box.mp)
            recall = float(metrics.box.mr)
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            sweep.append({"confidence": conf, "precision": precision, "recall": recall, "f1": f1})
        return sweep

    def _latency_benchmark(self, model: Any, max_images: int = 50) -> dict[str, Any]:
        test_images_dir = self.data_yaml_path.parent / "test" / "images"
        if not test_images_dir.exists():
            return {"mean": 0.0, "median": 0.0, "p95": 0.0, "device": "unknown", "n_images": 0}

        image_paths = sorted(test_images_dir.iterdir())[:max_images]
        durations_ms = []
        for image_path in image_paths:
            start = time.perf_counter()
            model.predict(str(image_path), verbose=False)
            durations_ms.append((time.perf_counter() - start) * 1000)

        arr = np.array(durations_ms) if durations_ms else np.array([0.0])
        device = str(next(model.model.parameters()).device) if hasattr(model, "model") else "unknown"
        return {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "device": device,
            "n_images": len(durations_ms),
        }

    def _copy_curve_plots(self, metrics: Any, output_path: Path) -> None:
        save_dir = getattr(metrics, "save_dir", None)
        if not save_dir:
            return
        for filename in CURVE_FILES:
            source = Path(save_dir) / filename
            if source.exists():
                shutil.copy(source, output_path / filename)

    def _write_per_class_csv(self, per_class: list[dict[str, Any]], path: Path) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["class", "precision", "recall", "f1", "ap50", "ap50_95"])
            writer.writeheader()
            writer.writerows(per_class)

    def _write_confidence_sweep_csv(self, sweep: list[dict[str, float]], path: Path) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["confidence", "precision", "recall", "f1"])
            writer.writeheader()
            writer.writerows(sweep)

    def _write_json(self, report: dict[str, Any], path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    def _write_markdown(self, report: dict[str, Any], path: Path) -> None:
        overall = report["overall"]
        lines = [
            f"# Reporte de Modelo: {report['model_path']}",
            "",
            f"- mAP50: {overall['map50']:.4f}",
            f"- mAP50-95: {overall['map50_95']:.4f}",
            f"- Precisión: {overall['precision']:.4f}",
            f"- Recall: {overall['recall']:.4f}",
            "",
        ]

        if "bootstrap_ci_95" in report:
            ci = report["bootstrap_ci_95"]
            lines.append(f"## Intervalo de confianza 95% (bootstrap, n={ci['n_images']} imágenes)")
            for metric_name in ("precision", "recall", "f1"):
                m = ci[metric_name]
                lines.append(f"- {metric_name}: {m['mean']:.3f} [{m['ci_low']:.3f}, {m['ci_high']:.3f}]")
            lines.append("")

        lines.append("## Métricas por clase")
        for row in report["per_class"]:
            lines.append(
                f"- {row['class']}: P={row['precision']:.3f} R={row['recall']:.3f} "
                f"F1={row['f1']:.3f} AP50={row['ap50']:.3f} AP50-95={row['ap50_95']:.3f}"
            )

        lines.append("")
        lines.append("## Barrido de umbral de confianza")
        for point in report["confidence_sweep"]:
            lines.append(
                f"- conf={point['confidence']}: P={point['precision']:.3f} "
                f"R={point['recall']:.3f} F1={point['f1']:.3f}"
            )

        latency = report["latency_ms"]
        lines.append("")
        lines.append("## Latencia de inferencia")
        lines.append(
            f"- media={latency['mean']:.1f}ms mediana={latency['median']:.1f}ms "
            f"p95={latency['p95']:.1f}ms device={latency['device']} (n={latency['n_images']})"
        )

        path.write_text("\n".join(lines), encoding="utf-8")

    def _write_confidence_sweep_chart(self, sweep: list[dict[str, float]], output_dir: Path) -> None:
        if not sweep:
            return
        fig, ax = plt.subplots(figsize=(8, 5))
        confidences = [s["confidence"] for s in sweep]
        ax.plot(confidences, [s["precision"] for s in sweep], marker="o", label="Precisión")
        ax.plot(confidences, [s["recall"] for s in sweep], marker="o", label="Recall")
        ax.plot(confidences, [s["f1"] for s in sweep], marker="o", label="F1")
        ax.set_xlabel("Umbral de confianza")
        ax.set_ylabel("Score")
        ax.set_title("Precisión/Recall/F1 vs. umbral de confianza")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "confidence_sweep.png")
        plt.close(fig)
