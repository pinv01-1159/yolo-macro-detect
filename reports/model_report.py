"""
Reporte de evaluación de un modelo YOLO: métricas por clase, intervalos de
confianza por bootstrap, barrido de umbral de confianza y benchmark de latencia.
"""

import csv
import json
import shutil
import subprocess
import time
from collections.abc import Sequence
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO

from utils.logger import setup_logger
from utils.runtime import pin_ultralytics_paths

CONFIDENCE_SWEEP = (0.1, 0.3, 0.5, 0.7, 0.9)
# El barrido corre sobre validacion: elegir el umbral mirando test lo convierte
# en un hiperparametro ajustado al conjunto de reporte (auditoria B6).
SWEEP_SPLIT = "val"
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 42
LATENCY_WARMUP = 10
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
        pin_ultralytics_paths()  # ver utils/runtime.py (auditoria S8)
        self.model_path = Path(model_path)
        self.data_yaml_path = Path(data_yaml_path)
        self.logger = setup_logger("model_report")
        self._last_model: Any = None
        self._protocol_overrides: dict[str, Any] = {}

    def generate(self,
                 metrics: Any = None,
                 output_dir: str | Path = "results/model_report",
                 protocol: dict[str, Any] | None = None) -> dict[str, Any]:
        """Genera el reporte.

        Args:
            metrics: métricas ya calculadas; si es None se evalúa acá.
            output_dir: destino de los artefactos.
            protocol: conf/iou/imgsz/split efectivos con que se produjo
                `metrics`. Conviene pasarlos cuando las métricas vienen de
                afuera: quien evaluó los conoce con certeza, mientras que
                deducirlos del objeto de métricas depende de la versión de
                Ultralytics.
        """
        self._protocol_overrides = dict(protocol or {})
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        model = YOLO(str(self.model_path))
        self._last_model = model

        if metrics is None:
            metrics = model.val(
                data=str(self.data_yaml_path), split="test", plots=True, verbose=False
            )

        self._copy_curve_plots(metrics, output_path)
        self._copy_raw_predictions(metrics, output_path)
        self._write_confusion_matrix_csv(metrics, model.names, output_path)

        report: dict[str, Any] = {
            "model_path": str(self.model_path),
            "data_yaml_path": str(self.data_yaml_path),
            # Sin esto, dos corridas del mismo modelo con protocolos distintos
            # producen JSONs indistinguibles. Es la razon por la que B5 y B6
            # pasaron desapercibidos durante meses (auditoria S9).
            "protocol": self._protocol(metrics),
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
        self._write_confidence_sweep_csv(
            report["confidence_sweep"], output_path / "confidence_sweep.csv"
        )
        self._write_json(report, output_path / "report.json")
        self._write_markdown(report, output_path / "report.md")
        self._write_confidence_sweep_chart(report["confidence_sweep"], output_path)

        self.logger.info(f"✅ Reporte de modelo generado en: {output_path}")
        return report

    def _protocol(self, metrics: Any) -> dict[str, Any]:
        """Deja por escrito con que se produjo cada numero de este reporte.

        `overall` y `per_class` salen de la evaluacion que se recibe o se corre
        aca; `confidence_sweep` sale de validacion (ver `_confidence_sweep`).
        Son protocolos distintos y no deben compararse entre si.
        """
        # Los parametros efectivos no viven en un solo lugar: segun la version
        # de Ultralytics `metrics.args` puede venir en None, y entonces hay que
        # leerlos del validador, que existe recien despues de val(). Se prueban
        # las tres fuentes en orden de confiabilidad.
        fuentes = [
            self._protocol_overrides,
            getattr(metrics, "args", None),
            getattr(getattr(self._last_model, "validator", None), "args", None),
        ]

        def _arg(name: str) -> Any:
            for fuente in fuentes:
                if fuente is None:
                    continue
                valor = (
                    fuente.get(name) if isinstance(fuente, dict)
                    else getattr(fuente, name, None)
                )
                if valor is not None:
                    return valor
            return None

        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True, timeout=5,
            ).stdout.strip()
        except (subprocess.SubprocessError, OSError):
            commit = None

        return {
            "overall_and_per_class": {
                "split": _arg("split") or "test",
                "conf": _arg("conf"),
                "iou": _arg("iou"),
                "imgsz": _arg("imgsz"),
            },
            "confidence_sweep": {"split": SWEEP_SPLIT, "thresholds": list(CONFIDENCE_SWEEP)},
            "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
            "ultralytics_version": version("ultralytics"),
            "git_commit": commit,
            "generated_at": datetime.now().astimezone().isoformat(),
        }

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

    def _load_group_of_image(self) -> dict[str, int]:
        """Mapa imagen -> especimen, desde el manifiesto del split.

        Vacio si no esta disponible: el bootstrap cae entonces a remuestreo por
        imagen, que subestima el error (ver `_bootstrap_ci`).
        """
        report_path = self.data_yaml_path.parent / "split_report.json"
        if not report_path.exists():
            return {}
        try:
            with open(report_path) as fh:
                return json.load(fh).get("group_of_image", {}) or {}
        except (json.JSONDecodeError, OSError):
            return {}

    def _bootstrap_ci(self, image_metrics: dict[str, dict[str, float]]) -> dict[str, Any]:
        """IC95 por bootstrap, remuestreando **especimenes** y no imagenes.

        El protocolo de captura toma varias fotos casi identicas de cada
        ejemplar, asi que las imagenes no son observaciones independientes.
        Remuestrearlas trata como informacion nueva lo que es la misma foto y
        estrecha el intervalo de forma artificial: medido sobre este conjunto,
        por un factor cercano a 1.5 (auditoria S1). La unidad independiente es
        el especimen: 87 en test, no 370.
        """
        group_of_image = self._load_group_of_image()

        # las claves de image_metrics son rutas; el manifiesto usa el stem
        def _group_for(key: str) -> Any:
            stem = Path(key).stem
            return group_of_image.get(stem, group_of_image.get(key, f"__img__{key}"))

        by_group: dict[Any, list[dict[str, float]]] = {}
        for key, m in image_metrics.items():
            by_group.setdefault(_group_for(key), []).append(m)

        grouped = bool(group_of_image) and len(by_group) < len(image_metrics)

        # cada especimen aporta el promedio de sus fotos: una foto extra de un
        # mismo ejemplar no debe pesar mas que un ejemplar entero
        def _mean(items: list[dict[str, float]], field: str) -> float:
            return float(np.mean([m[field] for m in items]))

        units = list(by_group.values())
        precisions = np.array([_mean(u, "precision") for u in units])
        recalls = np.array([_mean(u, "recall") for u in units])
        f1s = np.array([_mean(u, "f1") for u in units])

        rng = np.random.default_rng(BOOTSTRAP_SEED)
        n = len(units)
        boot_p, boot_r, boot_f1 = [], [], []
        for _ in range(BOOTSTRAP_RESAMPLES):
            idx = rng.integers(0, n, size=n)
            boot_p.append(precisions[idx].mean())
            boot_r.append(recalls[idx].mean())
            boot_f1.append(f1s[idx].mean())

        def _ci(values: Sequence[float]) -> dict[str, float]:
            arr = np.array(values)
            return {
                "mean": float(arr.mean()),
                "ci_low": float(np.percentile(arr, 2.5)),
                "ci_high": float(np.percentile(arr, 97.5)),
            }

        return {
            "precision": _ci(boot_p),
            "recall": _ci(boot_r),
            "f1": _ci(boot_f1),
            "resample_unit": "specimen" if grouped else "image",
            "n_units": n,
            "n_images": len(image_metrics),
            "note": (
                "Remuestreo por especimen: la unidad independiente es el ejemplar, "
                "no la foto." if grouped else
                "Sin split_report.json: remuestreo por imagen, el IC esta "
                "subestimado (ver auditoria S1)."
            ),
        }

    def _confidence_sweep(self, model: Any) -> list[dict[str, float]]:
        """Barrido de umbral sobre **validacion**, nunca sobre test.

        Elegir el punto de operacion mirando la curva sobre el conjunto de
        reporte es seleccionar un hiperparametro en test: deja de ser held-out y
        el numero publicado pasa a ser una cota optimista en vez de un
        resultado. El umbral que sale de aca se aplica a test una sola vez.
        """
        sweep = []
        for conf in CONFIDENCE_SWEEP:
            metrics = model.val(
                data=str(self.data_yaml_path),
                split=SWEEP_SPLIT,
                conf=conf,
                plots=False,
                verbose=False,
            )
            precision = float(metrics.box.mp)
            recall = float(metrics.box.mr)
            f1 = (
                (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            )
            sweep.append({"confidence": conf, "precision": precision, "recall": recall, "f1": f1})
        return sweep

    def _latency_benchmark(self, model: Any, max_images: int = 50) -> dict[str, Any]:
        test_images_dir = self.data_yaml_path.parent / "test" / "images"
        if not test_images_dir.exists():
            return {"mean": 0.0, "median": 0.0, "p95": 0.0, "device": "unknown", "n_images": 0}

        image_paths = sorted(test_images_dir.iterdir())[:max_images]
        if not image_paths:
            return {"mean": 0.0, "median": 0.0, "p95": 0.0, "device": "unknown", "n_images": 0}

        device = (
            str(next(model.model.parameters()).device) if hasattr(model, "model") else "unknown"
        )
        on_cuda = device.startswith("cuda")

        # Sin warm-up las primeras iteraciones miden la compilacion de kernels y
        # la reserva de memoria, no la inferencia.
        for image_path in image_paths[:LATENCY_WARMUP]:
            model.predict(str(image_path), verbose=False)
        if on_cuda:
            torch.cuda.synchronize()

        # Ultralytics ya separa preprocess/inference/postprocess en
        # results.speed; cronometrar el predict() completo mide ademas la
        # lectura de disco y la decodificacion JPEG, que no son latencia de
        # inferencia (auditoria S10).
        inference_ms, e2e_ms = [], []
        for image_path in image_paths:
            start = time.perf_counter()
            results = model.predict(str(image_path), verbose=False)
            if on_cuda:
                torch.cuda.synchronize()
            e2e_ms.append((time.perf_counter() - start) * 1000)
            speed = getattr(results[0], "speed", None) if results else None
            if speed:
                inference_ms.append(float(speed.get("inference", 0.0)))

        def _stats(values: list[float]) -> dict[str, float]:
            arr = np.array(values) if values else np.array([0.0])
            return {
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "p95": float(np.percentile(arr, 95)),
            }

        return {
            # `mean`/`median`/`p95` siguen siendo end-to-end por compatibilidad
            # con los reportes ya generados; `inference` es la cifra citable.
            **_stats(e2e_ms),
            "inference": _stats(inference_ms),
            "end_to_end": _stats(e2e_ms),
            "device": device,
            "imgsz": getattr(model, "overrides", {}).get("imgsz"),
            "batch": 1,
            "warmup_iters": min(LATENCY_WARMUP, len(image_paths)),
            "cuda_synchronized": on_cuda,
            "n_images": len(e2e_ms),
            "note": (
                "`inference` excluye I/O y decodificacion; `end_to_end` las incluye. "
                "Solo la primera es comparable entre modelos."
            ),
        }

    def _copy_curve_plots(self, metrics: Any, output_path: Path) -> None:
        save_dir = getattr(metrics, "save_dir", None)
        if not save_dir:
            return
        for filename in CURVE_FILES:
            source = Path(save_dir) / filename
            if source.exists():
                shutil.copy(source, output_path / filename)

    def _copy_raw_predictions(self, metrics: Any, output_path: Path) -> None:
        """Copia predictions.json (cajas + confianza por imagen, generado por
        val(save_json=True)) si está disponible, para tener el dato crudo
        detrás de las métricas agregadas."""
        save_dir = getattr(metrics, "save_dir", None)
        if not save_dir:
            return
        source = Path(save_dir) / "predictions.json"
        if source.exists():
            shutil.copy(source, output_path / "predictions_raw.json")

    def _write_confusion_matrix_csv(
        self, metrics: Any, names: dict[int, str], output_path: Path
    ) -> None:
        """Vuelca la matriz de confusión cruda (no solo el PNG) con nombres
        de clase. Convención de Ultralytics: matrix[predicha, real]
        (filas = clase predicha, columnas = clase real), con una fila/columna
        extra "background" para FP/FN sin contraparte."""
        confusion_matrix = getattr(metrics, "confusion_matrix", None)
        matrix = getattr(confusion_matrix, "matrix", None)
        if matrix is None:
            return

        labels = [names.get(i, str(i)) for i in range(len(names))] + ["background"]
        with open(output_path / "confusion_matrix_raw.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["predicha\\real", *labels])
            for i, row in enumerate(matrix):
                writer.writerow([labels[i], *[int(v) for v in row]])

    def _write_per_class_csv(self, per_class: list[dict[str, Any]], path: Path) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["class", "precision", "recall", "f1", "ap50", "ap50_95"]
            )
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
            lines.append(
                "## Intervalo de confianza 95% — precisión/recall/F1 promediados por "
                f"imagen a IoU=0.5 (bootstrap, n={ci['n_images']} imágenes) — no es el "
                "mismo estimador que las métricas generales de arriba"
            )
            for metric_name in ("precision", "recall", "f1"):
                m = ci[metric_name]
                lines.append(
                    f"- {metric_name}: {m['mean']:.3f} [{m['ci_low']:.3f}, {m['ci_high']:.3f}]"
                )
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

    def _write_confidence_sweep_chart(
        self, sweep: list[dict[str, float]], output_dir: Path
    ) -> None:
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
