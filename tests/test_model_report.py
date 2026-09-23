import json

import numpy as np
import pytest
from PIL import Image

from reports.model_report import ModelReport


class FakeBox:
    def __init__(self):
        self.map50 = 0.9
        self.map = 0.8
        self.mp = 0.95
        self.mr = 0.9
        self.p = np.array([0.95, 0.9])
        self.r = np.array([0.9, 0.85])
        self.f1 = np.array([0.92, 0.87])
        self.ap50 = np.array([0.95, 0.85])
        self.ap = np.array([0.85, 0.75])
        self.ap_class_index = np.array([0, 1])
        self.image_metrics = {
            "img1.jpg": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 2, "fp": 0, "fn": 0},
            "img2.jpg": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "tp": 1, "fp": 1, "fn": 1},
        }


class FakeMetrics:
    def __init__(self):
        self.box = FakeBox()
        self.save_dir = None


class FakeYoloModel:
    def __init__(self, *_args, **_kwargs):
        self.names = {0: "Physidae", 1: "Chironomidae"}
        self.val_calls = []

    def val(self, **kwargs):
        self.val_calls.append(kwargs)
        return FakeMetrics()

    def predict(self, *_args, **_kwargs):
        return []


def _make_data_yaml(tmp_path):
    (tmp_path / "test" / "images").mkdir(parents=True)
    Image.new("RGB", (50, 50)).save(tmp_path / "test" / "images" / "img1.jpg")
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        "path: " + str(tmp_path) + "\n"
        "test: test/images\n"
        "nc: 2\n"
        "names: [Physidae, Chironomidae]\n"
    )
    return data_yaml


def _make_model_report(tmp_path, monkeypatch):
    monkeypatch.setattr("reports.model_report.YOLO", FakeYoloModel)
    data_yaml = _make_data_yaml(tmp_path)
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"fake")
    return ModelReport(model_path, data_yaml)


def test_generate_reuses_provided_metrics_without_revalidating(tmp_path, monkeypatch):
    report_obj = _make_model_report(tmp_path, monkeypatch)

    report_obj.generate(metrics=FakeMetrics(), output_dir=tmp_path / "report_out")

    # Solo debe llamarse val() para el barrido de confianza (5 puntos), no
    # para la evaluación completa (que ya se pasó precalculada).
    assert len(report_obj._last_model.val_calls) == 5


def test_generate_evaluates_when_no_metrics_provided(tmp_path, monkeypatch):
    report_obj = _make_model_report(tmp_path, monkeypatch)

    report_obj.generate(output_dir=tmp_path / "report_out")

    assert len(report_obj._last_model.val_calls) == 6


def test_per_class_table_matches_box_arrays(tmp_path, monkeypatch):
    report_obj = _make_model_report(tmp_path, monkeypatch)

    report = report_obj.generate(metrics=FakeMetrics(), output_dir=tmp_path / "report_out")

    assert report["per_class"][0]["class"] == "Physidae"
    assert report["per_class"][0]["ap50"] == 0.95


def test_bootstrap_ci_bounds_contain_mean(tmp_path, monkeypatch):
    report_obj = _make_model_report(tmp_path, monkeypatch)

    report = report_obj.generate(metrics=FakeMetrics(), output_dir=tmp_path / "report_out")

    ci = report["bootstrap_ci_95"]["precision"]
    assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]
    assert report["bootstrap_ci_95"]["n_images"] == 2


def test_bootstrap_ci_markdown_disambiguates_from_overall_metrics(tmp_path, monkeypatch):
    report_obj = _make_model_report(tmp_path, monkeypatch)
    output_dir = tmp_path / "report_out"

    report_obj.generate(metrics=FakeMetrics(), output_dir=output_dir)

    md_content = (output_dir / "report.md").read_text(encoding="utf-8")
    assert "no es el mismo estimador que las métricas generales de arriba" in md_content


def test_output_files_created(tmp_path, monkeypatch):
    report_obj = _make_model_report(tmp_path, monkeypatch)
    output_dir = tmp_path / "report_out"

    report_obj.generate(metrics=FakeMetrics(), output_dir=output_dir)

    assert (output_dir / "report.json").exists()
    assert (output_dir / "report.md").exists()
    assert (output_dir / "per_class_metrics.csv").exists()
    assert (output_dir / "confidence_sweep.csv").exists()
    assert (output_dir / "confidence_sweep.png").exists()


def test_bootstrap_resamples_specimens_not_images(tmp_path, monkeypatch):
    """Cuatro fotos de dos especímenes deben contar como dos unidades.

    Es la corrección del hallazgo S1: remuestrear fotos trata tomas casi
    idénticas del mismo ejemplar como observaciones independientes y estrecha
    el intervalo de forma artificial.
    """
    report_obj = _make_model_report(tmp_path, monkeypatch)
    (tmp_path / "split_report.json").write_text(json.dumps({
        "group_of_image": {"a1": 0, "a2": 0, "b1": 1, "b2": 1},
    }))

    metrics = FakeMetrics()
    metrics.box.image_metrics = {
        "a1.jpg": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
        "a2.jpg": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
        "b1.jpg": {"precision": 0.5, "recall": 0.5, "f1": 0.5},
        "b2.jpg": {"precision": 0.5, "recall": 0.5, "f1": 0.5},
    }

    ci = report_obj.generate(metrics=metrics, output_dir=tmp_path / "out")["bootstrap_ci_95"]

    assert ci["resample_unit"] == "specimen"
    assert ci["n_units"] == 2
    assert ci["n_images"] == 4
    # cada espécimen aporta el promedio de sus fotos: 1.0 y 0.5 -> media 0.75
    assert ci["precision"]["mean"] == pytest.approx(0.75, abs=0.05)


def test_bootstrap_falls_back_to_images_without_split_report(tmp_path, monkeypatch):
    """Sin manifiesto no se puede agrupar; hay que decirlo, no fingir que sí."""
    report_obj = _make_model_report(tmp_path, monkeypatch)

    ci = report_obj.generate(
        metrics=FakeMetrics(), output_dir=tmp_path / "out"
    )["bootstrap_ci_95"]

    assert ci["resample_unit"] == "image"
    assert "subestimado" in ci["note"]


def test_confidence_sweep_runs_on_validation_not_test(tmp_path, monkeypatch):
    """El barrido sobre test convertiría el umbral en un hiperparámetro
    ajustado al conjunto de reporte (hallazgo B6)."""
    monkeypatch.setattr("reports.model_report.YOLO", FakeYoloModel)
    data_yaml = _make_data_yaml(tmp_path)
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"fake")
    report_obj = ModelReport(model_path, data_yaml)

    report = report_obj.generate(metrics=FakeMetrics(), output_dir=tmp_path / "out")

    sweep_splits = {c["split"] for c in report_obj._last_model.val_calls if "conf" in c}
    assert sweep_splits == {"val"}
    assert report["protocol"]["confidence_sweep"]["split"] == "val"


def test_protocol_registra_los_parametros_efectivos(tmp_path, monkeypatch):
    """Sin esto, dos corridas con protocolos distintos producen JSONs
    indistinguibles — la razón por la que B5 y B6 pasaron meses sin verse."""
    report_obj = _make_model_report(tmp_path, monkeypatch)

    report = report_obj.generate(
        metrics=FakeMetrics(),
        output_dir=tmp_path / "out",
        protocol={"split": "test", "conf": 0.001, "iou": 0.7, "imgsz": 640},
    )

    p = report["protocol"]["overall_and_per_class"]
    assert p == {"split": "test", "conf": 0.001, "iou": 0.7, "imgsz": 640}
    assert report["protocol"]["ultralytics_version"]


def test_protocol_cae_al_validador_si_las_metricas_no_traen_args(tmp_path, monkeypatch):
    """`metrics.args` viene en None en algunas versiones de Ultralytics; si el
    protocolo solo se leyera de ahí, quedaría vacío sin que nadie lo note."""
    report_obj = _make_model_report(tmp_path, monkeypatch)

    class FakeValidatorArgs:
        conf, iou, imgsz, split = 0.25, 0.65, 512, "val"

    class ModelConValidador(FakeYoloModel):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.validator = type("V", (), {"args": FakeValidatorArgs()})()

    monkeypatch.setattr("reports.model_report.YOLO", ModelConValidador)

    metrics = FakeMetrics()
    metrics.args = None  # como lo devuelve ultralytics 8.4.95
    report = report_obj.generate(metrics=metrics, output_dir=tmp_path / "out2")

    p = report["protocol"]["overall_and_per_class"]
    assert (p["conf"], p["iou"], p["imgsz"]) == (0.25, 0.65, 512)
