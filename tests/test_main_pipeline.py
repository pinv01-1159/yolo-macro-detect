import json
from pathlib import Path

import pytest

from main import MacroinvertebratePipeline


class FakeTrainer:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name

    def load_model(self, model_name):
        pass

    def train(self, **kwargs):
        return "fake/best.pt"

    def evaluate(self, model_path, data_yaml_path):
        return object()

    def get_training_summary(self):
        return {
            "experiment_name": self.experiment_name,
            "evaluated_split": "test",
            "metrics": {"map50": 0.9, "map50_95": 0.8, "precision": 0.95, "recall": 0.9},
        }


def test_train_model_persists_eval_metrics_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    monkeypatch.setattr("main.YOLOTrainer", FakeTrainer)

    class FakeModelReport:
        def __init__(self, model_path, data_yaml_path):
            pass

        def generate(self, metrics=None, output_dir=None, protocol=None):
            return {}

    monkeypatch.setattr("main.ModelReport", FakeModelReport)

    pipeline = MacroinvertebratePipeline()
    pipeline.train_model(data_yaml_path="fake_data.yaml", experiment_name="exp1")

    metrics_file = tmp_path / "results" / "exp1" / "eval_metrics.json"
    assert metrics_file.exists()
    saved = json.loads(metrics_file.read_text())
    assert saved["metrics"]["map50"] == 0.9
    assert saved["evaluated_split"] == "test"


def test_train_model_generates_model_report_reusing_eval_metrics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    monkeypatch.setattr("main.YOLOTrainer", FakeTrainer)

    generated = {}

    class FakeModelReport:
        def __init__(self, model_path, data_yaml_path):
            generated["model_path"] = model_path
            generated["data_yaml_path"] = data_yaml_path

        def generate(self, metrics=None, output_dir=None, protocol=None):
            generated["metrics"] = metrics
            generated["output_dir"] = output_dir
            return {}

    monkeypatch.setattr("main.ModelReport", FakeModelReport)

    pipeline = MacroinvertebratePipeline()
    pipeline.train_model(data_yaml_path="fake_data.yaml", experiment_name="exp1")

    assert generated["model_path"] == "fake/best.pt"
    assert generated["metrics"] is not None
    assert str(generated["output_dir"]) == str(tmp_path / "results" / "exp1" / "model_report")


def test_predict_image_passes_iou_threshold_through(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    monkeypatch.setattr("main.YOLOTrainer", FakeTrainer)

    calls = {}

    class FakeInference:
        def __init__(self, model_path):
            calls["model_path"] = model_path

        def predict_image(self, **kwargs):
            calls["predict_kwargs"] = kwargs
            return {"detecciones": [], "total_detecciones": 0, "familias_detectadas": 0}

        def export_results(self, results, output_file):
            pass

    monkeypatch.setattr("main.YOLOInference", FakeInference)

    pipeline = MacroinvertebratePipeline()
    pipeline.predict_image(
        image_path="fake.jpg",
        model_path="fake.pt",
        iou_threshold=0.5,
    )

    assert calls["predict_kwargs"]["iou_threshold"] == 0.5


def test_cli_dataset_report_flag_invokes_report(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    monkeypatch.setattr("main.YOLOTrainer", FakeTrainer)

    calls = {}

    class FakeDatasetReport:
        def __init__(self, data_yaml_path):
            calls["data_yaml_path"] = data_yaml_path

        def generate(self):
            calls["generated"] = True

    monkeypatch.setattr("main.DatasetReport", FakeDatasetReport)
    monkeypatch.setattr("sys.argv", ["main.py", "--dataset-report", "--data-yaml", "fake.yaml"])

    import main
    main.main()

    assert calls == {"data_yaml_path": "fake.yaml", "generated": True}


def test_predict_site_agrega_el_indice_del_sitio(tmp_path, monkeypatch):
    """El BMWP se define por sitio de muestreo, no por fotografía.

    Es el camino que usan los biólogos: hasta ahora `predict_batch` existía
    pero no tenía ningún llamador ni flag de CLI, así que el caso de uso real
    del proyecto no era alcanzable sin escribir Python.
    """
    from PIL import Image

    monkeypatch.chdir(tmp_path)
    sitio = tmp_path / "arroyo"
    sitio.mkdir()
    for n in range(3):
        Image.new("RGB", (64, 64)).save(sitio / f"foto{n}.jpg")
    (tmp_path / "modelo.pt").write_bytes(b"fake")

    class FakeInference:
        def __init__(self, model_path):
            self.model_path = model_path
            self.recibidas = None

        def predict_batch(self, image_paths, **kwargs):
            self.recibidas = image_paths
            assert kwargs["calculate_bmwp"] is True
            return [{"detecciones": []} for _ in image_paths] + [
                {"alcance": "sitio", "bmwp_total": 17, "aspt": 2.83,
                 "n_familias_puntuadas": 6, "calidad_agua": "Crítica (Clase IV)",
                 "familias": [], "advertencias": [], "n_imagenes": len(image_paths)}
            ]

        def export_results(self, resultados, output_file):
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            Path(output_file).write_text("{}")

    monkeypatch.setattr("main.YOLOInference", FakeInference)
    pipeline = MacroinvertebratePipeline()

    resultado = pipeline.predict_site(
        images_dir=str(sitio), model_path=str(tmp_path / "modelo.pt"),
        save_annotated=False, output_dir=str(tmp_path / "out"),
    )

    assert resultado["alcance"] == "sitio"
    assert resultado["n_imagenes"] == 3
    assert resultado["bmwp_total"] == 17


def test_predict_site_falla_claro_sin_imagenes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    vacia = tmp_path / "vacia"
    vacia.mkdir()
    pipeline = MacroinvertebratePipeline()

    with pytest.raises(FileNotFoundError, match="No hay imágenes"):
        pipeline.predict_site(images_dir=str(vacia), model_path="x.pt")

    with pytest.raises(NotADirectoryError, match="No es una carpeta"):
        pipeline.predict_site(images_dir=str(tmp_path / "nope"), model_path="x.pt")
