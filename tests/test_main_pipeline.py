import json

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

        def generate(self, metrics=None, output_dir=None):
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

        def generate(self, metrics=None, output_dir=None):
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
