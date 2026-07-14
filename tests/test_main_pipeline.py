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

    pipeline = MacroinvertebratePipeline()
    pipeline.train_model(data_yaml_path="fake_data.yaml", experiment_name="exp1")

    metrics_file = tmp_path / "results" / "exp1" / "eval_metrics.json"
    assert metrics_file.exists()
    saved = json.loads(metrics_file.read_text())
    assert saved["metrics"]["map50"] == 0.9
    assert saved["evaluated_split"] == "test"
