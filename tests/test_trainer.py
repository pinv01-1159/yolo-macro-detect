from pathlib import Path

import config as config_module
from models.trainer import YOLOTrainer


class FakeBox:
    def __init__(self, map50=0.9, map_=0.8, mp=0.95, mr=0.9):
        self.map50 = map50
        self.map = map_
        self.mp = mp
        self.mr = mr


class FakeMetrics:
    def __init__(self, val_kwargs=None, **kwargs):
        self.box = FakeBox(**kwargs)
        self.val_kwargs = val_kwargs or {}


class FakeYoloModel:
    """Sustituye a ultralytics.YOLO tanto para entrenar como para evaluar."""

    def __init__(self, *_args, **_kwargs):
        self.train_kwargs = None

    def train(self, **kwargs):
        self.train_kwargs = kwargs
        return FakeMetrics()

    def val(self, **kwargs):
        return FakeMetrics(val_kwargs=kwargs)


def _make_data_yaml(tmp_path: Path) -> Path:
    for folder in ("train", "valid"):
        (tmp_path / folder / "images").mkdir(parents=True)
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        "path: " + str(tmp_path) + "\n"
        "train: train/images\n"
        "val: valid/images\n"
        "nc: 1\n"
        "names: [Physidae]\n"
    )
    return data_yaml


def test_train_passes_configured_seed_to_model_train(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_yaml = _make_data_yaml(tmp_path)
    fake_model = FakeYoloModel()

    trainer = YOLOTrainer("test_experiment")
    trainer.model = fake_model

    weights_dir = tmp_path / "runs" / "detect" / "test_experiment" / "weights"
    weights_dir.mkdir(parents=True)
    (weights_dir / "best.pt").write_bytes(b"fake weights")

    trainer.train(data_yaml_path=str(data_yaml), epochs=1)

    assert fake_model.train_kwargs["seed"] == config_module.config.seed


def test_evaluate_defaults_to_test_split(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("models.trainer.YOLO", FakeYoloModel)

    trainer = YOLOTrainer("test_experiment")
    metrics = trainer.evaluate("fake_model.pt", "fake_data.yaml")

    assert metrics.val_kwargs["split"] == "test"
    assert metrics.val_kwargs["plots"] is True


def test_evaluate_stores_eval_metrics_on_trainer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("models.trainer.YOLO", FakeYoloModel)

    trainer = YOLOTrainer("test_experiment")
    metrics = trainer.evaluate("fake_model.pt", "fake_data.yaml")

    assert trainer.eval_metrics is metrics


def test_get_training_summary_uses_eval_metrics_when_available(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    trainer = YOLOTrainer("test_experiment")
    trainer.eval_metrics = FakeMetrics(map50=0.77, map_=0.66, mp=0.88, mr=0.55)

    summary = trainer.get_training_summary()

    assert summary["evaluated_split"] == "test"
    assert summary["metrics"]["map50"] == 0.77
    assert summary["metrics"]["recall"] == 0.55
