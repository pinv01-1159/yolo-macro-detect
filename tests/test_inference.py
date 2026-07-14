import numpy as np
import torch
from ultralytics.engine.results import Results

import config as config_module
from models.inference import YOLOInference


def _fake_result(names, boxes_xyxy_conf_cls):
    orig_img = np.zeros((100, 100, 3), dtype=np.uint8)
    boxes = torch.tensor(boxes_xyxy_conf_cls, dtype=torch.float32)
    return Results(orig_img=orig_img, path="fake.jpg", names=names, boxes=boxes)


class FakeYoloModel:
    """Sustituye a ultralytics.YOLO: registra los kwargs de cada llamada."""

    def __init__(self):
        self.names = {0: "Physidae"}
        self.calls = []

    def __call__(self, frame, **kwargs):
        self.calls.append(kwargs)
        return [_fake_result(self.names, [[10, 10, 50, 50, 0.9, 0]])]


def _make_inference(tmp_path):
    import cv2

    inference = YOLOInference()
    inference.model = FakeYoloModel()

    image_path = tmp_path / "sample.jpg"
    cv2.imwrite(str(image_path), np.zeros((100, 100, 3), dtype=np.uint8))
    return inference, image_path


def test_predict_image_passes_configured_conf_and_iou_to_model(tmp_path):
    inference, image_path = _make_inference(tmp_path)

    inference.predict_image(
        image_path, conf_threshold=0.42, iou_threshold=0.77, save_annotated=False
    )

    assert inference.model.calls[-1]["conf"] == 0.42
    assert inference.model.calls[-1]["iou"] == 0.77


def test_predict_image_uses_config_defaults_when_not_specified(tmp_path):
    inference, image_path = _make_inference(tmp_path)

    inference.predict_image(image_path, save_annotated=False)

    assert inference.model.calls[-1]["conf"] == config_module.config.confidence_threshold
    assert inference.model.calls[-1]["iou"] == config_module.config.iou_threshold


def test_predict_image_reports_detections_from_fake_model(tmp_path):
    inference, image_path = _make_inference(tmp_path)

    result = inference.predict_image(image_path, save_annotated=False)

    assert result["total_detecciones"] == 1
    assert result["detecciones"][0]["familia"] == "Physidae"
