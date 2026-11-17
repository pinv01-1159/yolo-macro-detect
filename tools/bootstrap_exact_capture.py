"""Corre model.val() (protocolo coco: conf=0.001, iou=0.7, split=test) con un
DetectionValidator subclasificado que captura, por imagen, exactamente los
stats que Ultralytics usa para ap_per_class (tp, conf, pred_cls, target_cls),
mas las cajas pred/GT para diagnostico. Una sola inferencia por modelo."""

import pickle
import sys
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/paper/bootstrap_stats"


class CaptureValidator(DetectionValidator):
    captured: list = []

    def update_metrics(self, preds, batch):
        for si, pred in enumerate(preds):
            pb = self._prepare_batch(si, batch)
            CaptureValidator.captured.append(
                {
                    "im_name": Path(pb["im_file"]).name,
                    "pred_boxes": pred["bboxes"].detach().cpu().numpy().copy(),
                    "gt_boxes": pb["bboxes"].detach().cpu().numpy().copy(),
                    "imgsz": tuple(int(x) for x in pb["imgsz"]),
                    "ori_shape": tuple(int(x) for x in pb["ori_shape"]),
                }
            )
        n0 = len(self.metrics.stats["tp"])
        super().update_metrics(preds, batch)
        # stats que acaba de anadir Ultralytics para estas imagenes (sin copia propia del matching)
        for j, k in enumerate(range(n0, len(self.metrics.stats["tp"]))):
            c = CaptureValidator.captured[len(CaptureValidator.captured) - len(preds) + j]
            # orden identico: update_metrics recorre preds en el mismo orden
            for key in ("tp", "conf", "pred_cls", "target_cls"):
                c[key] = np.asarray(self.metrics.stats[key][k]).copy()


DEVICE = "cuda:0" if __import__("torch").cuda.is_available() else "cpu"
for tag in sys.argv[1:] or ["yolo11s", "yolo12s", "yolo26s"]:
    CaptureValidator.captured = []
    model = YOLO(str(ROOT / f"checkpoints/{tag}_clean_best.pt"))
    m = model.val(
        validator=CaptureValidator,
        data=str(ROOT / "datasets/clean/data.yaml"),
        split="test",
        device=DEVICE,
        workers=0,
        conf=0.001,
        iou=0.7,
        plots=False,
        verbose=False,
        project=str(ROOT / "runs/bootstrap_stats"),
        name=tag,
        exist_ok=True,
    )
    ref = {
        "map50": float(m.box.map50),
        "map50_95": float(m.box.map),
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
    }
    print(tag, ref, len(CaptureValidator.captured), flush=True)
    with open(OUT / f"stats_{tag}.pkl", "wb") as f:
        pickle.dump(
            {
                "tag": tag,
                "val_ref": ref,
                "images": CaptureValidator.captured,
                "model_args": {k: model.overrides.get(k) for k in model.overrides},
            },
            f,
        )
