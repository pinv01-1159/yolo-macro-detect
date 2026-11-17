"""Paridad entre el modelo local y el desplegado en Roboflow (evidencia de despliegue).

Corre el mismo checkpoint localmente (PyTorch) y contra el endpoint hosteado
(ONNX en Roboflow) sobre una muestra fija del test, y guarda las diferencias.

    uv run --env-file .env python tools/check_roboflow_parity.py
"""

import base64
import glob
import json
import os

import requests
from ultralytics import YOLO

CKPT = "checkpoints/yolo26s_clean_best.pt"
ENDPOINT = "https://serverless.roboflow.com/pinv011159/yolo26s-clean"
CONF = 0.3
OUT = "reports/paper/roboflow_parity.json"


def main() -> None:
    key = os.environ["ROBOFLOW_API_KEY"]
    imgs = sorted(glob.glob("datasets/clean/test/images/*.jpg"))[::12]  # muestra fija, 1 de cada 12
    model = YOLO(CKPT)
    rows = []
    for p in imgs:
        r = model.predict(p, conf=CONF, verbose=False)[0]
        loc = sorted(
            [
                (model.names[int(c)], float(s), b.tolist())
                for c, s, b in zip(r.boxes.cls, r.boxes.conf, r.boxes.xywh, strict=True)
            ],
            key=lambda t: -t[1],
        )
        h = requests.post(
            f"{ENDPOINT}?api_key={key}&confidence={int(CONF * 100)}",
            data=base64.b64encode(open(p, "rb").read()),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=60,
        ).json()
        rem = sorted(
            [
                (d["class"], d["confidence"], [d["x"], d["y"], d["width"], d["height"]])
                for d in h["predictions"]
            ],
            key=lambda t: -t[1],
        )
        row = {"image": os.path.basename(p), "n_local": len(loc), "n_remote": len(rem)}
        if loc and rem:
            row |= {
                "cls_local": loc[0][0],
                "cls_remote": rem[0][0],
                "dconf": abs(loc[0][1] - rem[0][1]),
                "dbox_px": max(abs(a - b) for a, b in zip(loc[0][2], rem[0][2], strict=True)),
            }
        rows.append(row)

    both = [r for r in rows if "dconf" in r]
    summary = {
        "checkpoint": CKPT,
        "endpoint": ENDPOINT,
        "conf": CONF,
        "n_images": len(rows),
        "same_n_detections": sum(r["n_local"] == r["n_remote"] for r in rows),
        "same_top1_class": sum(r["cls_local"] == r["cls_remote"] for r in both),
        "n_with_detections": len(both),
        "max_abs_dconf": max(r["dconf"] for r in both),
        "max_abs_dbox_px": max(r["dbox_px"] for r in both),
    }
    json.dump({"summary": summary, "per_image": rows}, open(OUT, "w"), indent=2)
    print(json.dumps(summary, indent=2))
    assert summary["same_n_detections"] == len(rows) and summary["same_top1_class"] == len(both), (
        "sin paridad"
    )


if __name__ == "__main__":
    main()
