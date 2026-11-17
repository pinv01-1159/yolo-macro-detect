"""Barrido del umbral de operación tau sobre VALIDACIÓN con la inferencia del sistema.

Usa el mismo acondicionamiento que models/inference.py (margen de BORDER_PX) y el
emparejamiento de COCO: predicciones en orden de confianza descendente, cada una se
asigna al GT libre de su clase con mayor IoU (>= 0.5). La incertidumbre de
F1(tau*) - F1(0.30) se estima con bootstrap pareado agrupado por espécimen.

    uv run python tools/sweep_operating_point.py checkpoints/yolo26s_clean_best.pt --device cuda:0
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.inference import BORDER_PX, pad_border, unpad_detections  # noqa: E402

TAUS = np.round(np.arange(0.05, 0.9501, 0.05), 2)
BASELINE = 0.30


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def counts(dets, gt):
    """tp y n_pred por cada tau para una imagen."""
    tp, npred = np.zeros(len(TAUS), int), np.zeros(len(TAUS), int)
    for k, t in enumerate(TAUS):
        keep = [d for d in dets if d[1] >= t]
        used = set()
        for c, _, box in keep:
            cand = [(iou(box, g), j) for j, (gc, g) in enumerate(gt) if gc == c and j not in used]
            best = max(cand, default=(0.0, -1))
            if best[0] >= 0.5:
                used.add(best[1])
                tp[k] += 1
        npred[k] = len(keep)
    return tp, npred


def f1(tp, npred, ngt):
    p = np.divide(tp, npred, out=np.zeros(len(TAUS)), where=npred > 0)
    r = tp / ngt
    return p, r, np.divide(2 * p * r, p + r, out=np.zeros(len(TAUS)), where=(p + r) > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="reports/paper/operating_point_sweep.json")
    a = ap.parse_args()
    groups = json.load(open(ROOT / "datasets/clean/split_report.json"))["group_of_image"]
    model = YOLO(a.model)
    items = []
    for p in sorted((ROOT / f"datasets/clean/{a.split}/images").glob("*.jpg")):
        f = cv2.imread(str(p))
        h, w = f.shape[:2]
        lab = p.parent.parent / "labels" / (p.stem + ".txt")
        gt = []
        for line in open(lab):
            c, x, y, bw, bh = line.split()
            x, y, bw, bh = (float(v) for v in (x, y, bw, bh))
            gt.append(
                (int(c), [(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h])
            )
        r = model(
            pad_border(f),
            imgsz=640 + 2 * BORDER_PX,
            conf=float(TAUS[0]),
            iou=0.6,
            device=a.device,
            verbose=False,
        )[0]
        d = unpad_detections(sv.Detections.from_ultralytics(r), f.shape)
        dets = sorted(
            zip(d.class_id.tolist(), d.confidence.tolist(), d.xyxy.tolist(), strict=True),
            key=lambda t: -t[1],
        )
        tp, npred = counts(dets, gt)
        g = groups.get(p.name) or groups.get(p.stem)
        items.append({"group": g, "tp": tp, "npred": npred, "ngt": len(gt)})
    assert all(it["group"] is not None for it in items), "imagen sin grupo en split_report.json"

    def agg(sel):
        tp = sum(items[i]["tp"] for i in sel)
        npred = sum(items[i]["npred"] for i in sel)
        return f1(tp, npred, sum(items[i]["ngt"] for i in sel))

    P, R, F = agg(range(len(items)))
    k_star, k_base = int(np.argmax(F)), int(np.argmin(np.abs(TAUS - BASELINE)))
    by_group = {}
    for i, it in enumerate(items):
        by_group.setdefault(it["group"], []).append(i)
    keys = list(by_group)
    rng = np.random.default_rng(42)
    diffs, argmaxes = [], []
    for _ in range(1000):
        sel = [i for g in rng.choice(keys, len(keys)) for i in by_group[g]]
        Fb = agg(sel)[2]
        diffs.append(Fb[k_star] - Fb[k_base])
        argmaxes.append(float(TAUS[int(np.argmax(Fb))]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    out = {
        "model": a.model,
        "split": a.split,
        "n_images": len(items),
        "n_groups": len(keys),
        "inference": {
            "border_px": BORDER_PX,
            "nms_iou": 0.6,
            "matching": "COCO, confianza descendente, IoU>=0.5",
        },
        "sweep": [
            {
                "tau": float(t),
                "precision": round(float(p), 4),
                "recall": round(float(r), 4),
                "f1": round(float(f), 4),
            }
            for t, p, r, f in zip(TAUS, P, R, F, strict=True)
        ],
        "tau_star": float(TAUS[k_star]),
        "f1_tau_star": round(float(F[k_star]), 4),
        "f1_baseline": round(float(F[k_base]), 4),
        "paired_diff_ci95": [round(float(lo), 4), round(float(hi), 4)],
        "argmax_frequency": {
            str(t): argmaxes.count(float(t)) / 1000 for t in sorted(set(argmaxes))
        },
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "sweep"}, indent=1))
    for row in out["sweep"][:10]:
        print(row)


if __name__ == "__main__":
    main()
