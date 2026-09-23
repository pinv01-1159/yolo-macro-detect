"""
Re-evaluación independiente de los checkpoints versionados sobre datasets/clean/test.

Corre dos protocolos por modelo:
  - "coco"  : conf=0.001 (estándar COCO/Ultralytics). Es el único válido para
              reportar mAP: truncar la curva PR con un conf alto la sesga.
  - "pipeline": conf=0.3, iou=0.6 — exactamente lo que hace
              models/trainer.py:evaluate() y lo que hay detrás de la tabla
              del README. Se corre para poder contrastar.

Escribe en --out:
    eval_overall.csv / eval_overall.json   métricas globales por modelo/protocolo
    eval_per_class.csv                     una fila por (modelo, protocolo, familia)
    confusion_<modelo>_abs.csv / _norm.csv matriz de confusión (protocolo coco)
Artefactos crudos de Ultralytics en <out>/val_runs/.

Uso:
    python tools/audit_eval_checkpoints.py --data datasets/clean/data.yaml \
        --out reports/paper --device cpu
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO

PROTOCOLS = {
    "coco": {"conf": 0.001, "iou": 0.7},
    "pipeline": {"conf": 0.3, "iou": 0.6},
}


def f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("datasets/clean/data.yaml"))
    ap.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints"))
    ap.add_argument("--out", type=Path, default=Path("reports/paper"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    # RUTA ABSOLUTA a proposito: Ultralytics resuelve un `project` relativo
    # contra su propio `runs_dir` (que utils/runtime.py ancla a la raiz del
    # repo), no contra el cwd. Con una ruta relativa los artefactos terminaban
    # en runs/detect/reports/paper/val_runs/ y las figuras del informe se
    # quedaban con la version anterior sin que nada avisara.
    runs = (args.out / "val_runs").resolve()
    runs.mkdir(parents=True, exist_ok=True)

    overall: list[dict] = []
    per_class: list[dict] = []

    for ckpt in sorted(args.ckpt_dir.glob("*_best.pt")):
        tag = ckpt.stem.replace("_clean_best", "")
        for proto, kw in PROTOCOLS.items():
            print(f"\n=== {tag} / {proto} (conf={kw['conf']}, iou={kw['iou']}) ===", flush=True)
            model = YOLO(str(ckpt))
            m = model.val(
                data=str(args.data), split=args.split, device=args.device,
                plots=(proto == "coco"), verbose=False, save_json=True, save_conf=True,
                project=str(runs), name=f"{tag}_{proto}", exist_ok=True, **kw,
            )
            b = m.box
            row = {
                "model": tag, "protocol": proto, "conf": kw["conf"], "iou": kw["iou"],
                "split": args.split,
                "map50": float(b.map50), "map50_95": float(b.map),
                "precision": float(b.mp), "recall": float(b.mr),
                "f1": f1(float(b.mp), float(b.mr)),
                "save_dir": str(m.save_dir),
            }
            overall.append(row)
            print({k: round(v, 4) for k, v in row.items() if isinstance(v, float)})

            names = model.names
            for i, ci in enumerate(b.ap_class_index):
                per_class.append({
                    "model": tag, "protocol": proto,
                    "family": names[int(ci)],
                    "precision": float(b.p[i]), "recall": float(b.r[i]),
                    "f1": float(b.f1[i]),
                    "map50": float(b.ap50[i]), "map50_95": float(b.ap[i]),
                })

            if proto == "coco":
                cm = getattr(getattr(m, "confusion_matrix", None), "matrix", None)
                if cm is not None:
                    labels = [names[i] for i in range(len(names))] + ["background"]
                    arr = np.asarray(cm, dtype=float)
                    for suffix, mat in (
                        ("abs", arr),
                        ("norm", arr / np.clip(arr.sum(0, keepdims=True), 1e-9, None)),
                    ):
                        p = args.out / f"confusion_{tag}_{suffix}.csv"
                        with open(p, "w", newline="", encoding="utf-8") as f:
                            w = csv.writer(f)
                            w.writerow(["predicha\\real", *labels])
                            for j, r_ in enumerate(mat):
                                w.writerow([labels[j], *[
                                    (int(v) if suffix == "abs" else round(float(v), 5)) for v in r_
                                ]])

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "eval_overall.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(overall[0].keys()))
        w.writeheader()
        w.writerows(overall)
    (args.out / "eval_overall.json").write_text(json.dumps(overall, indent=2, ensure_ascii=False))
    with open(args.out / "eval_per_class.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_class[0].keys()))
        w.writeheader()
        w.writerows(per_class)

    print("\n== resumen ==")
    print(f"{'modelo':<10}{'proto':<10}{'mAP50':>9}{'mAP50-95':>10}{'P':>9}{'R':>9}{'F1':>9}")
    for r in overall:
        print(f"{r['model']:<10}{r['protocol']:<10}{r['map50']:>9.4f}{r['map50_95']:>10.4f}"
              f"{r['precision']:>9.4f}{r['recall']:>9.4f}{r['f1']:>9.4f}")


if __name__ == "__main__":
    main()
