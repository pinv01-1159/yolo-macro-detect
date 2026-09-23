#!/usr/bin/env bash
# Completa la regeneracion cuando el entrenamiento ya termino.
#
# Existe porque la corrida del 2026-09-23 murio en audit_sweep_and_bootstrap.py:
# ese script pide el checkpoint como argumento posicional y el guion lo invocaba
# sin argumentos. Con `set -e` eso corto todo lo que venia despues.
set -euo pipefail
cd "$(dirname "$0")/.."

# `uv` puede no estar en el PATH de una sesion no interactiva; el venv si.
PY=".venv/bin/python"
[ -x "$PY" ] || { echo "ERROR: falta $PY (corre: uv sync)"; exit 1; }

for m in yolo11s yolo12s yolo26s; do
  [ -f "checkpoints/${m}_clean_best.pt" ] || { echo "ERROR: falta checkpoints/${m}_clean_best.pt"; exit 1; }
done
echo "=== los tres checkpoints estan publicados ==="

"$PY" tools/audit_sweep_and_bootstrap.py checkpoints/yolo26s_clean_best.pt --device "${SWEEP_DEVICE:-0}"
"$PY" tools/make_report_figures.py
"$PY" tools/make_inference_examples.py
cp -v reports/paper/val_runs/yolo26s_coco/confusion_matrix_normalized.png informe/imagenes/
cp -v reports/paper/val_runs/yolo26s_coco/val_batch0_pred.jpg informe/imagenes/pipeline_validacion.jpg
"$PY" tools/make_provenance.py

echo
echo "=== PASO 4/4  Valores nuevos ==="
"$PY" - <<'PY'
import csv, pathlib
print("\n--- metricas (protocolo coco) ---")
for r in csv.DictReader(open("reports/paper/eval_overall.csv")):
    if r["protocol"] == "coco":
        print(f"  {r['model']:9} mAP50={float(r['map50']):.4f} "
              f"mAP50-95={float(r['map50_95']):.4f} P={float(r['precision']):.4f} "
              f"R={float(r['recall']):.4f} F1={float(r['f1']):.4f}")
print("\n--- calendario ---")
for m in ("yolo11s", "yolo12s", "yolo26s"):
    p = pathlib.Path(f"checkpoints/{m}_clean_training_curves.csv")
    if not p.exists():
        continue
    filas = list(csv.DictReader(open(p)))
    lr = [float(x["lr/pg0"]) for x in filas]
    col = next(c for c in filas[0] if "mAP50-95" in c)
    mp = [float(x[col]) for x in filas]
    mejor = max(range(len(mp)), key=lambda i: mp[i]) + 1
    print(f"  {m:9} corrio={len(filas):>4} mejor={mejor:>4} "
          f"cierra_mosaico={'SI' if len(filas) >= 185 else 'NO'} "
          f"LR_final/pico={lr[-1]/max(lr):.2%} mAP_val_best={max(mp):.4f}")
PY
