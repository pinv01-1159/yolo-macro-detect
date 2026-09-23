#!/usr/bin/env bash
# Reanuda el reentrenamiento: entrena SOLO yolo26s y regenera los artefactos.
#
# POR QUE EXISTE: la corrida del 2026-09-23 completo yolo11s y yolo12s, pero
# yolo26s murio antes de empezar porque `yolo26s.pt` (pesos preentrenados) no
# estaba en disco y el WiFi se habia caido -- "Download failure ... Environment
# may be offline". Los otros dos ya tenian sus pesos base descargados, por eso
# no les afecto. Volver a entrenar los tres seria tirar ~4 h de GPU ya pagadas.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA=datasets/clean/data.yaml

echo "=== Comprobaciones previas ==="
uv run python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "ERROR: no hay GPU"; exit 1; }
# La causa exacta de la caida anterior: sin esto el entrenamiento intenta
# descargar los pesos base al arrancar y muere si la red no esta.
[ -f yolo26s.pt ] || { echo "ERROR: falta yolo26s.pt. Descargalo antes:"; \
  echo "  curl -fL -o yolo26s.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt"; exit 1; }
for m in yolo11s yolo12s; do
  ls runs/detect/${m}_clean*/weights/best.pt >/dev/null 2>&1 \
    || { echo "ERROR: falta el best.pt de ${m}, que deberia estar de la corrida anterior"; exit 1; }
done
echo "  GPU OK · yolo26s.pt presente · yolo11s y yolo12s ya entrenados"

echo
echo "=== Entrenando yolo26s (~1 h 50) ==="
MODEL_NAME=yolo26s.pt EXPERIMENT_NAME=yolo26s_clean BATCH_SIZE=16 \
  uv run python main.py --train --data-yaml "$DATA"

echo
echo "=== Publicando los tres checkpoints ==="
for m in yolo11s yolo12s yolo26s; do
  src=$(ls -t runs/detect/${m}_clean*/weights/best.pt 2>/dev/null | head -1)
  [ -f "$src" ] || { echo "ERROR: no se encontro best.pt de ${m}"; exit 1; }
  cp -v "$src" "checkpoints/${m}_clean_best.pt"
done
uv run python tools/extract_train_args.py

echo
echo "=== Regenerando metricas y figuras ==="
uv run python tools/audit_eval_checkpoints.py
uv run python tools/audit_size_vs_performance.py
uv run python tools/audit_sweep_and_bootstrap.py checkpoints/yolo26s_clean_best.pt --device "${SWEEP_DEVICE:-0}"
uv run python tools/make_report_figures.py
uv run python tools/make_inference_examples.py
cp -v reports/paper/val_runs/yolo26s_coco/confusion_matrix_normalized.png informe/imagenes/
cp -v reports/paper/val_runs/yolo26s_coco/val_batch0_pred.jpg informe/imagenes/pipeline_validacion.jpg
uv run python tools/make_provenance.py

echo
echo "=== PASO 4/4  Valores nuevos ==="
uv run python - <<'PY'
import csv, pathlib
print("\n--- metricas (protocolo coco) ---")
for r in csv.DictReader(open("reports/paper/eval_overall.csv")):
    if r["protocol"] == "coco":
        print(f"  {r['model']:9} mAP50={float(r['map50']):.4f} "
              f"mAP50-95={float(r['map50_95']):.4f} P={float(r['precision']):.4f} "
              f"R={float(r['recall']):.4f} F1={float(r['f1']):.4f}")
print("\n--- calendario: se completo? ---")
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
          f"LR_final/pico={lr[-1]/max(lr):.2%} (objetivo 1.00%)")
PY
echo
echo "Listo. Pasale estos numeros a Claude."
