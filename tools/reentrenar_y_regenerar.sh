#!/usr/bin/env bash
# Reentrena las tres arquitecturas con el calendario de entrenamiento completo y
# regenera todos los artefactos del informe.
#
# POR QUE: cos_lr y close_mosaic se calendarizan contra `epochs` (200), pero
# patience=30 cortaba antes (156/188/174). La tasa de aprendizaje terminaba entre
# el 2 % y el 13 % de su pico en vez del 1 %, y la fase final sin mosaico --la que
# afina la localizacion-- no se ejecutaba en dos de los tres modelos.
# El arreglo esta en models/trainer.py: patience = epochs.
#
# COSTO: ~6-7 h en una RTX 4050 de 6 GB. Los tres modelos corren las 200 epocas
# completas. best.pt sigue siendo el mejor checkpoint por validacion, asi que
# desactivar la parada temprana no implica entregar un modelo sobreajustado.
#
# TRAS CORRERLO: todos los numeros del informe cambian y hay que actualizarlos.
# El paso 3 imprime los nuevos valores.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA=datasets/clean/data.yaml
[ -f "$DATA" ] || { echo "falta $DATA"; exit 1; }
uv run python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "ERROR: no hay GPU disponible. Este guion necesita CUDA."; exit 1; }

echo "=============================================================="
echo " PASO 0/4  Descargar los pesos base ANTES de entrenar"
echo "=============================================================="
# Ultralytics descarga los pesos preentrenados al instanciar el modelo. Si la
# red se cae entre un modelo y el siguiente, el entrenamiento muere ahi
# ("Download failure ... Environment may be offline"). Paso exactamente eso el
# 2026-09-23: yolo11s y yolo12s terminaron, y yolo26s no llego a arrancar.
# Bajarlos todos primero convierte 6 h de trabajo en algo que no depende de que
# la red aguante.
for m in yolo11s yolo12s yolo26s; do
  if [ -f "${m}.pt" ]; then
    echo "  ${m}.pt ya esta"
  else
    echo "  bajando ${m}.pt ..."
    curl -fL --retry 3 -o "${m}.pt" \
      "https://github.com/ultralytics/assets/releases/download/v8.4.0/${m}.pt" \
      || { echo "ERROR: no se pudo bajar ${m}.pt"; exit 1; }
  fi
done

echo "=============================================================="
echo " PASO 1/4  Entrenamiento (~6-7 h)"
echo "=============================================================="
# yolo12s va con batch 8: con 16 agota los 6 GB de VRAM (OOM registrado en agosto).
entrenar() {
  local modelo=$1 batch=$2
  echo; echo "--- ${modelo} (batch ${batch}) ---"
  MODEL_NAME="${modelo}.pt" EXPERIMENT_NAME="${modelo}_clean" BATCH_SIZE="${batch}" \
    uv run python main.py --train --data-yaml "$DATA"
}
entrenar yolo11s 16
entrenar yolo12s 8
entrenar yolo26s 16

echo "=============================================================="
echo " PASO 2/4  Publicar los pesos nuevos"
echo "=============================================================="
for m in yolo11s yolo12s yolo26s; do
  src="runs/detect/${m}_clean/weights/best.pt"
  [ -f "$src" ] || src=$(ls -t runs/detect/${m}_clean*/weights/best.pt 2>/dev/null | head -1)
  [ -f "$src" ] || { echo "ERROR: no se encontro best.pt de ${m}"; exit 1; }
  cp -v "$src" "checkpoints/${m}_clean_best.pt"
done
uv run python tools/extract_train_args.py

echo "=============================================================="
echo " PASO 3/4  Regenerar metricas y figuras"
echo "=============================================================="
uv run python tools/audit_eval_checkpoints.py
uv run python tools/audit_size_vs_performance.py
uv run python tools/audit_sweep_and_bootstrap.py checkpoints/yolo26s_clean_best.pt --device "${SWEEP_DEVICE:-0}"
uv run python tools/make_report_figures.py
uv run python tools/make_inference_examples.py
cp -v reports/paper/val_runs/yolo26s_coco/confusion_matrix_normalized.png informe/imagenes/
cp -v reports/paper/val_runs/yolo26s_coco/val_batch0_pred.jpg informe/imagenes/pipeline_validacion.jpg
uv run python tools/make_provenance.py

echo "=============================================================="
echo " PASO 4/4  Valores nuevos para actualizar los documentos"
echo "=============================================================="
uv run python - <<'PY'
import csv, json, pathlib
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
    cerro = "SI" if len(filas) >= 185 else "NO"
    print(f"  {m:9} corrio={len(filas):>4} mejor={mejor:>4} "
          f"cierra_mosaico={cerro} LR_final/pico={lr[-1]/max(lr):.1%} (objetivo 1.0%)")
PY
echo
echo "Listo. Pasale estos numeros a Claude para que actualice el .tex, el README,"
echo "checkpoints/README.md, docs/bitacora.md y reports/paper/variables_modelo.md."
