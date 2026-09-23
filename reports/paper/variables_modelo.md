# Registro de variables empleadas para la detección automática

Proyecto CONACYT-Paraguay **PINV01-1159** · ArtICS Lab
Documento generado por la auditoría técnica (`reports/paper/auditoria.md`) para cubrir el
**ítem 4 del Anexo I, Informe técnico N.º 1**: *"registro de variables empleadas para la
detección automática"*.

Checkpoints: `890f8e2` (según `checkpoints/*_environment.json`). El commit del código y la
huella SHA-256 de cada artefacto están en
[`provenance.json`](provenance.json), que se regenera con
`uv run python tools/make_provenance.py` y avisa si el árbol tenía cambios sin commitear.

> **Nota de procedencia.** Cada variable indica de dónde sale su valor efectivo. Donde dice
> *"default de Ultralytics"* significa que el proyecto **no la fija explícitamente** y hereda el
> valor de la librería: esos valores son reproducibles solo si se fija la versión
> (`ultralytics==8.4.95`, ya pineada en `uv.lock`).
>
> **Hallazgo S2 — cerrado.** El pipeline ya vuelca `train_kwargs` y las métricas en
> `environment.json` para entrenamientos nuevos. Y para los tres checkpoints ya entrenados no
> hizo falta reentrenar ni reconstruir nada: **Ultralytics guarda la configuración efectiva
> dentro del propio `.pt`**. Extraída a
> `checkpoints/*_train_args.json` (111 parámetros por modelo) con
> `tools/extract_train_args.py`. Es el registro que escribió la biblioteca durante el
> entrenamiento, no una reconstrucción a posteriori, así que los bloques §3 y §4 de este
> documento son ahora verificables contra una salida de máquina.
>
> Un dato que surgió de esa extracción: el campo `epochs` de `environment.json` es el máximo
> **solicitado** (200 en los tres), no el corrido. Las épocas reales —156, 188 y 174— están en
> el historial `train_results` del checkpoint.

---

## 1 · Dos rutas de entrenamiento — no confundirlas

El proyecto tiene **dos conjuntos de variables distintos** según dónde se entrenó. Mezclarlos en
el informe sería un error, porque los datos de entrada no son los mismos.

| | **Ruta A — local (la que reporta el informe)** | **Ruta B — nube (Roboflow)** |
|---|---|---|
| Pesos | `checkpoints/yolo{11s,12s,26s}_clean_best.pt` | 6 entrenamientos, solo `training_id` |
| Dataset | `datasets/clean/` — **1 660** imágenes de train | proyecto `macroinvertebrados-split-limpio-kuhq3` v1 — **4 980** imágenes de train |
| Augmentación | **dinámica**, en el dataloader de Ultralytics (§4) | **estática**, materializada en el dataset (§5) |
| Control de hiperparámetros | total (`models/trainer.py`) | ninguno (caja negra) |
| Registro | `checkpoints/*_environment.json` (parcial) | `reports/paper/trainings.json` (solo IDs) |

**Las métricas del informe provienen de la Ruta A.** La Ruta B queda como comparación de
arquitecturas; sus variables internas no son auditables y así debe declararse.

---

## 2 · Datos de entrada

| Variable | Valor | Procedencia |
|---|---|---|
| Dataset fuente | `roboflow://pinv011159/macroinvertebrados-acuaticos` v9 | `datasets/v9/data.yaml:20-26` |
| Imágenes fuente únicas | 2 403 | `datasets/clean/split_report.json` → `images` |
| Clases (familias) | 19 | `datasets/clean/data.yaml:5` |
| Instancias anotadas | 3 032 | `reports/paper/bbox_summary.json` → `n_instances` |
| Formato de anotación | YOLO (`class cx cy w h`, normalizado) | `datasets/clean/*/labels/*.txt` |
| Resolución del canvas | 640 × 640 px (uniforme, verificado en las 2 403) | `reports/paper/bbox_summary.json` → `canvas_sizes` |
| Preprocesamiento | `auto-orient` + `resize 640×640 stretch` | `reports/paper/trainings.json:14` (aplicado en Roboflow) |
| Imágenes negativas (sin objeto) | **0** | verificado: Ultralytics reporta `0 backgrounds` |
| Instancias por imagen | media 1.26 · máx 4 · un solo taxón por imagen | `reports/paper/audit_splits.json` |

### Partición

| Variable | Valor | Procedencia |
|---|---|---|
| Método | agrupado por espécimen (union-find), estratificado por clase | `tools/build_clean_split.py:82-143` |
| Criterio de agrupamiento | gap temporal ≤ **60 s** **O** coseno de miniatura > **0.95** | `tools/build_clean_split.py:150-151` (`--gap`, `--vis`) |
| Descriptor de similitud | miniatura 32×32 gris, z-score por imagen, L2 | `tools/build_clean_split.py:71-79` |
| Proporciones | **0.70 / 0.15 / 0.15** | `tools/build_clean_split.py:152` (`--ratios`) |
| Semilla | **42** | `tools/build_clean_split.py:153` (`--seed`) |
| Imágenes por split | 1 660 / 373 / 370 | `split_report.json` → `images_per_split` |
| Instancias por split | 2 109 / 465 / 458 | `reports/paper/audit_splits.json` |
| **Grupos independientes por split** | **473 / 81 / 87** | `split_report.json` → `groups_per_split` |
| MD5 de `split_report.json` | `6a7e79a8aa7a3eecb6a43aa9d2ae1b46` | verificado reproducible bit-a-bit (auditoría §8.1) |

> El **n efectivo** para cualquier intervalo de confianza es **87 grupos** en test (2–7 por familia),
> no 370 imágenes. Ver `reports/paper/groups_per_class.csv` y `bootstrap_grouped.json`.

---

## 3 · Arquitectura y optimización (Ruta A)

| Variable | YOLO11s | YOLO12s | YOLO26s | Procedencia |
|---|---|---|---|---|
| Pesos iniciales | `yolo11s.pt` | `yolo12s.pt` | `yolo26s.pt` | `checkpoints/*_environment.json` → `config.model_name` |
| Épocas corridas | **200** | **200** | **200** | `train_args`/`train_results` del checkpoint |
| Mejor época | 173 | 158 | 144 | ídem |
| Mejor época | 126 | 158 | 144 | `checkpoints/README.md` |
| `batch` | 16 | **8** | 16 | ídem → `config.batch_size` |
| `imgsz` | 640 | 640 | 640 | ídem → `config.img_size` |
| `workers` | 8 | 8 | 8 | ídem → `config.workers` |
| `seed` | 42 | 42 | 42 | ídem → `config.seed` |

> YOLO12s usó `batch=8` porque el primer intento con 16 murió por OOM en la RTX 4050 de 6 GB
> (documentado en el campo `note` de `yolo12s_clean_environment.json`). **No es una variable
> controlada**: es una restricción de hardware que rompe la comparabilidad estricta entre las tres
> arquitecturas. Debe declararse en el informe.

### Hiperparámetros de entrenamiento (idénticos en las tres)

| Variable | Valor | Procedencia | Nota |
|---|---|---|---|
| `deterministic` | `True` | `models/trainer.py:216` | |
| `patience` | **= `epochs`** (desactivada) | `models/trainer.py` | Antes 30; con la parada temprana activa el calendario de `cos_lr` y `close_mosaic` quedaba trunco |
| `cos_lr` | `True` | `models/trainer.py:220` | annealing coseno |
| `amp` | `True` | `models/trainer.py:221` | mixed precision, por los 6 GB de VRAM |
| `save_period` | 10 | `models/trainer.py:218` | |
| `optimizer` | **AdamW** (resuelto por `optimizer='auto'`) | `train_args` + regla de Ultralytics | ver aviso abajo |
| `lr0` **efectivo** | **0.000435** | derivado + curvas `lr/pg*` del checkpoint | **no** 0.01 |
| `momentum` **efectivo** | **0.9** | ídem | **no** 0.937 |
| `warmup_bias_lr` **efectivo** | **0.0** | forzado por la rama `auto` | **no** 0.1 |
| `lrf` | 0.01 | default de Ultralytics | sí se usa (factor final del coseno) |
| `accumulate` | 4 (yolo11s/26s) · **8** (yolo12s) | `nbs/batch` | consecuencia del lote reducido |
| `weight_decay` | 0.0005 | default de Ultralytics | **no fijado** |
| `warmup_epochs` | 3.0 | default de Ultralytics | **no fijado** |
| `warmup_momentum` | 0.8 | default de Ultralytics | **no fijado** |
| `warmup_bias_lr` | 0.1 | default de Ultralytics | **no fijado** |
| `box` / `cls` / `dfl` (pesos de pérdida) | 7.5 / 0.5 / 1.5 | default de Ultralytics | **no fijado** |
| `nbs` (nominal batch size) | 64 | default de Ultralytics | **no fijado** |

> **El learning rate publicado antes estaba 23× por encima del real.** Con
> `optimizer='auto'`, Ultralytics **ignora** los `lr0` y `momentum` que figuran en
> `train_args` y los recalcula. El propio código lo anuncia al arrancar:
> *"'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937'"*
> (`engine/trainer.py:1064-1073`).
>
> Para este conjunto (`nc=19`, 1 660 imágenes de entrenamiento, `nbs=64`,
> `iterations = 5 200 ≤ 10 000`) la regla resuelve a **AdamW** con
> `lr0 = round(0.002·5/(4+19), 6) = 0.000435` y `momentum = 0.9`. Corroborado
> contra las curvas registradas: `lr/pg0-2` sube en warmup
> (0.000144 → 0.000289) y se estabiliza en **0.000434761** en la época 4.
>
> Fuente por modelo: `checkpoints/*_train_args.json` → `derivado.optimizador`.
> Curvas completas: `checkpoints/*_training_curves.csv`.
| `attention_reg_lambda` | **no aplica** | — | `models/attention_regularization.py` **no existe** en `890f8e2`: la regularización por atención es posterior a estos pesos y no estuvo activa |

---

## 4 · Augmentación dinámica — Ruta A (dataloader de Ultralytics)

**Estos 14 parámetros son los que no quedan registrados en ninguna salida de máquina** (hallazgo
S2). Están hardcodeados en `models/trainer.py:222-229` y solo se recuperan leyendo el código en el
commit correspondiente.

Se apartan deliberadamente de los valores de fábrica: el dataset es chico y cada familia se
fotografió en pocas sesiones, así que fondo e iluminación están correlacionados con la clase. La
augmentación fuerte busca romper ese atajo (ver `docs/leakage_analysis.md` §3).

| # | Variable | Valor | Default Ultralytics | Procedencia | Propósito declarado |
|---|---|---:|---:|---|---|
| 1 | `hsv_h` | **0.02** | 0.015 | `trainer.py:223` | tono |
| 2 | `hsv_s` | **0.8** | 0.7 | `trainer.py:223` | saturación — color de sesión |
| 3 | `hsv_v` | **0.5** | 0.4 | `trainer.py:223` | brillo — iluminación de sesión |
| 4 | `degrees` | **20.0** | 0.0 | `trainer.py:224` | el espécimen en bandeja no tiene "arriba" |
| 5 | `translate` | **0.15** | 0.1 | `trainer.py:225` | traslación |
| 6 | `scale` | **0.6** | 0.5 | `trainer.py:225` | escala |
| 7 | `shear` | **5.0** | 0.0 | `trainer.py:225` | cizalladura |
| 8 | `fliplr` | **0.5** | 0.5 | `trainer.py:226` | espejado horizontal |
| 9 | `flipud` | **0.5** | 0.0 | `trainer.py:226` | espejado vertical |
| 10 | `mosaic` | **1.0** | 1.0 | `trainer.py:227` | mezcla fondos entre imágenes |
| 11 | `close_mosaic` | **15** | 10 | `trainer.py:227` | últimas épocas sin mosaico |
| 12 | `mixup` | **0.1** | 0.0 | `trainer.py:228` | mezcla de pares |
| 13 | `erasing` | **0.4** | 0.4 | `trainer.py:229` | ocluye partes: penaliza memorizar el fondo |
| 14 | `perspective` | 0.0 | 0.0 | default | no usado |

Valores en **negrita** = distintos del default de Ultralytics.

> **Garantía metodológica**: la augmentación se aplica **solo al dataloader de entrenamiento**.
> Ultralytics nunca augmenta en `val()`/`predict()`. Verificado además a nivel de datos:
> `datasets/clean/` tiene **0 archivos con sufijo `.rf.<hash>`** en los tres splits
> (`reports/paper/audit_splits.json` → `roboflow_aug_suffix_count`).

---

## 5 · Augmentación estática — Ruta B (Roboflow, solo comparación)

Aplicada **dentro del dataset**, no en el dataloader. Multiplica el train ×3 (1 660 → 4 980) y
**no afecta a valid/test**, verificado: en v9, valid y test tienen 1 imagen por fuente y el solape
de nombres entre splits es 0.

| Variable | Valor | Procedencia |
|---|---|---|
| Versiones por imagen | 3 | `reports/paper/trainings.json:18` |
| Flip | horizontal + vertical | ídem `:19` |
| Rotación | ±20° | ídem `:20` |
| Brillo | ±25 % | ídem `:21` |
| Exposición | ±15 % | ídem `:22` |
| Saturación | ±25 % | ídem `:23` |
| Ámbito | **solo split de train** | ídem `:24` |

Los hiperparámetros internos de los 6 entrenamientos cloud (optimizador, LR, épocas, augmentación
adicional) **no son accesibles**. Solo se dispone de `model_type` y `training_id`
(`reports/paper/trainings.json:31-38`).

---

## 6 · Umbrales de inferencia y evaluación

| Variable | Valor | Ámbito | Procedencia |
|---|---:|---|---|
| `CONFIDENCE_THRESHOLD` (τ) | **0.30** | inferencia y eval del pipeline | `config.py:53-55`, `env.example:19` |
| `IOU_THRESHOLD` (NMS) | **0.60** | inferencia y eval del pipeline | `config.py:56`, `env.example:20` |
| `IMG_SIZE` | 640 | inferencia | `config.py:44` |
| `max_det` | 300 | inferencia | default de Ultralytics |
| IoU de emparejamiento (métricas) | 0.50 : 0.05 : 0.95 | cálculo de mAP | estándar COCO |

### Protocolos de evaluación — tres, y hay que distinguirlos

| Protocolo | `conf` | `iou` (NMS) | Split | Para qué sirve | Procedencia |
|---|---:|---:|---|---|---|
| **A · mAP estándar** | **0.001** | 0.7 | `test` | **la métrica citable**; es la que usa la literatura | auditoría §4, `tools/audit_eval_checkpoints.py` |
| **B · pipeline (histórico)** | 0.30 | 0.6 | `test` | el que produjo la tabla del README | `models/trainer.py:285-310` |
| **C · punto de operación** | τ elegido en `valid` | 0.6 | `valid` → `test` | umbral de producción | `tools/audit_sweep_and_bootstrap.py` |

> **Hallazgo B5 — corregido.** Fijar `conf=0.30` trunca la cola de la curva precisión-recall y
> sesga el mAP. El código separa ahora ambos protocolos (`MAP_EVAL_CONF = 0.001` y
> `MAP_EVAL_IOU = 0.7` en `models/trainer.py`), y el informe, el README y
> `checkpoints/README.md` publican el protocolo estándar. El punto de operación (τ=0.30) se
> reporta como tal y por separado.
>
> **Hallazgo B6 — corregido.** El barrido corre ahora sobre validación
> (`reports/model_report.py` → `SWEEP_SPLIT = "val"`), con un test que lo verifica. Tras el
> reentrenamiento del 2026-09-23 el óptimo cae en **τ=0.25** (F1 0.9849) frente a τ=0.30
> (F1 0.9838): una milésima de diferencia, y 0.30 comete menos falsos positivos. Se conserva
> τ=0.30. Ver `reports/paper/threshold_sweep_valid.csv`.

---

## 7 · Entorno de cómputo

| Variable | Valor | Procedencia |
|---|---|---|
| GPU de entrenamiento | NVIDIA GeForce RTX 4050 Laptop (6 GB) | `checkpoints/*_environment.json` → `gpu` |
| Python | 3.13.14 | ídem → `python_version` |
| PyTorch | 2.13.0+cu130 | ídem → `torch_version` |
| CUDA | 13.0 | ídem → `cuda_version` |
| Ultralytics | 8.4.95 | ídem → `ultralytics_version` |
| Gestor de dependencias | `uv` con `uv.lock` pineado | `pyproject.toml` |
| Proveedor de `cv2` | `opencv-python-headless` (forzado) | `pyproject.toml:53-58` |

---

## 8 · Variables NO controladas — declarar como limitación

Un registro de variables honesto incluye lo que no se controló:

| Variable | Estado | Impacto |
|---|---|---|
| **Sesión de captura** | 12 de 19 familias fotografiadas en **un solo día** | fondo e iluminación predicen la clase; un clasificador trivial sobre miniaturas 32×32 llega a 39.5 % de accuracy (azar 5.3 %) |
| **Fondo / bandeja / iluminación** | sin aleatorizar entre familias | atajo por fondo ≈ 16 % de las instancias |
| **Equipo fotográfico y montaje** | no documentado en el repo | irreproducible |
| **Distancia y encuadre** | no estandarizado explícitamente | la caja ocupa entre 6.9 % y 100 % del cuadro (mediana 52.6 %) |
| **Anotador y criterio de encuadre** | no documentado | sin medición de acuerdo inter-anotador |
| **Validación taxonómica** | no documentada | no hay registro de verificación por un biólogo |
| `batch` de YOLO12s | 8 en vez de 16, por OOM | rompe la comparabilidad estricta entre arquitecturas |
| Hiperparámetros de la Ruta B | inaccesibles (Roboflow) | los 6 modelos cloud no son auditables |

---

## 9 · Cómo reproducir

```bash
git clone https://github.com/pinv01-1159/yolo-macro-detect && cd yolo-macro-detect
uv sync
cp env.example .env   # y completar ROBOFLOW_API_KEY

# 1. dataset fuente (v9) desde Roboflow
uv run main.py --setup-dataset --dataset-version 9

# 2. partición agrupada por espécimen (determinista, seed 42)
uv run tools/build_clean_split.py datasets/v9 datasets/clean
md5sum datasets/clean/split_report.json   # debe dar 6a7e79a8aa7a3eecb6a43aa9d2ae1b46

# 3. verificación de integridad de la partición
uv run tools/audit_splits.py datasets/clean --out reports/paper

# 4. evaluación con el protocolo correcto sobre los pesos versionados
git lfs pull
uv run tools/audit_eval_checkpoints.py --data datasets/clean/data.yaml --out reports/paper

# 5. punto de operación (valid) + IC agrupados (test)
uv run tools/audit_sweep_and_bootstrap.py checkpoints/yolo26s_clean_best.pt \
    --data datasets/clean --out reports/paper
```

Para reentrenar desde cero en GPU gratuita (2× T4): `notebooks/kaggle.ipynb`.

---

## 10 · Archivos de respaldo

| Archivo | Contenido |
|---|---|
| `datasets/clean/split_report.json` | grupos, conteos por clase, residual de similitud, `group_of_image` |
| `checkpoints/*_environment.json` | versiones, GPU, commit, config parcial del run |
| `reports/paper/trainings.json` | IDs de los 6 entrenamientos cloud |
| `reports/paper/audit_splits.json` | integridad de la partición y de las etiquetas |
| `reports/paper/bbox_summary.json` · `bbox_per_family.csv` | geometría de las cajas |
| `reports/paper/eval_overall.csv` · `eval_per_class.csv` | métricas, 3 modelos × 2 protocolos |
| `reports/paper/threshold_sweep_valid.csv` | barrido de τ sobre validación |
| `reports/paper/bootstrap_grouped.json` | IC95 agrupados por espécimen |
| `reports/paper/family_size_vs_performance.csv` | geometría ↔ desempeño por familia |
| `reports/paper/auditoria.md` | auditoría técnica completa |
