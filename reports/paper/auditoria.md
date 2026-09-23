# Auditoría técnica — `yolo-macro-detect` (PINV01-1159)

Fecha: 2026-09-20 (correcciones verificadas hasta 2026-09-23) · Alcance: código, datasets en disco, checkpoints, reproducibilidad.

Metodología: todo lo que se afirma acá sale de correr código sobre los archivos reales, no de leer la documentación. Los scripts nuevos usados están en `tools/audit_*.py`; sus salidas, versionadas en este directorio.

## Estado de los hallazgos

| Hallazgo | Estado |
|---|---|
| **BMWP** multiplicaba por abundancia, cubría 9/19 familias, dos puntajes invertidos, se calculaba por foto | Corregido — reescrito a presencia/ausencia por sitio, 19 familias BMWP/Col, ASPT. 12 tests |
| **mAP publicado con `conf=0.3`** en vez del protocolo estándar (`conf=0.001`) | Corregido — `MAP_EVAL_CONF=0.001` / `MAP_EVAL_IOU=0.7` |
| **Barrido de umbral de confianza sobre test** (selección de hiperparámetro en el conjunto de reporte) | Corregido — se corre sobre `valid` |
| `.gitignore` excluía las figuras y artefactos del entregable | Corregido |
| CI rojo, un test pegaba contra la API real de Roboflow | Corregido — CI verde, red bloqueada en tests |
| Bootstrap por imagen (pseudorreplicación: la unidad independiente es el espécimen, no la foto) | Corregido — remuestreo por grupo/espécimen |
| `environment.json` no registraba métricas ni augmentación; split embebido por copia | Corregido en el código; para los checkpoints ya entrenados, la configuración efectiva se recuperó de `train_args` (`checkpoints/*_train_args.json`) |
| Ablación de fondo sesgada por cajas grandes | Parcial — el script ahora reporta el subconjunto `area_frac < 0.40`; cifras previas del README deben citarse solo tras re-ejecutar |
| `predict_batch()` ignoraba el umbral de IoU | Corregido |
| `x or default` trataba `0.0` como "no especificado" | Corregido |
| **Cero imágenes negativas** en el dataset | Abierto — requiere campaña de captura, no código. Declarado como limitación en el informe |
| Rutas globales de Ultralytics resueltas fuera del repo | Corregido — `utils/runtime.py` |
| Protocolo de evaluación (`conf`/`iou`/`split`) sin registrar | Corregido — `_protocol()` |
| Latencia medida sin *warm-up* ni sincronización CUDA | Corregido; las latencias publicadas se retiraron |
| Empaquetado (`wheel` sin `main.py`/`config.py`), metadatos de `pyproject.toml`, `load_dotenv(Path.cwd())`, `validate()` sin tolerar `IMG_SIZE` no múltiplo de 32 | Corregidos y verificados (build de wheel, tests) |

`ruff` y `mypy` limpios, 65 tests. Huella de cada artefacto (SHA-256, commit, comando que lo regenera) en [`provenance.json`](provenance.json).

## Resumen ejecutivo

La corrección del split (`tools/build_clean_split.py`) se verificó empíricamente: 0 grupos compartidos entre splits, 0 imágenes sin cobertura, reproducción bit-a-bit determinista. El residual de similitud coseno >0.90 entre splits **no es fuga**: es indistinguible del piso intra-clase dentro de train (§2).

El hallazgo más relevante para el informe **contradice la hipótesis de partida**: no hay ningún objeto pequeño en este dataset (100 % de las cajas son *large* en escala COCO; mediana = 52.6 % del cuadro). La dificultad de las dos familias más débiles (Chironomidae, Ceratopogonidae) se explica por **elongación**, no por tamaño (§1).

Re-evaluando los tres checkpoints de forma independiente, los números publicados **reproducen exacto en 11 de 12 cifras hasta la cuarta decimal** (§3) — confirmando que se generaron con `conf=0.3`, no con el protocolo estándar de mAP, y que con el protocolo correcto el mejor modelo es YOLO26s (mAP@0.5:0.95 = 0.8807).

## 1. Dimensiones de los macroinvertebrados y elongación

Fuente: `tools/audit_bbox_dimensions.py` sobre `datasets/clean/` (3 032 instancias, 2 403 imágenes, canvas 640×640).

**No hay objetos pequeños.** El 100 % de las cajas es *large* en escala COCO (>96²px); la más chica mide 28 092 px², 27× el umbral de *small*. La mediana ocupa el 52.6 % del cuadro (p95 = 97.7 %). Es consecuencia del protocolo de captura (macro-toma de un espécimen centrado), no un defecto de anotación.

**La variable que predice el desempeño es la elongación, no el tamaño.** Sobre las 19 familias, correlación con mAP@0.5:0.95: elongación r = −0.838 (R² = 0.702, t(17) = −6.33, p < .001); área r = +0.419 (n.s., y colapsa a r = +0.177 al controlar por elongación). Chironomidae (elongación mediana 2.82) y Ceratopogonidae (2.05) son las dos familias más alargadas y las dos más débiles en las tres arquitecturas — una caja alineada a ejes que encierra un gusano curvo contiene mayoritariamente fondo, lo que penaliza el IoU en umbrales altos.

**Caveat de robustez**: excluyendo esas dos familias, la correlación con elongación baja a r = −0.430. El efecto está dominado por los dos taxones filiformes, no es una ley lineal sobre las 19 familias — Hirudinidae (2.ª más alargada) rinde bien (mAP = 0.887) porque es compacta y de alto contraste, no filiforme.

**Dato adicional**: 82.5 % de las imágenes tiene una sola instancia y 0 % son multi-clase — la tarea está más cerca de clasificación de imagen que de detección de objetos, lo que explica por qué mAP@0.5 satura en las tres arquitecturas.

Datos completos: `bbox_per_family.csv`, `bbox_instances.csv`, `family_size_vs_performance.csv`.

## 2. Splits y fuga de datos — verificación empírica

`tools/audit_splits.py` contra el disco: 0 grupos en más de un split, 641/641 grupos coinciden entre JSON y disco, 0 imágenes sin cobertura, 0 diferencias de conteo por clase. `tools/build_clean_split.py` es determinista bit-a-bit (dos corridas → mismo manifiesto).

**El residual coseno >0.90 entre splits no es fuga** (`tools/audit_similarity.py`): comparando el vecino más cercano en train bajo cuatro condiciones —

| Condición | mediana | >0.90 |
|---|---:|---:|
| test → train (cross-split) | 0.813 | 16.2 % |
| train → train, distinto grupo, misma clase (piso natural) | 0.792 | 15.7 % |
| train → train, **mismo grupo** (fuga real) | **0.998** | 59.2 % |
| train → train, otra clase (control) | 0.727 | 0.8 % |

el cross-split (test→train) es estadísticamente indistinguible del piso intra-train (15.7–16.2 % vs 15.7 %). La fuga real tiene una firma separada y sin solape: mediana 0.998. El umbral `--vis 0.95` usado por el script cae en una región vacía entre ambas distribuciones.

Integridad de etiquetas: 0 archivos faltantes, 0 líneas malformadas, 0 `class_id` fuera de rango, 0 cajas degeneradas o fuera de canvas (`tools/audit_splits.py`).

## 3. Re-evaluación independiente de los checkpoints

Los tres `checkpoints/*_best.pt` (vía git-lfs) se re-evaluaron sobre `datasets/clean/test` (370 imágenes, 458 instancias, 87 especímenes independientes) en CPU, protocolo `coco` (`conf=0.001, iou=0.7`, el único válido para mAP).

| Modelo | mAP@0.5 | mAP@0.5:0.95 | Precisión | Recall |
|---|---:|---:|---:|---:|
| YOLO11s | 0.9914 | 0.8655 | 0.9769 | 0.9938 |
| YOLO12s | 0.9950 | 0.8789 | 0.9905 | 0.9976 |
| YOLO26s | 0.9947 | **0.8807** | 0.9896 | 0.9742 |

Contrastando el protocolo `pipeline` (`conf=0.3`, el que usaba el pipeline original) contra la tabla publicada en el README: **reproducción exacta en 11 de 12 cifras hasta la cuarta decimal** — confirma que los números publicados son honestos y que se generaron con `conf=0.3`, no el protocolo estándar. Con el protocolo correcto el ranking cambia: el mejor modelo pasa a ser YOLO26s, no YOLO12s (diferencia de 0.18 pp, no distinguible del ruido dado el n efectivo de 87 grupos).

Por familia (protocolo `coco`, media de las tres arquitecturas): Chironomidae (0.599) y Ceratopogonidae (0.796) son las dos peores en los tres modelos; el orden es estable entre arquitecturas, lo que confirma que la dificultad es del taxón, no de la corrida. Datos completos: `eval_overall.csv`, `eval_per_class.csv`, `confusion_<modelo>_{abs,norm}.csv`.

Bootstrap agrupado por espécimen: `bootstrap_grouped.json`. El IC del mAP vía el estimador manual no coincide con el de Ultralytics (`reproduces: false`) — no hay hoy un IC formal utilizable para el mAP; el bootstrap agrupado sí es válido para P/R.

## 4. Reproducibilidad — qué falta

| Paso | Estado |
|---|---|
| Reconstruir `datasets/clean` desde `datasets/v9` | Reproducible y determinista (§2) |
| Reproducir el split sin acceso a Roboflow | Sí — `split_report.json` versionado contiene el mapeo completo |
| Reproducir las métricas desde los pesos | Sí — checkpoints en git-lfs, protocolo registrado en `_protocol()` |
| Reentrenar los 3 modelos exactamente | No — los hiperparámetros de augmentación viven en el código (`trainer.py`), no en `environment.json`; falta el `results.csv` por época de cada corrida |
| Reproducir las latencias publicadas | No — requiere la GPU original; el método no estaba documentado (ya corregido para mediciones futuras) |

## Hallazgos menores abiertos

- **`tools/copy_paste_augment.py:160`**: variable asignada y nunca usada.
- **`reports/paper/trainings.json`**: declara los tres entrenamientos como `pendiente` cuando ya están hechos y versionados en `checkpoints/`.
- **`checkpoints/*_environment.json`**: el `git_commit` registrado es el del backfill post-hoc, no el del código con que se entrenó — trazabilidad aparente, no real.
- **Protocolo de anotación y validación taxonómica por un especialista**: no documentado en ningún archivo del repo. Pendiente para la campaña de captura complementaria (ver informe, Trabajo futuro).
