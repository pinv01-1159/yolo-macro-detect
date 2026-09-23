# Checkpoints versionados

Los tres mejores modelos entrenados sobre el split limpio (`datasets/clean/`, ver
[`docs/leakage_analysis.md`](../docs/leakage_analysis.md)), versionados con
[git-lfs](https://git-lfs.com/) para no tener que reentrenar desde cero. Son los mismos
pesos detrás de la tabla de resultados del [README principal](../README.md#-resultados-experimentales).

Reentrenados el 2026-09-23 con el calendario completo (200 épocas, sin parada
temprana). Protocolo estándar de mAP (`conf=0.001`, `iou=0.7`), el mismo que usa
el informe técnico.

> **Reproducibilidad verificada:** YOLO12s y YOLO26s salieron **bit a bit
> idénticos** a los entrenados en agosto en otra máquina —cero tensores
> distintos de 691 y 708 respectivamente—, 45 días después. El procedimiento
> regenera exactamente los modelos que reporta. Fuente: [`reports/paper/eval_overall.csv`](../reports/paper/eval_overall.csv).

| Archivo | Arquitectura | mAP@0.5:0.95 (test) | Época del mejor checkpoint |
|---|---|---|---|
| `yolo11s_clean_best.pt` | YOLO11s | 86.41% | 173/200 |
| `yolo12s_clean_best.pt` | YOLO12s | 87.89% | 158/200 |
| `yolo26s_clean_best.pt` | YOLO26s | **88.07%** | 144/200 |

La diferencia entre las tres cae dentro del ruido muestral (87 especímenes
independientes en test), así que **el proyecto no declara una arquitectura
ganadora**; YOLO26s se adopta como referencia por tener el mejor valor puntual.

> Una versión anterior de esta tabla usaba el protocolo de inferencia
> (`conf=0.3`), que **invertía el ranking** (daba YOLO12s por encima de YOLO26s)
> y no es comparable con la literatura.

> ⚠️ Los `*_environment.json` **quedaron obsoletos**: registran la corrida de
> agosto de 2026 (`patience=30`, parada temprana en 156/188/174 épocas) y ya no
> describen los pesos que tienen al lado, que son de un reentrenamiento del
> 2026-09-23 con el calendario completo. Cada archivo lleva un campo
> `_OBSOLETO` que lo indica. **Usar los `*_train_args.json`**, que se extraen
> del propio checkpoint y registran el commit que Ultralytics escribió durante
> la corrida.

Los `*_environment.json` se generaron el 2026-08-10 con una versión del pipeline
que registraba sólo 12–13 campos. La configuración **completa** de cada
entrenamiento está en los `*_train_args.json` que los acompañan: **111 parámetros
efectivos**, extraídos del diccionario `train_args` que Ultralytics guarda dentro
de cada `.pt` al entrenar. No es una reconstrucción posterior sino el registro que
escribió la propia biblioteca durante la corrida, y se regeneran con:

```bash
uv run python tools/extract_train_args.py
```

Incluyen la aumentación completa, `patience`/`cos_lr`/`amp`, el optimizador y las
tasas de aprendizaje efectivas, las épocas realmente corridas (el campo `epochs`
es el máximo *solicitado*: 200 en los tres) y las métricas de validación finales.

Las huellas SHA-256 de cada artefacto están en
[`reports/paper/provenance.json`](../reports/paper/provenance.json).

## Clonar/traer los pesos

Este repo usa git-lfs. Si `git lfs` no está instalado, los archivos `.pt` de esta carpeta
quedan como *pointers* de texto en vez del binario real:

```bash
git lfs install       # una sola vez por máquina
git lfs pull          # trae el contenido real de los archivos ya trackeados
```

## Uso

```python
from ultralytics import YOLO

model = YOLO("checkpoints/yolo11s_clean_best.pt")
results = model.predict("imagen.jpg")
```

## ⚠️ Antes de usar estos pesos para algo que no sea reproducir la tabla del README

Estos modelos tienen un confusor de sesión sin resolver — ver
[README §Limitaciones y Fuga de Datos](../README.md#-limitaciones-y-fuga-de-datos). En
particular, predicen la familia correcta en ~16% de los casos usando *solo* el fondo, sin
el organismo visible, concentrado en Chironomidae, Hydrophilidae, Planorbidae y
Glossiphoniidae. No están validados para generalizar fuera de las condiciones fotográficas
de este laboratorio.
