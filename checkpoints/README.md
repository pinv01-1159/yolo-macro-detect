# Checkpoints versionados

Los tres mejores modelos entrenados sobre el split limpio (`datasets/clean/`, ver
[`docs/leakage_analysis.md`](../docs/leakage_analysis.md)), versionados con
[git-lfs](https://git-lfs.com/) para no tener que reentrenar desde cero. Son los mismos
pesos detrás de la tabla de resultados del [README principal](../README.md#-resultados-experimentales).

| Archivo | Arquitectura | mAP@0.5:0.95 (test) | Época del mejor checkpoint |
|---|---|---|---|
| `yolo11s_clean_best.pt` | YOLO11s | 86.15% | 126/156 |
| `yolo12s_clean_best.pt` | YOLO12s | 87.75% | 158/188 |
| `yolo26s_clean_best.pt` | YOLO26s | 87.51% | 144/174 |

Cada `*_environment.json` acompañante tiene la metadata completa de reproducibilidad de
ese entrenamiento: versiones (torch/CUDA/ultralytics), GPU, commit de git, config efectiva
(épocas, batch, seed) y una copia del reporte del split usado.

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
