# 🦐 YOLO Macroinvertebrados - Detección Automática de Macroinvertebrados Acuáticos

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![YOLO](https://img.shields.io/badge/YOLO-11%2C12%2C26-green.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Project](https://img.shields.io/badge/Project-PINV01--1159-red.svg)]()

Sistema de visión por computadora para la detección automática de macroinvertebrados acuáticos y evaluación de calidad del agua mediante inteligencia artificial. Este proyecto entrena y compara tres arquitecturas de detección de objetos (YOLO11s, YOLO12s y YOLO26s) para identificar 19 familias de macroinvertebrados y calcular índices bióticos BMWP para inferir la calidad ecológica del agua.

> ⚠️ **Antes de leer las métricas: este dataset tiene una fuga de datos conocida y un confusor de fondo/sesión sin resolver.**
> Los números de este README son del split reconstruido (ver [Limitaciones y fuga de datos](#-limitaciones-y-fuga-de-datos)) — léela antes de citar cualquier cifra de este proyecto en un paper.

## 📋 Tabla de Contenidos

- [Características](#-características)
- [Resultados Destacados](#-resultados-destacados)
- [Instalación](#-instalación)
- [Configuración](#-configuración)
- [Uso](#-uso)
- [Pipeline](#-pipeline)
- [Modelos Implementados](#-modelos-implementados)
- [Evaluación de Calidad del Agua](#-evaluación-de-calidad-del-agua)
- [Resultados Experimentales](#-resultados-experimentales)
- [Limitaciones y Fuga de Datos](#-limitaciones-y-fuga-de-datos)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [API Reference](#-api-reference)
- [Contribución](#-contribución)
- [Licencia](#-licencia)

## ✨ Características

- 🔍 **Detección Automática**: Identificación de macroinvertebrados comparando tres arquitecturas YOLO (11s, 12s, 26s)
- 📊 **19 Familias Detectadas**: Ampullariidae, Ancylidae, Belostomatidae, Ceratopogonidae, Chironomidae, Coenagrionidae, Dytiscidae, Gerridae, Glossiphoniidae, Hirudinidae, Hydrophilidae, Hyriidae, Libellulidae, Miridae, Noteridae, Notonectidae, Physidae, Planorbidae, Psychodidae
- 🌊 **Evaluación de Calidad del Agua**: Cálculo automático del índice BMWP
- 📈 **Métricas Detalladas**: mAP@0.5, mAP@0.5:0.95, precisión, recall, intervalos de confianza por bootstrap, barrido de confianza y latencia, por clase y por modelo
- 🕵️ **Auditoría de dataset integrada**: detección de fuga por ráfaga/duplicados (`tools/build_clean_split.py`), medición del confusor de fondo (`tools/confound_check.py`, `tools/background_ablation.py`)
- 🖼️ **Anotación Visual**: Generación automática de imágenes anotadas
- 📝 **Logging Completo**: Sistema de logs para seguimiento de entrenamiento e inferencia
- ⚙️ **Configuración Flexible**: Variables de entorno para personalización
- 🚀 **Pipeline Automatizado**: Proceso completo desde dataset hasta inferencia

## 🏆 Resultados Destacados

### Métricas de Rendimiento (split limpio, sin fuga — test set, n=370 imágenes / 465 instancias)

| Modelo | Precisión | Recall | mAP@0.5 | mAP@0.5:0.95 | Latencia media (GPU) |
|--------|-----------|--------|---------|--------------|----------------------|
| **YOLO11s** | 99.6% | 99.3% | 98.7% | 86.1% | 9.5 ms |
| **YOLO12s** | 100% | 99.8% | 99.3% | 87.8% | 12.2 ms |
| **YOLO26s** | 98.4% | 99.4% | 98.9% | 87.5% | 9.8 ms |

mAP@0.5 está prácticamente saturado en los tres modelos (satura por ser un umbral de IoU muy laxo dado el tamaño de los especímenes en estas fotos) — **mAP@0.5:0.95 es la métrica que separa modelos y la que hay que citar como principal.**

Estos números son casi idénticos a los que reportaba el dataset con la fuga de datos original (99.4% mAP@0.5). Eso **no** significa que la fuga no importara — significa que el split limpio no alcanza para exponer del todo un segundo problema estructural del dataset (ver [Limitaciones](#-limitaciones-y-fuga-de-datos)): el fondo/sesión de laboratorio sigue siendo parcialmente predictivo de la clase. Ningún número de este README debe leerse como evidencia de generalización a un arroyo real sin leer esa sección primero.

### Dataset
- **2,403 imágenes fuente únicas** (tras descartar duplicados por aumentación de Roboflow), reagrupadas en **641 grupos por espécimen/ráfaga**
- **19 familias** de macroinvertebrados acuáticos
- **Split reconstruido por grupo** (no por imagen): 1,660 train / 373 valid / 370 test — ver [`tools/build_clean_split.py`](tools/build_clean_split.py)
- **Aumentación en entrenamiento** con parámetros no estándar, elegidos específicamente para pelear contra el confusor de fondo (color/geometría/mosaico fuertes) — ver [`models/trainer.py`](models/trainer.py)

## 🛠️ Instalación

### Requisitos Previos

- Python 3.10 o superior
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- CUDA compatible (recomendado para GPU)
- Git

### Instalación del Proyecto

1. **Clonar el repositorio**
   ```bash
   git clone https://github.com/gsmkev/yolo-macro-detect.git
   cd yolo-macro-detect
   ```

2. **Instalar dependencias**
   ```bash
   uv sync
   ```
   Esto crea automáticamente un entorno virtual en `.venv` con las dependencias fijadas en `uv.lock`. Antepón `uv run` a cualquier comando (p. ej. `uv run main.py ...`) para ejecutarlo dentro de ese entorno.

3. **Configurar variables de entorno**
   ```bash
   cp env.example .env
   # Editar .env con tus credenciales
   ```

## ⚙️ Configuración

### Variables de Entorno

Crea un archivo `.env` basado en `env.example`:

```env
# Configuración de Roboflow
ROBOFLOW_API_KEY=tu_api_key_aqui
ROBOFLOW_WORKSPACE=pinv011159
ROBOFLOW_PROJECT=macroinvertebrados-acuaticos

# Configuración del modelo (yolo11s.pt, yolo12s.pt o yolo26s.pt)
MODEL_NAME=yolo11s.pt
EXPERIMENT_NAME=macros
TRAINING_EPOCHS=200
IMG_SIZE=640
BATCH_SIZE=16
WORKERS=8

# Regularización por atención (opcional, default 0 = desactivada).
# lambda>0 penaliza en la función de pérdida la activación de features
# fuera de las cajas GT -- ver models/attention_regularization.py.
# Probado (λ=1.0): no redujo el atajo por fondo, ver README §Limitaciones.
ATTENTION_REG_LAMBDA=0.0

# Configuración de inferencia
CONFIDENCE_THRESHOLD=0.3
IOU_THRESHOLD=0.6

# Configuración de logging
LOG_LEVEL=INFO
SAVE_RESULTS=True

# Configuración BMWP (Evaluación de Calidad del Agua)
ENABLE_BMWP=True
BMWP_CONFIDENCE_WEIGHT=True
```

### Obtener API Key de Roboflow

1. Registrarse en [Roboflow](https://roboflow.com)
2. Ir a Account Settings > API Key
3. Copiar la API key y agregarla al archivo `.env`

## 🚀 Uso

### Pipeline Completo

Ejecutar todo el proceso desde la descarga del dataset hasta el entrenamiento:

```bash
uv run main.py --pipeline-complete
```

### Solo Configurar Dataset

Descargar y configurar el dataset sin entrenar:

```bash
uv run main.py --setup-dataset
```

### Solo Entrenamiento

Entrenar modelo con dataset existente:

```bash
uv run main.py --train --data-yaml datasets/data.yaml
```

### Solo Predicción

Realizar predicción en una imagen:

```bash
uv run main.py --predict --image test.jpg --model runs/detect/macros/weights/best.pt
```

### Predicción con Evaluación de Calidad del Agua

Calcular índice BMWP basado en las detecciones:

```bash
uv run main.py --predict --image sample.jpg --model best_model.pt --calculate-bmwp
```

### Opciones Avanzadas

```bash
# Pipeline completo con parámetros personalizados
uv run main.py --pipeline-complete \
    --dataset-version 5 \
    --epochs 100 \
    --experiment-name "macros_v2"

# Predicción con umbral personalizado y BMWP
uv run main.py --predict \
    --image sample.jpg \
    --model best_model.pt \
    --confidence 0.5 \
    --calculate-bmwp
```

## 🔄 Pipeline

### 1. Descarga de Dataset
- Conexión automática con Roboflow
- Descarga de la versión del dataset elegida (`--dataset-version`) — el export crudo trae duplicados por aumentación de Roboflow y el split original **tiene fuga por ráfaga**, así que hay que reconstruirlo con `tools/build_clean_split.py` antes de entrenar (ver [Limitaciones y Fuga de Datos](#-limitaciones-y-fuga-de-datos))
- Validación de estructura y contenido
- Generación de archivo `data.yaml`

### 2. Entrenamiento del Modelo
- Carga del modelo base YOLO (yolo11s, yolo12s o yolo26s, vía `MODEL_NAME`)
- Configuración de hiperparámetros (augmentación anti-atajo fuerte, opcionalmente regularización por atención)
- Entrenamiento con early stopping
- Guardado de checkpoints

### 3. Evaluación
- Validación en conjunto de test
- Cálculo de métricas (mAP, precisión, recall)
- Generación de gráficos de rendimiento
- Análisis de matriz de confusión

### 4. Inferencia
- Carga del modelo entrenado
- Predicción en imágenes nuevas
- Anotación automática
- Cálculo de índice BMWP (opcional)
- Exportación de resultados

## 🤖 Modelos Implementados

Las tres arquitecturas se entrenan localmente con el mismo pipeline (`main.py --train`), el mismo split, la misma configuración y la misma semilla (42) — solo cambia `MODEL_NAME`. Eso hace la comparación entre ellas válida.

| Modelo | Parámetros | Hardware usado | Notas |
|--------|-----------|-----------------|-------|
| **YOLO11s** | 9.4M | RTX 4050 Laptop (6 GB), AMP | mejor checkpoint en época 126/156 |
| **YOLO12s** | ~9.3M | RTX 4050 Laptop (6 GB), AMP, batch=8 | requirió bajar batch por OOM; mejor en época 158/188 |
| **YOLO26s** | 10.0M | RTX 4050 Laptop (6 GB), AMP | arquitectura end-to-end (NMS-free); mejor en época 144/174 |

Config de entrenamiento: `imgsz=640`, `epochs=200` con early stopping (`patience=30`), `cos_lr=True`, augmentación anti-atajo fuerte (ver [`models/trainer.py`](models/trainer.py)). El `EXPERIMENT_NAME=<modelo>_clean` en `runs/detect/` y `results/` identifica cada corrida.

## 🌊 Evaluación de Calidad del Agua

### Índice BMWP (Biological Monitoring Working Party)

El sistema calcula automáticamente la calidad del agua basándose en las familias detectadas:

| Clase | BMWP | Calidad del Agua |
|-------|------|------------------|
| I | >101 | Muy limpia |
| II | 61–100 | Aceptable |
| III | 36–60 | Dudosa |
| IV | 16–35 | Crítica |
| V | <15 | Muy crítica |

### Puntajes por Familia

| Familia | Puntaje BMWP |
|---------|--------------|
| Belostomatidae | 5 |
| Coenagrionidae | 7 |
| Dytiscidae | 3 |
| Physidae | 3 |
| Planorbidae | 5 |
| Chironomidae | 8 |
| Noteridae | 4 |
| Libellulidae | 8 |
| Hirudinidae | 9 |

> ⚠️ `utils/bmwp_calculator.py` solo tiene puntaje BMWP cargado para 9 de las 19 familias que detectan los modelos actuales (falta Ampullariidae, Ancylidae, Ceratopogonidae, Gerridae, Glossiphoniidae, Hydrophilidae, Hyriidae, Miridae, Notonectidae y Psychodidae). Si el detector encuentra una de esas 10 familias, `get_family_score()` devuelve `None` y esa detección queda fuera del cálculo del índice — no fallar en silencio, pero sí subestimar el BMWP total si aparecen. Hay que completar la tabla con la fuente bibliográfica correspondiente antes de usar el índice en un reporte real.

### Ejemplo de Cálculo

```json
{
  "detecciones": [
    {"familia": "Physidae", "cantidad": 6, "bmwp": 3},
    {"familia": "Planorbidae", "cantidad": 4, "bmwp": 5},
    {"familia": "Chironomidae", "cantidad": 3, "bmwp": 8},
    {"familia": "Hirudinidae", "cantidad": 2, "bmwp": 9}
  ],
  "bmwp_total": 55,
  "calidad_agua": "Dudosa (Clase III)",
  "confianza": 0.94
}
```

### Uso de la Calculadora BMWP

```python
from utils.bmwp_calculator import bmwp_calculator

# Detecciones de ejemplo
detections = [
    {"familia": "Physidae", "cantidad": 6, "confidence_promedio": 0.95},
    {"familia": "Chironomidae", "cantidad": 3, "confidence_promedio": 0.88}
]

# Calcular BMWP
result = bmwp_calculator.calculate_bmwp(detections)
print(f"BMWP: {result.total_score} - {result.water_quality_description}")
```

## 📊 Resultados Experimentales

Todos los números de esta sección son del **split reconstruido sin fuga** (`datasets/clean/`, ver siguiente sección), evaluados una sola vez sobre `test/` (nunca usado para elegir pesos ni para tunear nada). Los artefactos crudos detrás de cada número —predicciones por imagen, matriz de confusión completa, metadata de entorno/reproducibilidad— están en `results/<experimento>/` para cada modelo.

### Comparación entre arquitecturas (test set)

| Modelo | mAP@0.5 | mAP@0.5:0.95 | Precisión | Recall | Latencia (GPU) | Mejor época |
|--------|---------|--------------|-----------|--------|-----------------|-------------|
| YOLO11s | 98.71% | **86.15%** | 99.63% | 99.31% | 9.5 ms | 126/156 |
| YOLO12s | 99.34% | **87.75%** | 100.00% | 99.81% | 12.2 ms | 158/188 |
| YOLO26s | 98.90% | **87.51%** | 98.40% | 99.37% | 9.8 ms | 144/174 |

Las tres arquitecturas convergen al mismo techo de rendimiento (mAP@0.5:0.95 entre 86-88%), entrenadas por separado con el mismo split. Que arquitecturas independientes lleguen al mismo lugar es un dato en sí mismo — ver [Limitaciones](#-limitaciones-y-fuga-de-datos).

### Clases más débiles (consistentes en las tres arquitecturas)

| Clase | AP@0.5:0.95 (YOLO11s / YOLO12s / YOLO26s) | Nota |
|-------|---------------------------------------------|------|
| Chironomidae | 0.531 / 0.633 / 0.575 | la peor en los tres modelos — larva de díptero chica y alargada |
| Ceratopogonidae | 0.774 / 0.798 / 0.790 | segunda peor en los tres — también díptero chico |

Que el mismo par de clases sea el más débil en tres arquitecturas independientes apunta a dificultad genuina de detección de objeto chico, no a ruido de una corrida particular.

## 🕵️ Limitaciones y Fuga de Datos

### 1. Fuga por ráfaga (encontrada y corregida)

El dataset original de Roboflow son fotos en ráfaga del mismo espécimen (`AAAA_MMDD_HHMMSS_seq`), y Roboflow asignaba el split **por imagen**, al azar — así que cuadros casi idénticos del mismo individuo terminaban repartidos entre train y valid/test. Evidencia medida (ver [`docs/leakage_analysis.md`](docs/leakage_analysis.md)):

- 87.3% de las imágenes fuente caían en ráfagas repartidas entre splits
- 29.2% de `valid` tenía un gemelo visual en `train` (coseno > 0.99 sobre miniaturas 32×32)
- Las métricas viejas (99.4% mAP@0.5, recall 100%) medían memorización de cuadro, no reconocimiento de familia

**Corrección**: [`tools/build_clean_split.py`](tools/build_clean_split.py) reagrupa las imágenes por espécimen (union-find sobre cercanía temporal ≤60s o similitud visual >0.95) y asigna **grupos enteros** a un único split, estratificado por clase. Resultado: 641 grupos, 1660/373/370 imágenes, **0.0% de residuo con coseno >0.95** entre train y valid/test.

```bash
python tools/build_clean_split.py datasets/v9 datasets/clean
```

### 2. Confusor de sesión (no corregido — estructural del dataset)

12 de las 19 familias se fotografiaron en una sola sesión de laboratorio. Arreglar la fuga por ráfaga no arregla esto: el fondo/iluminación de esa sesión sigue correlacionado con la clase, y el split por grupos no puede desenredarlo porque no hay una segunda sesión de la que tomar imágenes independientes para esas 12 familias.

**Medición del confusor puro** ([`tools/confound_check.py`](tools/confound_check.py)): un clasificador de centroide más cercano sobre miniaturas 32×32 en gris —sin ver al bicho, del tamaño de un emoji— alcanza **39.5% de accuracy en test** (azar: 5.3%; clase mayoritaria: 7.8%) usando *solo* fondo/iluminación.

**Medición sobre los modelos reales** ([`tools/background_ablation.py`](tools/background_ablation.py)), dos experimentos por instancia de test:

- **Solo el bicho** (recorte ajustado a la caja GT, sin nada de fondo): 97.6-99.8% de accuracy en los tres modelos — el modelo generaliza sobre morfología real en la gran mayoría de los casos.
- **Solo el fondo** (bicho tapado con relleno gris, resto de la imagen intacto): el modelo predice la clase correcta *donde debería estar el bicho* en **~16% de las instancias**, sin que haya ningún animal visible. No está distribuido parejo — se concentra en clases puntuales:

| Clase | Tasa de atajo por fondo (bicho tapado) |
|-------|------------------------------------------|
| Chironomidae | 76-84% |
| Hydrophilidae | 58% |
| Planorbidae | 47% |
| Glossiphoniidae | 36% |
| Dytiscidae | 21-25% |
| resto (14 clases) | ≤8% |

### 3. Tres intentos de mitigación, mismo resultado nulo

Se probaron tres técnicas de la literatura para reducir el atajo por fondo, sin tocar el split ni bajar la augmentación existente:

| Técnica | Mecanismo | Resultado (tasa de atajo agregada) |
|---------|-----------|--------------------------------------|
| Baseline (3 arquitecturas) | — | 0.157 – 0.162 |
| Copy-paste augmentation 1x ([`tools/copy_paste_augment.py`](tools/copy_paste_augment.py)) | pega el recorte del bicho sobre fondo de otra sesión, mismo label | 0.166 |
| Copy-paste augmentation 3x (79% del train sintético) | igual, con más volumen | 0.162 |
| Regularización por atención λ=1.0 (3 arquitecturas, [`models/attention_regularization.py`](models/attention_regularization.py)) | penaliza en la función de pérdida la activación fuera de la caja GT (estilo "Right for the Right Reasons", Ross et al. 2017) | 0.157 – 0.159 |

**Ocho corridas independientes, tres arquitecturas, tres mecanismos de intervención — rango de 0.157 a 0.166.** Esa estabilidad tan extrema sugiere que la correlación fondo↔clase en los datos de entrenamiento es lo bastante limpia como para que cualquier modelo razonablemente entrenado la encuentre y la use, sin importar cómo se penalice internamente: mientras la señal exista en los píxeles de entrada, el descenso de gradiente tiene incentivo genuino para explotarla. Ninguna técnica que dejamos de probar tocó la distribución de entrada en sí misma de forma suficientemente agresiva.

### Qué hace falta para arreglarlo de verdad

Ninguna intervención de entrenamiento va a resolver esto — es un problema de qué fotos existen, no de cómo se entrena con ellas. La mitigación real es de captura: fotografiar cada familia en **≥3 sesiones distintas** (fondo/bandeja/iluminación diferentes) y reservar una sesión completa como test *out-of-session*. Ese es el número que hay que reportar para afirmar generalización real, y hoy no existe en este dataset.

**Consecuencia práctica:** un mAP alto acá no demuestra que el sistema vaya a funcionar identificando macroinvertebrados en un arroyo nuevo, fuera de las condiciones fotográficas de este laboratorio. Cualquier paper basado en este proyecto tiene que reportar el número agregado, la tabla de atajo por clase, y esta limitación explícitamente — no como nota al pie.

## 📁 Estructura del Proyecto

```
yolo-macro-detect/
├── main.py                        # Script principal
├── config.py                      # Configuración centralizada
├── pyproject.toml                 # Dependencias y configuración (uv)
├── env.example                    # Ejemplo de variables de entorno
├── README.md                      # Documentación
├── LICENSE                        # Licencia
│
├── docs/
│   └── leakage_analysis.md        # Análisis medido de la fuga por ráfaga y el confusor de sesión
│
├── data/                          # Manejo de datasets
│   ├── __init__.py
│   └── dataset_manager.py
│
├── models/                        # Modelos, entrenamiento y regularización
│   ├── __init__.py
│   ├── trainer.py
│   ├── inference.py
│   └── attention_regularization.py  # Penalización de activación fuera de la caja GT
│
├── reports/                       # Reportes estadísticos de dataset y modelo
│   ├── dataset_report.py
│   └── model_report.py
│
├── tools/                         # Auditoría de dataset y experimentos de mitigación
│   ├── build_clean_split.py       # Reconstruye el split agrupando por espécimen
│   ├── confound_check.py          # Mide el confusor de sesión puro (sin ver al bicho)
│   ├── background_ablation.py     # Mide dependencia del fondo en un modelo ya entrenado
│   └── copy_paste_augment.py      # Augmentación copy-paste "Same Y" (intento de mitigación, no funcionó)
│
├── utils/                         # Utilidades
│   ├── __init__.py
│   ├── logger.py
│   ├── validators.py
│   └── bmwp_calculator.py
│
├── examples/                      # Ejemplos de uso
│   └── example_usage.py
│
├── logs/                          # Logs del sistema
├── datasets/                      # Datasets descargados (gitignored)
├── results/                       # Resultados de inferencia y reportes (gitignored)
└── runs/                          # Checkpoints y resultados de entrenamiento (gitignored)
```

## 📚 API Reference

### MacroinvertebratePipeline

Clase principal para manejo del pipeline completo.

```python
from main import MacroinvertebratePipeline

pipeline = MacroinvertebratePipeline()

# Configurar dataset
data_yaml = pipeline.setup_dataset(version=5)

# Entrenar modelo
model_path = pipeline.train_model(data_yaml, epochs=50)

# Realizar predicción con BMWP
results = pipeline.predict_image("test.jpg", model_path, calculate_bmwp=True)
```

### YOLOTrainer

Clase para entrenamiento de modelos YOLO.

```python
from models import YOLOTrainer

trainer = YOLOTrainer("experimento_1")
trainer.load_model("yolo11s.pt")
model_path = trainer.train("data.yaml", epochs=50)
metrics = trainer.evaluate(model_path, "data.yaml")
```

### YOLOInference

Clase para inferencia con modelos entrenados.

```python
from models import YOLOInference

inference = YOLOInference("best_model.pt")
results = inference.predict_image("image.jpg", conf_threshold=0.3, calculate_bmwp=True)
bmwp_score = inference.calculate_bmwp(results['detecciones'])
inference.export_results(results, "output.json")
```

### BMWPCalculator

Clase para cálculo del índice BMWP.

```python
from utils.bmwp_calculator import bmwp_calculator

# Calcular BMWP
result = bmwp_calculator.calculate_bmwp(detections)

# Obtener información
families = bmwp_calculator.get_available_families()
water_quality_info = bmwp_calculator.get_water_quality_info()

# Formatear para JSON
json_result = bmwp_calculator.format_result_for_json(result)
```

### DatasetManager

Clase para manejo de datasets.

```python
from data import DatasetManager

manager = DatasetManager()
manager.setup_roboflow_connection()
dataset_info = manager.download_dataset(version=5)
manager.validate_dataset_structure(dataset_info["location"])
```


### Guías de Contribución

- Seguir las convenciones de código PEP 8
- Agregar docstrings a todas las funciones
- Incluir tests para nuevas funcionalidades
- Actualizar documentación según sea necesario

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

## 👨‍💻 Autor

**Kevin M. Galeano**
- **Proyecto**: PINV01-1159
- **Institución**: Universidad Nacional de Asunción
- **Email**: [gsmkev@gmail.com](mailto:gsmkev@gmail.com)
- **GitHub**: [@gsmkev](https://github.com/gsmkev)


## 🙏 Agradecimientos

- **CONACYT Paraguay** por el financiamiento del proyecto PROCIENCIA
- [Ultralytics](https://github.com/ultralytics/ultralytics) por los modelos YOLO
- [Roboflow](https://roboflow.com) por la plataforma de datasets y entrenamiento automático
- [Supervision](https://github.com/roboflow/supervision) por las herramientas de anotación
- Equipo del proyecto PINV01-1159 por el soporte y colaboración
- Biólogos especialistas por la validación en campo

---

⭐ Si este proyecto te ha sido útil, ¡considera darle una estrella en GitHub!
 
