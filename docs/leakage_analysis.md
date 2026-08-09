# Análisis de fuga de datos — dataset Roboflow v9

Fecha: 2026-08-09 · Dataset: `pinv011159/macroinvertebrados-acuaticos` v9 (19 clases)

## Resumen

El split de Roboflow es **inválido**. Las métricas reportadas hasta ahora
(99.4 % mAP@0.5, recall 100 %) miden memorización de cuadros, no capacidad de
reconocer familias. Además hay un segundo problema, independiente del split,
que sobrevive a cualquier re-partición: **cada familia se fotografió en muy
pocas sesiones**, así que el fondo predice la clase.

## 1. Fuga por ráfaga (confirmada)

Las fotos vienen de una cámara con nombre `AAAA_MMDD_HHMMSS_seq`, disparada
en ráfagas sobre el mismo individuo. Roboflow asigna el split **por imagen**,
al azar, así que cuadros consecutivos del mismo bicho terminan repartidos.

Evidencia (2 403 imágenes fuente únicas, tras descartar copias aumentadas):

| Medición | Resultado |
|---|---|
| Imágenes fuente duplicadas exactas entre splits | 0 (por eso el chequeo MD5 anterior no lo detectó) |
| Mediana de distancia temporal al vecino más cercano de **otro** split | **16 s** |
| Fotos con un vecino de otro split a < 30 s | **68 %** |
| Imágenes de `valid` con un gemelo visual en `train` (coseno > 0.99) | **29.2 %** |
| Ídem `test` | **25.6 %** |
| Imágenes en ráfagas repartidas entre splits (union-find, 60 s / cos 0.95) | **2 099 / 2 403 = 87.3 %** |

Coseno calculado sobre miniaturas 32×32 en gris normalizadas por contraste;
> 0.99 a esa resolución significa, en la práctica, el mismo cuadro.

## 2. Split corregido

`tools/build_clean_split.py` agrupa por espécimen (union-find sobre cercanía
temporal ≤ 60 s **o** similitud visual > 0.95) y asigna **grupos enteros** a un
único split, estratificando por clase (70/15/15).

```bash
python tools/build_clean_split.py datasets/v9 datasets/clean
```

Resultado: 641 grupos, 1 660 / 373 / 370 imágenes, todas las clases presentes
en los tres splits con proporciones dentro del 1 %.

| Fuga residual | valid | test |
|---|---|---|
| coseno > 0.99 con train | **0.0 %** | **0.0 %** |
| coseno > 0.95 | **0.0 %** | **0.0 %** |
| coseno > 0.90 | 15.3 % | 16.2 % |
| mediana | 0.801 | 0.813 |

El residuo > 0.90 es similitud legítima: mismo montaje de laboratorio,
distinto individuo. No es fuga, pero sí es el síntoma del problema 2.

## 3. Confusor de sesión (no lo arregla el split)

12 de 19 familias se capturaron **en un solo día**. Clase, sesión, fondo e
iluminación están confundidos.

Un clasificador trivial de centroide más cercano sobre miniaturas 32×32 en
gris —donde el macroinvertebrado es un borrón de pocos píxeles— alcanza
**39.5 % de accuracy en test** (azar 5.3 %, clase mayoritaria 7.8 %): **7.5×
el azar sin ver al animal**.

```bash
python tools/confound_check.py datasets/clean
```

Implicaciones:

- Un mAP alto en este dataset **no** demuestra generalización a un arroyo
  nuevo. Hay que decirlo en el paper como limitación explícita.
- Mitigación parcial en entrenamiento (ya aplicada en `models/trainer.py`):
  augmentación fuerte de color (`hsv_s=0.8`, `hsv_v=0.5`), geometría
  (`degrees=20`, `scale=0.6`, `flipud=0.5`), `mosaic=1.0` que mezcla fondos
  entre imágenes, `mixup=0.1` y `erasing=0.4`.
- Mitigación real: capturar cada familia en **≥ 3 sesiones distintas**, con
  distinto fondo/bandeja/iluminación, y reservar una sesión completa por
  familia como test *out-of-session*. Ese es el número que hay que reportar.

## 4. Qué esperar

Con el split limpio las métricas van a **bajar mucho** respecto al 99.4 %
publicado. Esa caída es el resultado correcto: es la primera medición
honesta. El README actual documenta números derivados del split contaminado y
hay que rehacerlo tras el primer entrenamiento válido.
