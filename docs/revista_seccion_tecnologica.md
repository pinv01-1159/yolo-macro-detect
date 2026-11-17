# 2. Materiales y métodos: sistema de identificación por visión por computadora

## 2.1. Adquisición y anotación de imágenes

Los ejemplares recolectados en las campañas de muestreo de 2025 se fotografiaron individualmente entre mayo y diciembre de ese año, sobre bandeja de laboratorio y mediante macrofotografía, con el organismo centrado y ocupando la mayor parte del encuadre. El protocolo se diseñó en función de la tarea taxonómica: la determinación a nivel de familia descansa en caracteres diagnósticos de escala submilimétrica (quetotaxia, venación alar, número de vueltas de la concha, segmentación de apéndices) que deben quedar resueltos en la imagen, y el encuadre ajustado a cada ejemplar normaliza diferencias de tamaño corporal de hasta dos órdenes de magnitud entre familias. Cada ejemplar se registró en ráfagas de tomas consecutivas.

Las imágenes se anotaron con cajas delimitadoras alineadas a los ejes y se preprocesaron con corrección de orientación y redimensionamiento a 640 × 640 px. El conjunto comprende 2 403 imágenes y 3 032 instancias de 19 familias (Ampullariidae, Ancylidae, Belostomatidae, Ceratopogonidae, Chironomidae, Coenagrionidae, Dytiscidae, Gerridae, Glossiphoniidae, Hirudinidae, Hydrophilidae, Hyriidae, Libellulidae, Miridae, Noteridae, Notonectidae, Physidae, Planorbidae y Psychodidae), con una media de 1,26 instancias por imagen (máximo 4) y un único taxón por imagen. La revisión de integridad no detectó imágenes sin anotar, coordenadas fuera de rango ni cajas degeneradas. La totalidad de las instancias pertenece a la categoría de objeto grande del protocolo COCO (área > 96² px), con una caja mediana que ocupa el 52,6 % de la imagen; el sistema opera, por lo tanto, en el régimen de resolución en que los detectores actuales son más fiables.

## 2.2. Partición del conjunto agrupada por espécimen

Un aporte metodológico de este trabajo es la partición del conjunto por espécimen y no por imagen. Cuando cada organismo se fotografía en ráfaga, la partición aleatoria por imagen, habitual en la literatura, distribuye tomas casi idénticas del mismo ejemplar entre entrenamiento y evaluación. En la partición original de este conjunto, el 87,3 % de las imágenes (2 099/2 403) pertenecía a ráfagas repartidas entre subconjuntos, y el 29,2 % de las imágenes de validación tenía un duplicado visual en entrenamiento (similitud coseno > 0,99). Esta forma de fuga de datos infla las métricas sin producir síntomas durante el entrenamiento (Kapoor y Narayanan, 2023), por lo que su control es condición para que las cifras reportadas midan generalización a ejemplares nuevos.

Las imágenes se agruparon por espécimen cuando su marca temporal difería en 60 s o menos, o cuando la similitud coseno entre sus descriptores superaba 0,95; el descriptor es una miniatura de 32 × 32 px en escala de grises, normalizada por contraste. Los grupos se obtuvieron como componentes conexas mediante *union-find* y se asignaron completos a entrenamiento, validación o prueba en proporción 0,70/0,15/0,15, con estratificación por familia y semilla fija (42). Tras la partición, ninguna imagen de validación o prueba supera una similitud de 0,95 con entrenamiento; la similitud residual por encima de 0,90 (15,3 % en validación y 16,2 % en prueba) corresponde a ejemplares distintos de la misma familia fotografiados con el mismo montaje. La partición es reproducible de forma exacta (Tabla 1).

**Tabla 1.** Composición de la partición agrupada por espécimen.

|  | Entrenamiento | Validación | Prueba | Total |
|---|---|---|---|---|
| Imágenes | 1 660 | 373 | 370 | 2 403 |
| Instancias | 2 109 | 465 | 458 | 3 032 |
| Especímenes (grupos independientes) | 473 | 81 | 87 | 641 |

## 2.3. Arquitecturas y entrenamiento

Se compararon tres detectores de una etapa de la familia YOLO en su variante pequeña (*s*), adecuada para su ejecución en hardware de laboratorio: YOLO11s (9,4 M parámetros; Jocher y Qiu, 2024), YOLO12s (9,3 M; Tian et al., 2025), que incorpora módulos de atención por áreas, y YOLO26s (9,5 M en inferencia; Jocher, Qiu, Liu et al., 2026), de arquitectura extremo a extremo que prescinde de la supresión de no máximos (NMS). Las tres partieron de pesos preentrenados en COCO y se entrenaron con Ultralytics 8.4.95 (Jocher, Qiu y Chaurasia, 2026) sobre la misma partición, con idéntica configuración y semilla (42), en una GPU NVIDIA RTX 4050 Laptop de 6 GB con precisión mixta (AMP) y modo determinista.

El entrenamiento se extendió por 200 épocas sin parada temprana, con imágenes de 640 px, tamaño de lote 16 y lote nominal de 64 (acumulación de gradientes en 4 pasos); YOLO12s utilizó un lote de 8 con acumulación en 8 pasos por limitación de memoria, de modo que el lote efectivo fue el mismo en las tres arquitecturas. El optimizador se resolvió mediante la selección automática de Ultralytics, que para este conjunto (19 clases, 5 200 iteraciones) adopta AdamW con tasa de aprendizaje inicial de 4,35 × 10⁻⁴ y momento de 0,9, valores verificados en el registro de cada corrida. La tasa decreció con un calendario coseno hasta el 1 % del valor inicial, tras 3 épocas de calentamiento, con decaimiento de pesos de 5 × 10⁻⁴. Se conservó como modelo final el de mayor mAP@0,5:0,95 en validación.

Para reforzar la generalización con un conjunto de este tamaño, se aplicó aumento de datos en línea más intenso que el predeterminado, orientado a desacoplar la identificación de las condiciones de color, iluminación y orientación de cada sesión fotográfica (Tabla 2). El aumento se aplica solo durante el entrenamiento.

**Tabla 2.** Parámetros de aumento de datos aplicados durante el entrenamiento.

| Transformación | Valor | Predeterminado |
|---|---|---|
| Tono / saturación / brillo (HSV) | 0,02 / 0,8 / 0,5 | 0,015 / 0,7 / 0,4 |
| Rotación (grados) | ±20 | 0 |
| Traslación / escala | 0,15 / 0,6 | 0,1 / 0,5 |
| Cizalladura (grados) | 5 | 0 |
| Volteo horizontal / vertical (probabilidad) | 0,5 / 0,5 | 0,5 / 0 |
| Mosaico (probabilidad) | 1,0 | 1,0 |
| Épocas finales sin mosaico | 15 | 10 |
| MixUp (probabilidad) | 0,1 | 0 |

## 2.4. Protocolo de evaluación

Todas las métricas se calcularon sobre el subconjunto de prueba, que no intervino en el entrenamiento, en la selección del modelo ni en ninguna decisión de configuración. La métrica principal es el mAP@0,5:0,95 del protocolo COCO (Lin et al., 2014), promedio de la precisión media sobre umbrales de IoU entre 0,50 y 0,95 con paso 0,05, calculado con umbral de confianza de 0,001 e IoU de NMS de 0,7 (este último no interviene en YOLO26s). Por exigir una delimitación precisa del organismo, esta métrica discrimina entre modelos donde el mAP@0,5 se satura. La precisión, la exhaustividad y el F1 se reportan en el umbral de confianza que maximiza el F1, según la implementación de Ultralytics.

Dado que las tomas de un mismo espécimen no son observaciones independientes, los intervalos de confianza del 95 % se estimaron mediante *bootstrap* por conglomerados (Efron y Tibshirani, 1993), con 1 000 réplicas sobre los 87 especímenes de prueba, recomputando cada métrica con la misma implementación de Ultralytics que produce las estimaciones puntuales. Las diferencias entre arquitecturas se evaluaron de forma pareada, con las mismas réplicas para los tres modelos. Los intervalos agrupados resultaron entre 1,1 y 2 veces más amplios que los obtenidos remuestreando imágenes, lo que confirma la necesidad de este enfoque.

El umbral de confianza de operación (τ) se fijó mediante un barrido entre 0,05 y 0,95 sobre el subconjunto de validación, con el mismo acondicionamiento de imagen que en operación (sección 2.5) y emparejamiento de predicciones por confianza descendente (IoU ≥ 0,5). El F1 máximo se obtuvo en τ = 0,20 (0,985), sin diferencia significativa respecto de τ = 0,30 (0,984; diferencia pareada por espécimen, IC 95 % [−0,007; 0,011]). Se adoptó τ = 0,30 por su mayor precisión (0,987 frente a 0,977), criterio prioritario en bioindicación: un falso positivo agrega al índice una familia ausente, mientras que un falso negativo solo omite una presente.

## 2.5. Inferencia y cálculo del índice biótico

En operación, cada imagen de 640 × 640 px se amplía con un margen uniforme de 16 px en gris neutro (valor 114) antes de la detección, y las coordenadas se reproyectan luego a la imagen original. Esta decisión responde a la geometría del conjunto: los organismos ocupan la mayor parte del encuadre y con frecuencia tocan sus bordes, y el margen permite al detector delimitar con precisión los ejemplares que alcanzan el límite de la imagen. Se seleccionó sobre el subconjunto de validación, donde elevó el mAP@0,5:0,95 en las tres arquitecturas (YOLO11s: 0,848 → 0,867; YOLO12s: 0,865 → 0,886; YOLO26s: 0,813 → 0,892), y coincide con el acondicionamiento que aplica Ultralytics durante la evaluación, de modo que las métricas de la sección 3 corresponden exactamente a las condiciones de operación. La detección se realiza con τ = 0,30 e IoU de NMS de 0,6.

Las detecciones de todas las imágenes de un sitio de muestreo se agregan por familia y el índice BMWP/Col (Hawkes, 1998; Roldán-Pérez, 2003) se calcula por presencia/ausencia como la suma de los puntajes de las familias detectadas. Este total se asigna a una de cinco clases de calidad (I: > 100; II: 61–100; III: 36–60; IV: 16–35; V: ≤ 15). Se calcula también el ASPT (Armitage et al., 1983), cociente entre el BMWP y el número de familias puntuadas, menos dependiente del esfuerzo de muestreo. Hyriidae y Miridae no figuran en la tabla de referencia; su puntaje se asignó por proximidad taxonómica y se identifica como provisional en cada resultado.

## 2.6. Análisis de dependencia del contexto fotográfico

Como parte de la validación, se evaluó si los detectores identifican la familia a partir del organismo o del contexto fotográfico. El análisis es pertinente porque doce de las 19 familias provienen de una única sesión fotográfica. Como referencia, un clasificador de centroide más cercano entrenado sobre miniaturas de 32 × 32 px en escala de grises, resolución a la que se pierden los caracteres diagnósticos, alcanzó una exactitud del 39,5 % en prueba (azar: 5,3 %), lo que confirma que la sesión contiene información predictiva y justifica medir directamente su uso por parte de los detectores.

Para ello se realizó una ablación con dos variantes de cada una de las 458 instancias de prueba: (a) un recorte a la caja anotada con un margen del 15 % por lado, que conserva el organismo y elimina la mayor parte del contexto, y (b) la imagen completa con la caja del organismo cubierta por un relleno gris uniforme, que conserva solo el contexto. Ambas variantes se procesaron con τ = 0,30. En (a) se registró si la detección de mayor confianza correspondía a la familia correcta, y se comparó con la imagen completa sin modificar. En (b) se registró si alguna detección de la familia correcta recaía sobre la zona cubierta (IoU ≥ 0,3 con la caja o centro dentro de ella), excluyendo en las 51 imágenes con varios ejemplares las detecciones correspondientes a los otros organismos visibles. Los intervalos se estimaron con *bootstrap* por conglomerados sobre los 87 especímenes (2 000 réplicas).

## 3. Resultados del sistema de detección

Las tres arquitecturas alcanzaron un desempeño elevado sobre ejemplares no vistos durante el entrenamiento (Tabla 3), con mAP@0,5 de 0,993 a 0,995 y mAP@0,5:0,95 de 0,864 a 0,881. YOLO12s y YOLO26s superaron a YOLO11s en mAP@0,5:0,95 por 1,5 y 1,6 puntos, con diferencias significativas (IC 95 % pareado: [0,006; 0,026] y [0,007; 0,027], respectivamente), y no se diferenciaron entre sí (−0,002; IC 95 % [−0,009; 0,007]). Se adoptó YOLO26s como modelo de referencia por presentar el mayor valor puntual y por su arquitectura sin NMS, que simplifica el despliegue.

**Tabla 3.** Desempeño sobre el subconjunto de prueba (370 imágenes, 458 instancias, 87 especímenes; protocolo COCO). Entre corchetes, IC 95 % por *bootstrap* agrupado por espécimen.

| Modelo | mAP@0,5 | mAP@0,5:0,95 | Precisión | Exhaustividad | F1 |
|---|---|---|---|---|---|
| YOLO11s | 0,993 | 0,864 [0,853–0,880] | 0,988 | 0,960 | 0,974 [0,957–0,992] |
| YOLO12s | 0,995 | 0,879 [0,872–0,894] | 0,991 | 0,998 | 0,994 [0,988–0,995] |
| YOLO26s | 0,995 | 0,881 [0,871–0,897] | 0,990 | 0,974 | 0,982 [0,971–0,990] |

En el punto de operación (τ = 0,30), YOLO26s alcanzó una precisión de 0,984, una exhaustividad de 0,994 y un F1 de 0,989. El modelo no confundió ninguna familia con otra: la matriz de confusión es diagonal, y los únicos errores son detecciones duplicadas sobre organismos ya identificados correctamente (nueve, una en cada una de nueve familias, con IoU de 0,89 a 0,97 respecto de la caja anotada). En el punto de operación quedan siete duplicados y tres omisiones (Chironomidae, Hydrophilidae y Noteridae). Para el índice biótico, que opera por presencia/ausencia de familias, los duplicados no tienen efecto, dado que no agregan familias nuevas, y una omisión solo lo modifica si afecta a la única toma de esa familia en el sitio.

El desempeño por familia fue consistente entre arquitecturas: 17 de las 19 familias se ubicaron entre 0,82 y 0,94 de mAP@0,5:0,95 (promedio de las tres arquitecturas). Las dos restantes, Chironomidae (0,564–0,633) y Ceratopogonidae (0,751–0,823), son larvas filiformes y curvas, y su menor valor se explica por la geometría de la anotación y no por errores de identificación: el desempeño por familia se correlacionó con la elongación de la caja (r = −0,84; R² = 0,70; p < 0,001; n = 19), ya que un rectángulo alineado a los ejes encierra mayoritariamente sustrato cuando el organismo es alargado y curvo, lo que penaliza el IoU en los umbrales altos aun cuando la familia se identifica sin ambigüedad. Esta relación está concentrada en esas dos familias (sin ellas, r = −0,41; p = 0,10; n = 17).

La ablación de contexto mostró que la identificación descansa en el organismo. El recorte ajustado al organismo igualó o superó la exactitud de la imagen completa en las tres arquitecturas: 98,5 % (IC 95 %: 95,7–100) frente a 98,0 % en YOLO26s, 99,1 % frente a 96,5 % en YOLO11s y 100 % frente a 99,8 % en YOLO12s. Con el organismo cubierto, YOLO12s y YOLO26s no produjeron ninguna detección sobre la zona cubierta (0 de 458 instancias) y YOLO11s produjo una (0,2 %; IC 95 %: 0–0,7): los detectores no generan identificaciones a partir del contexto. La información de sesión que capta el clasificador de miniaturas no se traduce, por lo tanto, en detecciones sin organismo; la evaluación sobre sesiones fotográficas nuevas, prevista como siguiente etapa, permitirá cuantificar su contribución cuando el organismo está presente.

El procedimiento es reproducible de forma exacta: YOLO12s y YOLO26s se regeneraron idénticos en un segundo entrenamiento ejecutado 45 días después, sin ningún tensor distinto (0 de 691 y 0 de 708, respectivamente). El código, los pesos, la configuración efectiva de cada entrenamiento y los artefactos de evaluación están disponibles en https://github.com/pinv01-1159/yolo-macro-detect.

## Referencias

Armitage, P. D., Moss, D., Wright, J. F., y Furse, M. T. (1983). The performance of a new biological water quality score system based on macroinvertebrates over a wide range of unpolluted running-water sites. *Water Research, 17*(3), 333–347. https://doi.org/10.1016/0043-1354(83)90188-4

Efron, B., y Tibshirani, R. J. (1993). *An introduction to the bootstrap*. Chapman & Hall/CRC. https://doi.org/10.1201/9780429246593

Hawkes, H. A. (1998). Origin and development of the Biological Monitoring Working Party score system. *Water Research, 32*(3), 964–968. https://doi.org/10.1016/S0043-1354(97)00275-3

Jocher, G., y Qiu, J. (2024). *Ultralytics YOLO11* (versión 11.0.0) [Software]. GitHub. https://github.com/ultralytics/ultralytics

Jocher, G., Qiu, J., y Chaurasia, A. (2026). *Ultralytics YOLO* (versión 8.4.95) [Software]. Zenodo. https://doi.org/10.5281/zenodo.21346675

Jocher, G., Qiu, J., Liu, M., Lyu, S., Akyon, F. C., y Kalfaoglu, M. E. (2026). *Ultralytics YOLO26: Unified real-time end-to-end vision models* (arXiv:2606.03748) [Preprint]. arXiv. https://doi.org/10.48550/arXiv.2606.03748

Kapoor, S., y Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns, 4*(9), Artículo 100804. https://doi.org/10.1016/j.patter.2023.100804

Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., y Zitnick, C. L. (2014). Microsoft COCO: Common objects in context. En D. Fleet, T. Pajdla, B. Schiele y T. Tuytelaars (Eds.), *Computer vision – ECCV 2014* (Lecture Notes in Computer Science, Vol. 8693, pp. 740–755). Springer. https://doi.org/10.1007/978-3-319-10602-1_48

Roldán-Pérez, G. (2003). *Bioindicación de la calidad del agua en Colombia: Propuesta para el uso del método BMWP/Col*. Editorial Universidad de Antioquia.

Tian, Y., Ye, Q., y Doermann, D. (2025). YOLOv12: Attention-centric real-time object detectors. En *Advances in Neural Information Processing Systems 38*. Curran Associates. https://doi.org/10.52202/085713-2627
