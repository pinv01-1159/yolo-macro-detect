# Cómo publicar esto en una revista indexada, con lo que tenemos

Fecha: 2026-08-11 · Contexto: proyecto PINV01-1159

## El problema de fondo

Si el paper se plantea como "detectamos macroinvertebrados con 99% de precisión", no sobrevive
revisión seria: ya sabemos (ver [`leakage_analysis.md`](leakage_analysis.md)) que ese número
depende de un confusor de sesión que no logramos romper con tres técnicas distintas. Un revisor
que conozca la literatura de *shortcut learning* (Geirhos et al. 2020; Xiao et al. 2020 sobre
fondo en reconocimiento de objetos; Beery et al. 2018 sobre camera traps que aprenden la cámara
en vez del animal) va a hacer exactamente las preguntas que ya nos hicimos nosotros mismos. La
única forma de que esto sea publicable es que **la respuesta a esas preguntas ya esté en el
paper, medida, antes de que las hagan.**

La buena noticia es que eso es exactamente lo que tenemos. No hay que inflar nada — hay que
cambiar qué es "el resultado".

## El argumento central: el contribution no es el 87%, es el proceso de auditoría

Reformular qué es lo publicable:

| Lo que NO alcanza solo | Lo que sí es publicable |
|---|---|
| "Un detector que llega a 87% mAP@0.5:0.95" | Un pipeline reproducible de auditoría de dataset (fuga por ráfaga + confusor de sesión) aplicable a cualquier dataset ecológico chico armado en pocas sesiones de campo/laboratorio |
| Números de rendimiento sin más | Cuantificación exacta de cuánto rendimiento depende de morfología real vs. atajo de fondo, por clase, con metodología reproducible (`tools/background_ablation.py`) |
| "Probamos que funciona" | Evidencia negativa rigurosa: tres mitigaciones estándar de la literatura, documentadas y descartadas con números, no con una frase |

La mayoría de los papers aplicados de CV para ecología/biomonitoreo **no hacen esta auditoría**.
Publicar con ella hecha y documentada es, paradójicamente, más fuerte que publicar un número
alto sin ella — es la diferencia entre "funciona (creemos)" y "esto es lo que funciona, esto es
lo que no, y esto es por qué".

## Qué tipo de venue encaja

No puedo decirte con certeza qué revista específica te va a aceptar (eso depende de alcance,
factor de impacto que busques, e idioma), pero hay tres categorías donde este trabajo tiene
encaje real — vale la pena mirar 2-3 candidatas en cada una y comparar scope/tiempos de revisión:

1. **Informática ambiental / monitoreo ecológico aplicado** (ej. revistas de la familia
   *Ecological Informatics*, *Ecological Indicators*, *Environmental Monitoring and Assessment*,
   o equivalentes regionales de acceso abierto). Encaje: el sistema BMWP + detección es el
   contribution principal, la auditoría de dataset es una sección metodológica robusta que
   distingue el paper de otros similares.
2. **Visión por computadora aplicada / sensores** (revistas tipo MDPI *Sensors*, *Applied
   Sciences*, o similares con foco en sistemas aplicados). Encaje: pipeline reproducible,
   comparación entre arquitecturas, metodología de auditoría como aporte técnico.
3. **Workshop o sección corta sobre shortcut learning / robustez en dominios de datos chicos**
   (si el programa de conferencia lo tiene). Encaje: el hallazgo de que tres mitigaciones
   estándar no alcanzan es interesante *en sí mismo*, independientemente del dominio de
   macroinvertebrados — es un caso de estudio de cuán persistente puede ser un atajo cuando la
   señal espuria es limpia. Esto podría ser un paper corto separado, no compite con el aplicado.

Recomendación: apuntar primero a la categoría 1 (tu campo, revisores van a valorar el rigor
metodológico sin exigir la profundidad de ML de un venue de CV puro), y considerar la categoría 3
como salida adicional para el hallazgo negativo, que es reciclable como contribución independiente.

## Estructura de paper sugerida

1. **Introducción**: necesidad de monitoreo de calidad de agua vía BMWP, costo de identificación
   manual, motivación para automatizar. Mencionar desde acá — no escondido — que el paper incluye
   una auditoría rigurosa del dataset porque los datasets ecológicos chicos son propensos a fuga
   y confusores (citar Beery/Xiao/Geirhos).
2. **Dataset y auditoría**:
   - Descripción del dataset original (19 familias, captura en ráfagas).
   - Fuga por ráfaga encontrada, medida (87.3% de imágenes en ráfagas repartidas, 29.2% de
     `valid` con gemelo visual en `train`) y corregida (`build_clean_split.py`, agrupación por
     espécimen, resultado 0.0% de residuo >0.95).
   - Confusor de sesión: 12/19 familias en una sola sesión, medido con clasificador trivial sobre
     miniaturas (39.5% de accuracy sin ver al bicho, 7.5× el azar).
3. **Métodos**: las tres arquitecturas (YOLO11s/12s/26s), config de entrenamiento, augmentación
   anti-atajo, y **los tres intentos de mitigación** (copy-paste 1x/3x, regularización por
   atención) descritos como experimentos formales, no como anécdota.
4. **Resultados**:
   - Tabla comparativa de las tres arquitecturas (mAP@0.5:0.95 como métrica principal, con
     mAP@0.5 marcado explícitamente como saturado/poco informativo en este dataset).
   - Tabla de clases más débiles (Chironomidae, Ceratopogonidae) con la lectura correcta:
     dificultad genuina de objeto chico, consistente en tres arquitecturas independientes.
   - **Ablación de fondo** como resultado central, no como apéndice: accuracy con el bicho solo
     (97.6-99.8%, evidencia de que la mayoría de las clases se aprenden por morfología real) vs.
     con el fondo solo (~16% de instancias con clase correcta sin bicho visible, concentrado en
     4-5 clases).
   - Tabla de las tres mitigaciones con el resultado nulo (0.157-0.166 en las ocho corridas).
5. **Discusión**: qué significa que ninguna mitigación de entrenamiento haya funcionado — el
   confusor está en la distribución de los datos, no es un problema de optimización. Alcance
   real del sistema: útil bajo condiciones fotográficas controladas de laboratorio, generalización
   a campo no demostrada todavía.
6. **Limitaciones y trabajo futuro**: la mitigación real (captura en ≥3 sesiones por familia,
   test *out-of-session*) como el siguiente paso necesario, explícitamente no hecho en este
   trabajo.
7. **Conclusión**: sistema funcional bajo el alcance declarado, metodología de auditoría
   reproducible y reutilizable para otros datasets ecológicos con la misma estructura de captura.

## Qué conviene sumar antes de mandar (barato, ya lo discutimos)

- **Grad-CAM** sobre una muestra de instancias correctamente clasificadas: evidencia visual
  complementaria de que el modelo atiende al organismo, no al fondo, en el grueso de los casos.
  Barato de generar, buen material de figura.
- **Kappa de Cohen / F1 macro** entre clases (ya tenemos la matriz de confusión cruda por
  modelo) — es el estándar en papers de precisión de identificación taxonómica, más esperable
  para revisores de esta categoría de revista que las métricas puramente de detección.
- Si hay tiempo/recursos: **una sesión de captura adicional** para 2-3 de las familias más
  comprometidas (Chironomidae, Hydrophilidae), aunque sea chica, para poder reportar *aunque sea
  un ejemplo* de test out-of-session real. No hace falta resolver las 19 familias — mostrar que
  el rendimiento se sostiene (o cae, y reportarlo) en un caso real de otra sesión es mucho más
  convincente que solo la ablación sintética.

## Cómo responder de antemano a la objeción obvia del revisor

> "¿Cómo saben que el modelo no está memorizando el fondo?"

Respuesta ya armada, en el paper, con evidencia: *"Lo medimos directamente (Sección X): el
modelo mantiene 97.6-99.8% de accuracy cuando se le oculta todo el contexto excepto el
organismo, y solo en ~16% de las instancias —concentradas en 4 de 19 familias— predice la clase
correcta sin ningún organismo visible. Probamos tres mitigaciones estándar de la literatura para
esas clases específicas; ninguna redujo la dependencia del fondo, lo que sugiere que la señal es
lo bastante limpia en los datos de entrenamiento como para sobrevivir a intervenciones a nivel de
entrenamiento — el problema requiere captura adicional, no una arquitectura o pérdida distinta."*

Esa respuesta convierte la limitación en la parte más citable del paper.
