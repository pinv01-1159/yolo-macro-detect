#!/usr/bin/env python3
"""
Ejemplo de uso del sistema YOLO Macroinvertebrados.

Este script demuestra cómo usar las diferentes funcionalidades
del sistema de detección de macroinvertebrados, incluyendo
evaluación de calidad del agua mediante el índice BMWP.
"""

import os
import sys
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from main import MacroinvertebratePipeline
from utils.bmwp_calculator import bmwp_calculator


def example_pipeline_complete():
    """
    Ejemplo de pipeline completo.
    """
    print("🔄 Ejemplo: Pipeline Completo")
    print("=" * 40)

    try:
        # Inicializar pipeline
        pipeline = MacroinvertebratePipeline()

        # Ejecutar pipeline completo
        model_path = pipeline.run_complete_pipeline(
            version=5,  # Versión específica del dataset
            epochs=10,  # Pocas épocas para el ejemplo
            experiment_name="ejemplo_pipeline"
        )

        print(f"✅ Pipeline completado! Modelo: {model_path}")

    except Exception as e:
        print(f"❌ Error: {e}")


def example_dataset_setup():
    """
    Ejemplo de configuración de dataset.
    """
    print("\n📥 Ejemplo: Configuración de Dataset")
    print("=" * 40)

    try:
        # Inicializar pipeline
        pipeline = MacroinvertebratePipeline()

        # Configurar dataset
        data_yaml_path = pipeline.setup_dataset(version=5)

        print(f"✅ Dataset configurado! data.yaml: {data_yaml_path}")

    except Exception as e:
        print(f"❌ Error: {e}")


def example_training_only():
    """
    Ejemplo de solo entrenamiento.
    """
    print("\n🏋️ Ejemplo: Solo Entrenamiento")
    print("=" * 40)

    try:
        # Inicializar pipeline
        pipeline = MacroinvertebratePipeline()

        # Ruta al data.yaml (asumiendo que ya existe)
        data_yaml_path = "datasets/data.yaml"

        if not os.path.exists(data_yaml_path):
            print("⚠️ data.yaml no encontrado. Configurando dataset primero...")
            data_yaml_path = pipeline.setup_dataset(version=5)

        # Entrenar modelo
        model_path = pipeline.train_model(
            data_yaml_path=data_yaml_path,
            experiment_name="ejemplo_entrenamiento",
            epochs=5  # Pocas épocas para el ejemplo
        )

        print(f"✅ Entrenamiento completado! Modelo: {model_path}")

    except Exception as e:
        print(f"❌ Error: {e}")


def example_inference_only():
    """
    Ejemplo de solo inferencia.
    """
    print("\n🔍 Ejemplo: Solo Inferencia")
    print("=" * 40)

    try:
        # Inicializar pipeline
        pipeline = MacroinvertebratePipeline()

        # Ruta al modelo (asumiendo que ya existe)
        model_path = "runs/detect/macros/weights/best.pt"

        if not os.path.exists(model_path):
            print("⚠️ Modelo no encontrado. Usando modelo de ejemplo...")
            # Aquí podrías descargar un modelo pre-entrenado
            model_path = "models/example_model.pt"

        # Ruta a imagen de ejemplo
        image_path = "examples/sample_image.jpg"

        if not os.path.exists(image_path):
            print("⚠️ Imagen de ejemplo no encontrada. Creando imagen dummy...")
            # Crear imagen dummy para el ejemplo
            import cv2
            import numpy as np

            dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            cv2.imwrite(image_path, dummy_image)

        # Realizar predicción
        results = pipeline.predict_image(
            image_path=image_path,
            model_path=model_path,
            conf_threshold=0.3,
            save_annotated=True
        )

        print("✅ Predicción completada!")
        print(f"   - Total detecciones: {results['total_detecciones']}")
        print(f"   - Familias detectadas: {results['familias_detectadas']}")

        for det in results['detecciones']:
            print(
                f"   - {det['familia']}: {det['cantidad']} "
                f"(conf: {det['confidence_promedio']:.3f})"
            )

    except Exception as e:
        print(f"❌ Error: {e}")


def example_inference_with_bmwp():
    """
    Ejemplo de inferencia con cálculo BMWP.
    """
    print("\n🌊 Ejemplo: Inferencia con Cálculo BMWP")
    print("=" * 40)

    try:
        # Inicializar pipeline
        pipeline = MacroinvertebratePipeline()

        # Ruta al modelo (asumiendo que ya existe)
        model_path = "runs/detect/macros/weights/best.pt"

        if not os.path.exists(model_path):
            print("⚠️ Modelo no encontrado. Usando modelo de ejemplo...")
            model_path = "models/example_model.pt"

        # Ruta a imagen de ejemplo
        image_path = "examples/sample_image.jpg"

        if not os.path.exists(image_path):
            print("⚠️ Imagen de ejemplo no encontrada. Creando imagen dummy...")
            import cv2
            import numpy as np

            dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            cv2.imwrite(image_path, dummy_image)

        # Realizar predicción con BMWP
        results = pipeline.predict_image(
            image_path=image_path,
            model_path=model_path,
            conf_threshold=0.3,
            save_annotated=True,
            calculate_bmwp=True  # ¡Aquí está la diferencia!
        )

        print("✅ Predicción con BMWP completada!")
        print(f"   - Total detecciones: {results['total_detecciones']}")
        print(f"   - Familias detectadas: {results['familias_detectadas']}")

        for det in results['detecciones']:
            print(
                f"   - {det['familia']}: {det['cantidad']} "
                f"(conf: {det['confidence_promedio']:.3f})"
            )

        # Mostrar resultados BMWP
        if 'bmwp_total' in results:
            print("\n🌊 Evaluación de Calidad del Agua (BMWP):")
            print(f"   - Puntaje total: {results['bmwp_total']}")
            print(f"   - Calidad del agua: {results['calidad_agua']}")
            print(f"   - Confianza: {results['confianza']:.3f}")

            if results['detalles_familias']:
                print("   - Detalles por familia:")
                for det in results['detalles_familias']:
                    print(f"     * {det['familia']}: {det['cantidad']} (BMWP: {det['bmwp']})")

    except Exception as e:
        print(f"❌ Error: {e}")


def example_bmwp_calculator_direct():
    """
    Ejemplo de uso directo de la calculadora BMWP.
    """
    print("\n🧮 Ejemplo: Uso Directo de Calculadora BMWP")
    print("=" * 40)

    try:
        # Simular detecciones
        detections = [
            {
                "familia": "Physidae",
                "cantidad": 6,
                "confidence_promedio": 0.95
            },
            {
                "familia": "Planorbidae",
                "cantidad": 4,
                "confidence_promedio": 0.92
            },
            {
                "familia": "Chironomidae",
                "cantidad": 3,
                "confidence_promedio": 0.88
            },
            {
                "familia": "Hirudinidae",
                "cantidad": 2,
                "confidence_promedio": 0.96
            }
        ]

        print("📊 Detecciones simuladas:")
        for det in detections:
            print(f"   - {det['familia']}: {det['cantidad']} individuos")

        # Calcular BMWP directamente
        bmwp_result = bmwp_calculator.calculate_bmwp(detections)

        print("\n✅ Resultados BMWP:")
        print(f"   - Puntaje total: {bmwp_result.total_score}")
        print(f"   - Calidad del agua: {bmwp_result.water_quality_description}")
        print(f"   - Confianza: {bmwp_result.confidence:.3f}")

        # Mostrar detalles por familia
        print("\n📋 Contribución por familia:")
        for score in bmwp_result.family_scores:
            print(
                f"   - {score['familia']}: {score['cantidad']} × "
                f"{score['bmwp_individual']} = {score['bmwp_contribution']} puntos"
            )

    except Exception as e:
        print(f"❌ Error: {e}")


def example_bmwp_family_scores():
    """
    Ejemplo que muestra los puntajes BMWP disponibles.
    """
    print("\n📋 Ejemplo: Puntajes BMWP Disponibles")
    print("=" * 40)

    try:
        print("🏷️ Familias y sus puntajes BMWP:")
        for family, score in bmwp_calculator.FAMILY_SCORES.items():
            print(f"   - {family}: {score} puntos")

        print("\n📊 Clases de calidad del agua:")
        for class_name, criteria in bmwp_calculator.WATER_QUALITY_CLASSES.items():
            print(
                f"   - Clase {class_name}: {criteria['min']}-{criteria['max']} puntos "
                f"→ {criteria['description']}"
            )

        print("\n💡 Información adicional:")
        print(f"   - Familias disponibles: {len(bmwp_calculator.get_available_families())}")
        print(f"   - Puntaje máximo posible: {sum(bmwp_calculator.FAMILY_SCORES.values())}")

    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """
    Función principal que ejecuta todos los ejemplos.
    """
    print("🦐 Ejemplos de Uso - Sistema YOLO Macroinvertebrados")
    print("=" * 60)

    try:
        # Ejemplo 1: Pipeline completo
        example_pipeline_complete()

        # Ejemplo 2: Configuración de dataset
        example_dataset_setup()

        # Ejemplo 3: Solo entrenamiento
        example_training_only()

        # Ejemplo 4: Solo inferencia
        example_inference_only()

        # Ejemplo 5: Inferencia con BMWP
        example_inference_with_bmwp()

        # Ejemplo 6: Calculadora BMWP directa
        example_bmwp_calculator_direct()

        # Ejemplo 7: Puntajes BMWP disponibles
        example_bmwp_family_scores()

        print("\n🎉 Todos los ejemplos ejecutados exitosamente!")
        print("\n💡 Comandos útiles:")
        print("   # Predicción con BMWP")
        print("   python main.py --predict --image sample.jpg --model best.pt --calculate-bmwp")
        print("   ")
        print("   # Pipeline completo")
        print("   python main.py --pipeline-complete --epochs 50")
        print("   ")
        print("   # Solo configuración de dataset")
        print("   python main.py --setup-dataset --dataset-version 5")

    except Exception as e:
        print(f"❌ Error ejecutando ejemplos: {e}")


if __name__ == "__main__":
    main()
