#!/usr/bin/env python3
"""
Pipeline de Detección de Macroinvertebrados Acuáticos
====================================================

Pipeline completo para detección de macroinvertebrados usando YOLOv8.
Incluye descarga de dataset, entrenamiento, evaluación e inferencia
con cálculo automático del índice BMWP para evaluación de calidad del agua.

Autor: Kevin Galeano
Proyecto: PINV01-1159
Fecha: 2024
"""

import argparse
import sys
from pathlib import Path

from config import config
from data import DatasetManager
from models import YOLOInference, YOLOTrainer
from utils.logger import setup_logger


class MacroinvertebratePipeline:
    """
    Pipeline completo para detección de macroinvertebrados acuáticos.
    
    Esta clase maneja todo el flujo de trabajo desde la descarga del dataset
    hasta la inferencia con modelos entrenados, incluyendo evaluación de
    calidad del agua mediante el índice BMWP.
    """

    def __init__(self):
        """Inicializa el pipeline."""
        self.logger = setup_logger("macroinvertebrate_pipeline")
        self.dataset_manager = DatasetManager()
        self.trainer = None
        self.inference = None

        # Validar configuración
        if not config.validate():
            self.logger.error("❌ Configuración inválida. Revisa tu archivo .env")
            sys.exit(1)

        self.logger.info("✅ Pipeline inicializado correctamente")
        self.logger.info(str(config))

    def setup_dataset(self, version: int | None = None) -> str:
        """
        Configura y descarga el dataset.
        
        Args:
            version: Versión específica del dataset (opcional)
            
        Returns:
            Ruta al archivo data.yaml
        """
        self.logger.info("📥 Configurando dataset...")

        try:
            # Configurar conexión con Roboflow
            self.dataset_manager.setup_roboflow_connection()

            # Descargar dataset
            dataset_info = self.dataset_manager.download_dataset(version=version)

            # Validar estructura
            self.dataset_manager.validate_dataset_structure(dataset_info["location"])

            # Generar data.yaml
            data_yaml_path = self.dataset_manager.generate_data_yaml(dataset_info["location"])

            self.logger.info("✅ Dataset configurado exitosamente")
            self.logger.info(f"   - Ubicación: {dataset_info['location']}")
            self.logger.info(f"   - data.yaml: {data_yaml_path}")

            return data_yaml_path

        except Exception as e:
            self.logger.error(f"❌ Error configurando dataset: {e}")
            raise

    def train_model(self,
                   data_yaml_path: str,
                   experiment_name: str | None = None,
                   epochs: int | None = None) -> str:
        """
        Entrena el modelo YOLO.
        
        Args:
            data_yaml_path: Ruta al archivo data.yaml
            experiment_name: Nombre del experimento
            epochs: Número de épocas
            
        Returns:
            Ruta al modelo entrenado
        """
        self.logger.info("🏋️ Iniciando entrenamiento del modelo...")

        try:
            # Inicializar trainer
            experiment_name = experiment_name or config.experiment_name
            self.trainer = YOLOTrainer(experiment_name)

            # Cargar modelo base
            self.trainer.load_model(config.model_name)

            # Entrenar modelo
            model_path = self.trainer.train(
                data_yaml_path=data_yaml_path,
                epochs=epochs or config.training_epochs,
                img_size=config.img_size,
                batch_size=config.batch_size,
                workers=config.workers
            )

            # Evaluar modelo
            self.logger.info("📊 Evaluando modelo entrenado...")
            metrics = self.trainer.evaluate(model_path, data_yaml_path)

            # Obtener resumen
            summary = self.trainer.get_training_summary()
            self.logger.info("📈 Resumen del entrenamiento:")
            self.logger.info(f"   - mAP50: {summary['metrics']['map50']:.4f}")
            self.logger.info(f"   - Precisión: {summary['metrics']['precision']:.4f}")
            self.logger.info(f"   - Recall: {summary['metrics']['recall']:.4f}")

            return model_path

        except Exception as e:
            self.logger.error(f"❌ Error durante el entrenamiento: {e}")
            raise

    def run_complete_pipeline(self,
                            version: int | None = None,
                            epochs: int | None = None,
                            experiment_name: str | None = None) -> str:
        """
        Ejecuta el pipeline completo.
        
        Args:
            version: Versión del dataset
            epochs: Número de épocas
            experiment_name: Nombre del experimento
            
        Returns:
            Ruta al modelo entrenado
        """
        self.logger.info("🚀 Iniciando pipeline completo...")

        try:
            # 1. Configurar dataset
            data_yaml_path = self.setup_dataset(version=version)

            # 2. Entrenar modelo
            model_path = self.train_model(
                data_yaml_path=data_yaml_path,
                experiment_name=experiment_name,
                epochs=epochs
            )

            self.logger.info("🎉 Pipeline completo finalizado exitosamente!")
            return model_path

        except Exception as e:
            self.logger.error(f"❌ Error en pipeline completo: {e}")
            raise

    def predict_image(self,
                     image_path: str,
                     model_path: str,
                     conf_threshold: float | None = None,
                     save_annotated: bool = True,
                     calculate_bmwp: bool = False) -> dict:
        """
        Realiza predicción en una imagen.
        
        Args:
            image_path: Ruta a la imagen
            model_path: Ruta al modelo entrenado
            conf_threshold: Umbral de confianza
            save_annotated: Si guardar imagen anotada
            calculate_bmwp: Si calcular el índice BMWP
            
        Returns:
            Resultados de la predicción
        """
        self.logger.info(f"🔍 Realizando predicción en: {image_path}")

        try:
            # Inicializar inferencia
            self.inference = YOLOInference(model_path)

            # Realizar predicción
            results = self.inference.predict_image(
                image_path=image_path,
                conf_threshold=conf_threshold,
                save_annotated=save_annotated,
                calculate_bmwp=calculate_bmwp
            )

            # Exportar resultados
            output_file = f"results/prediction_{Path(image_path).stem}.json"
            self.inference.export_results(results, output_file)

            return results

        except Exception as e:
            self.logger.error(f"❌ Error durante la predicción: {e}")
            raise


def main():
    """
    Función principal del script.
    
    Maneja los argumentos de línea de comandos y ejecuta
    las operaciones correspondientes.
    """
    parser = argparse.ArgumentParser(
        description="Pipeline de Detección de Macroinvertebrados Acuáticos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Pipeline completo
  python main.py --pipeline-complete

  # Solo configurar dataset
  python main.py --setup-dataset

  # Solo entrenamiento
  python main.py --train --data-yaml datasets/data.yaml

  # Solo predicción
  python main.py --predict --image test.jpg --model runs/detect/macros/weights/best.pt

  # Predicción con cálculo BMWP
  python main.py --predict --image sample.jpg --model best_model.pt --calculate-bmwp
        """
    )

    # Argumentos principales
    parser.add_argument(
        "--pipeline-complete",
        action="store_true",
        help="Ejecutar pipeline completo (dataset + entrenamiento)"
    )

    parser.add_argument(
        "--setup-dataset",
        action="store_true",
        help="Solo configurar y descargar dataset"
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Solo entrenar modelo"
    )

    parser.add_argument(
        "--predict",
        action="store_true",
        help="Solo realizar predicción"
    )

    # Argumentos de configuración
    parser.add_argument(
        "--dataset-version",
        type=int,
        help="Versión específica del dataset"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        help="Número de épocas para entrenamiento"
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Nombre del experimento"
    )

    parser.add_argument(
        "--data-yaml",
        type=str,
        help="Ruta al archivo data.yaml"
    )

    parser.add_argument(
        "--image",
        type=str,
        help="Ruta a la imagen para predicción"
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Ruta al modelo entrenado"
    )

    parser.add_argument(
        "--confidence",
        type=float,
        help="Umbral de confianza para predicción"
    )

    # Argumentos adicionales
    parser.add_argument(
        "--save-annotated",
        action="store_true",
        default=True,
        help="Guardar imagen anotada (por defecto: True)"
    )

    parser.add_argument(
        "--calculate-bmwp",
        action="store_true",
        help="Calcular índice BMWP para evaluación de calidad del agua"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directorio de salida (por defecto: results)"
    )

    args = parser.parse_args()

    # Inicializar pipeline
    pipeline = MacroinvertebratePipeline()

    try:
        if args.pipeline_complete:
            # Pipeline completo
            model_path = pipeline.run_complete_pipeline(
                version=args.dataset_version,
                epochs=args.epochs,
                experiment_name=args.experiment_name
            )
            print(f"\n🎉 Pipeline completado! Modelo guardado en: {model_path}")

        elif args.setup_dataset:
            # Solo configurar dataset
            data_yaml_path = pipeline.setup_dataset(version=args.dataset_version)
            print(f"\n✅ Dataset configurado! data.yaml en: {data_yaml_path}")

        elif args.train:
            # Solo entrenamiento
            if not args.data_yaml:
                print("❌ Error: --data-yaml es requerido para entrenamiento")
                sys.exit(1)

            model_path = pipeline.train_model(
                data_yaml_path=args.data_yaml,
                experiment_name=args.experiment_name,
                epochs=args.epochs
            )
            print(f"\n✅ Entrenamiento completado! Modelo guardado en: {model_path}")

        elif args.predict:
            # Solo predicción
            if not args.image or not args.model:
                print("❌ Error: --image y --model son requeridos para predicción")
                sys.exit(1)

            results = pipeline.predict_image(
                image_path=args.image,
                model_path=args.model,
                conf_threshold=args.confidence,
                save_annotated=args.save_annotated,
                calculate_bmwp=args.calculate_bmwp
            )

            print("\n🔍 Predicción completada!")
            print(f"   - Total detecciones: {results['total_detecciones']}")
            print(f"   - Familias detectadas: {results['familias_detectadas']}")

            for det in results['detecciones']:
                print(f"   - {det['familia']}: {det['cantidad']} (conf: {det['confidence_promedio']:.3f})")

            # Mostrar resultados BMWP si se calculó
            if args.calculate_bmwp and 'bmwp_total' in results:
                print("\n🌊 Evaluación de Calidad del Agua (BMWP):")
                print(f"   - Puntaje total: {results['bmwp_total']}")
                print(f"   - Calidad del agua: {results['calidad_agua']}")
                print(f"   - Confianza: {results['confianza']:.3f}")

                if results['detalles_familias']:
                    print("   - Detalles por familia:")
                    for det in results['detalles_familias']:
                        print(f"     * {det['familia']}: {det['cantidad']} (BMWP: {det['bmwp']})")

        else:
            # Mostrar ayuda si no se especifican argumentos
            parser.print_help()

    except KeyboardInterrupt:
        print("\n⚠️ Operación cancelada por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
