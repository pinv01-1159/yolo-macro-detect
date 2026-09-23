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
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from config import config
from data import DatasetManager
from models import YOLOInference, YOLOTrainer
from models.trainer import MAP_EVAL_CONF, MAP_EVAL_IOU
from reports import DatasetReport, ModelReport
from utils.logger import setup_logger


def _capture_environment_metadata(
    data_yaml_path: str,
    experiment_name: str,
    epochs: int,
    started_at: str,
    train_kwargs: dict | None = None,
    eval_summary: dict | None = None,
) -> dict:
    """Metadata de reproducibilidad para acompañar los resultados crudos:
    versiones, hardware, commit de código y config efectiva del run.
    No se puede reconstruir después de que el entrenamiento termina."""
    import torch
    import ultralytics

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_commit = "unknown"

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None

    # Se guarda el hash del manifiesto, no su contenido: embeberlo duplicaba
    # ~98 KB por experimento (el 99 % es el mapa imagen->especimen, identico en
    # todos) y creaba copias que pueden divergir del original (auditoria S3).
    dataset_dir = Path(data_yaml_path).parent
    split_report_path = dataset_dir / "split_report.json"
    split_report_ref = None
    if split_report_path.exists():
        raw = split_report_path.read_bytes()
        summary = json.loads(raw)
        split_report_ref = {
            "path": str(split_report_path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "groups": summary.get("groups"),
            "images": summary.get("images"),
            "images_per_split": summary.get("images_per_split"),
            "groups_per_split": summary.get("groups_per_split"),
        }

    return {
        "experiment_name": experiment_name,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "ultralytics_version": ultralytics.__version__,
        "gpu": gpu_name,
        "config": {
            "model_name": config.model_name,
            "epochs": epochs,
            "img_size": config.img_size,
            "batch_size": config.batch_size,
            "workers": config.workers,
            "seed": config.seed,
        },
        # Configuracion efectiva completa que recibio Ultralytics, augmentacion
        # incluida. `config` de arriba es un resumen; esto es lo que de verdad
        # hace falta para reproducir el entrenamiento.
        "train_kwargs": train_kwargs or None,
        # Metricas del modelo entrenado, para cerrar la cadena
        # numero publicado -> artefacto -> commit -> config.
        "eval_metrics": eval_summary or None,
        "data_yaml_path": str(data_yaml_path),
        "dataset_split_report": split_report_ref,
    }


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
            self.logger.error("Configuración inválida. Revisa tu archivo .env:")
            for error in config.validation_errors():
                self.logger.error(f"   - {error}")
            sys.exit(1)
        for aviso in config.validation_errors():
            self.logger.warning(f"{aviso}")

        self.logger.info("Pipeline inicializado correctamente")
        self.logger.info(str(config))

    def setup_dataset(self, version: int | None = None) -> str:
        """
        Configura y descarga el dataset.

        Args:
            version: Versión específica del dataset (opcional)

        Returns:
            Ruta al archivo data.yaml
        """
        self.logger.info("Configurando dataset...")

        try:
            # Configurar conexión con Roboflow
            self.dataset_manager.setup_roboflow_connection()

            # Descargar dataset
            dataset_info = self.dataset_manager.download_dataset(version=version)

            # Validar estructura
            self.dataset_manager.validate_dataset_structure(dataset_info["location"])

            # Generar data.yaml
            data_yaml_path = self.dataset_manager.generate_data_yaml(dataset_info["location"])

            self.logger.info("Dataset configurado exitosamente")
            self.logger.info(f"   - Ubicación: {dataset_info['location']}")
            self.logger.info(f"   - data.yaml: {data_yaml_path}")

            return data_yaml_path

        except Exception as e:
            self.logger.error(f"Error configurando dataset: {e}")
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
        self.logger.info("Iniciando entrenamiento del modelo...")
        started_at = datetime.now(timezone.utc).isoformat()

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

            # Evaluar en el split de test (no val: ese ya se usó para elegir best.pt)
            self.logger.info("Evaluando modelo en el split de test...")
            eval_metrics = self.trainer.evaluate(model_path, data_yaml_path)

            # Obtener resumen y persistirlo como registro del experimento
            summary = self.trainer.get_training_summary()
            self.logger.info("Resumen de evaluación (split de test):")
            self.logger.info(f"   - mAP50: {summary['metrics']['map50']:.4f}")
            self.logger.info(f"   - Precisión: {summary['metrics']['precision']:.4f}")
            self.logger.info(f"   - Recall: {summary['metrics']['recall']:.4f}")

            results_dir = (Path("results") / experiment_name).resolve()
            results_dir.mkdir(parents=True, exist_ok=True)
            with open(results_dir / "eval_metrics.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            self.logger.info(f"   - Métricas guardadas en: {results_dir / 'eval_metrics.json'}")

            env_metadata = _capture_environment_metadata(
                data_yaml_path=data_yaml_path,
                experiment_name=experiment_name,
                epochs=epochs or config.training_epochs,
                started_at=started_at,
                train_kwargs=getattr(self.trainer, "train_kwargs", None),
                eval_summary=summary,
            )
            with open(results_dir / "environment.json", "w", encoding="utf-8") as f:
                json.dump(env_metadata, f, indent=2, ensure_ascii=False)
            env_path = results_dir / "environment.json"
            self.logger.info(f"   - Metadata de entorno guardada en: {env_path}")

            self.logger.info("Generando reporte estadístico del modelo...")
            ModelReport(model_path, data_yaml_path).generate(
                metrics=eval_metrics,
                output_dir=results_dir / "model_report",
                # Quien evaluó conoce el protocolo con certeza; deducirlo del
                # objeto de métricas depende de la versión de Ultralytics.
                protocol={
                    "split": "test",
                    "conf": MAP_EVAL_CONF,
                    "iou": MAP_EVAL_IOU,
                    "imgsz": config.img_size,
                },
            )

            return model_path

        except Exception as e:
            self.logger.error(f"Error durante el entrenamiento: {e}")
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
        self.logger.info("Iniciando pipeline completo...")

        try:
            # 1. Configurar dataset
            data_yaml_path = self.setup_dataset(version=version)

            # 2. Entrenar modelo
            model_path = self.train_model(
                data_yaml_path=data_yaml_path,
                experiment_name=experiment_name,
                epochs=epochs
            )

            self.logger.info("Pipeline completo finalizado exitosamente!")
            return model_path

        except Exception as e:
            self.logger.error(f"Error en pipeline completo: {e}")
            raise

    def predict_site(self,
                    images_dir: str,
                    model_path: str,
                    conf_threshold: float | None = None,
                    iou_threshold: float | None = None,
                    save_annotated: bool = True,
                    output_dir: str = "results") -> dict:
        """Procesa todas las fotografías de un sitio de muestreo.

        Es el camino que el índice BMWP necesita: se define por sitio, no por
        fotografía. Devuelve el registro agregado, con el índice, el ASPT y las
        advertencias.
        """
        carpeta = Path(images_dir)
        if not carpeta.is_dir():
            raise NotADirectoryError(f"No es una carpeta: {images_dir}")
        extensiones = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        fotos = sorted(
            f for f in carpeta.iterdir()
            if f.is_file() and f.suffix.lower() in extensiones
        )
        if not fotos:
            raise FileNotFoundError(f"No hay imágenes en {images_dir}")

        self.logger.info(f"Procesando sitio: {len(fotos)} fotografías de {carpeta}")
        self.inference = YOLOInference(model_path)
        resultados = self.inference.predict_batch(
            [str(f) for f in fotos],
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            save_annotated=save_annotated,
            output_dir=output_dir,
            calculate_bmwp=True,
        )
        salida = Path(output_dir) / f"sitio_{carpeta.name}.json"
        self.inference.export_results(resultados, str(salida))
        return resultados[-1]  # el registro con alcance="sitio"

    def predict_image(self,
                     image_path: str,
                     model_path: str,
                     conf_threshold: float | None = None,
                     iou_threshold: float | None = None,
                     save_annotated: bool = True,
                     calculate_bmwp: bool = False,
                     output_dir: str = "results") -> dict:
        """
        Realiza predicción en una imagen.

        Args:
            image_path: Ruta a la imagen
            model_path: Ruta al modelo entrenado
            conf_threshold: Umbral de confianza
            iou_threshold: Umbral de IoU para NMS
            save_annotated: Si guardar imagen anotada
            calculate_bmwp: Si calcular el índice BMWP
            output_dir: Directorio donde guardar la imagen anotada y el JSON

        Returns:
            Resultados de la predicción
        """
        self.logger.info(f"Realizando predicción en: {image_path}")

        try:
            # Inicializar inferencia
            self.inference = YOLOInference(model_path)

            # Realizar predicción
            results = self.inference.predict_image(
                image_path=image_path,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                save_annotated=save_annotated,
                calculate_bmwp=calculate_bmwp,
                output_dir=output_dir,
            )

            # Exportar resultados
            output_file = str(
                Path(output_dir) / f"prediction_{Path(image_path).stem}.json"
            )
            self.inference.export_results(results, output_file)

            return results

        except Exception as e:
            self.logger.error(f"Error durante la predicción: {e}")
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

    parser.add_argument(
        "--dataset-report",
        action="store_true",
        help="Generar reporte estadístico de auditoría del dataset (requiere --data-yaml)"
    )

    parser.add_argument(
        "--model-report",
        action="store_true",
        help="Generar reporte estadístico de evaluación del modelo (requiere --model y --data-yaml)"
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
        "--site",
        type=str,
        help=("Carpeta con las fotografías de un sitio de muestreo. Calcula el "
              "índice BMWP del sitio, que es como se define el índice.")
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

    parser.add_argument(
        "--iou",
        type=float,
        help="Umbral de IoU (NMS) para predicción"
    )

    # Argumentos adicionales
    parser.add_argument(
        "--save-annotated",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Guardar imagen anotada (por defecto: sí; usar --no-save-annotated)"
    )

    # El default sale de ENABLE_BMWP: antes la variable se leia, se imprimia
    # en el banner y no controlaba nada, con lo que el programa podia anunciar
    # "BMWP: Habilitado: True" y dos lineas despues "Cálculo BMWP: False".
    parser.add_argument(
        "--calculate-bmwp",
        action=argparse.BooleanOptionalAction,
        default=config.enable_bmwp,
        help=("Calcular índice BMWP de calidad del agua "
              f"(por defecto: {config.enable_bmwp}, vía ENABLE_BMWP)")
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
            print(f"\nPipeline completado! Modelo guardado en: {model_path}")

        elif args.setup_dataset:
            # Solo configurar dataset
            data_yaml_path = pipeline.setup_dataset(version=args.dataset_version)
            print(f"\nDataset configurado! data.yaml en: {data_yaml_path}")

        elif args.train:
            # Solo entrenamiento
            if not args.data_yaml:
                print("Error: --data-yaml es requerido para entrenamiento")
                sys.exit(1)

            model_path = pipeline.train_model(
                data_yaml_path=args.data_yaml,
                experiment_name=args.experiment_name,
                epochs=args.epochs
            )
            print(f"\nEntrenamiento completado! Modelo guardado en: {model_path}")

        elif args.site:
            if not args.model:
                print("Error: --model es requerido para --site")
                sys.exit(1)

            sitio = pipeline.predict_site(
                images_dir=args.site,
                model_path=args.model,
                conf_threshold=args.confidence,
                iou_threshold=args.iou,
                save_annotated=args.save_annotated,
                output_dir=args.output_dir,
            )

            print(f"\nSitio procesado: {sitio['n_imagenes']} fotografías")
            print(f"\nÍndice BMWP del sitio: {sitio['bmwp_total']}")
            print(f"   - ASPT: {sitio['aspt']}")
            print(f"   - Familias con puntaje: {sitio['n_familias_puntuadas']}")
            print(f"   - Calidad del agua: {sitio['calidad_agua']}")
            if sitio['familias']:
                print("   - Familias encontradas:")
                for f in sitio['familias']:
                    marca = " (provisional)" if f.get("procedencia") == "proximidad" else ""
                    print(f"     * {f['familia']:18} BMWP {f['bmwp_individual']}{marca}")
            for aviso in sitio.get('advertencias', []):
                print(f"    {aviso}")

        elif args.predict:
            # Solo predicción
            if not args.image or not args.model:
                print("Error: --image y --model son requeridos para predicción")
                sys.exit(1)

            results = pipeline.predict_image(
                image_path=args.image,
                model_path=args.model,
                conf_threshold=args.confidence,
                iou_threshold=args.iou,
                save_annotated=args.save_annotated,
                calculate_bmwp=args.calculate_bmwp,
                output_dir=args.output_dir,
            )

            print("\nPredicción completada!")
            print(f"   - Total detecciones: {results['total_detecciones']}")
            print(f"   - Familias detectadas: {results['familias_detectadas']}")

            for det in results['detecciones']:
                print(
                    f"   - {det['familia']}: {det['cantidad']} "
                    f"(conf: {det['confidence_promedio']:.3f})"
                )

            # Mostrar resultados BMWP si se calculó
            if args.calculate_bmwp and 'bmwp_total' in results:
                parcial = results.get('parcial', False)
                titulo = "Aporte BMWP de esta imagen" if parcial else "Índice BMWP del sitio"
                print(f"\n{titulo}:")
                print(f"   - Puntaje: {results['bmwp_total']}")
                print(f"   - ASPT: {results.get('aspt', 0.0)}")
                # La clase de calidad solo se enuncia a nivel de sitio: ponerla
                # bajo una foto suelta convierte un dato parcial en un veredicto
                # ambiental que nadie midió.
                if not parcial:
                    print(f"   - Calidad del agua: {results['calidad_agua']}")

                for det in results.get('familias', []):
                    marca = " (provisional)" if det.get("procedencia") == "proximidad" else ""
                    print(f"     * {det['familia']}: BMWP {det['bmwp_individual']}{marca}")

                for aviso in results.get('advertencias', []):
                    print(f"    {aviso}")

        elif args.dataset_report:
            if not args.data_yaml:
                print("Error: --data-yaml es requerido para --dataset-report")
                sys.exit(1)
            DatasetReport(args.data_yaml).generate()
            print("\nReporte de dataset generado en: results/dataset_report/")

        elif args.model_report:
            if not args.model or not args.data_yaml:
                print("Error: --model y --data-yaml son requeridos para --model-report")
                sys.exit(1)
            ModelReport(args.model, args.data_yaml).generate()
            print("\nReporte de modelo generado en: results/model_report/")

        else:
            # Mostrar ayuda si no se especifican argumentos
            parser.print_help()

    except KeyboardInterrupt:
        print("\nOperación cancelada por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
