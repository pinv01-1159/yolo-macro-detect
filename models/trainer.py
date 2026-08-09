"""
Módulo de entrenamiento para modelos YOLO de macroinvertebrados.

Este módulo maneja todo el proceso de entrenamiento de modelos YOLO,
incluyendo la configuración, entrenamiento y evaluación.
"""

import os
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO

from config import config
from utils.logger import get_training_logger
from utils.validators import validate_data_yaml_path, validate_directory_path


class YOLOTrainer:
    """
    Clase para manejar el entrenamiento de modelos YOLO.

    Esta clase encapsula toda la lógica de entrenamiento, incluyendo
    configuración, entrenamiento, evaluación y guardado de resultados.
    """

    def __init__(self, experiment_name: str | None = None):
        """
        Inicializa el entrenador YOLO.

        Args:
            experiment_name: Nombre del experimento (usa config por defecto)
        """
        self.experiment_name = experiment_name or config.experiment_name
        self.logger = get_training_logger(self.experiment_name)
        self.model: YOLO | None = None
        self.training_results = None
        self.eval_metrics: Any = None

        # Crear directorios necesarios
        self._setup_directories()

    def _setup_directories(self):
        """Configura los directorios necesarios para el entrenamiento."""
        directories = [
            "logs",
            "models",
            "results",
            "datasets"
        ]

        for directory in directories:
            validate_directory_path(directory, create=True)

    def load_model(self, model_name: str | None = None) -> YOLO:
        """
        Carga el modelo YOLO base.

        Args:
            model_name: Nombre del modelo base (usa config por defecto)

        Returns:
            Modelo YOLO cargado
        """
        model_name = model_name or config.model_name
        self.logger.info(f"Cargando modelo base: {model_name}")

        try:
            self.model = YOLO(model_name)
            self.logger.info(f"✅ Modelo {model_name} cargado exitosamente")
            return self.model
        except Exception as e:
            self.logger.error(f"❌ Error al cargar el modelo {model_name}: {e}")
            raise

    def validate_dataset(self, data_yaml_path: str) -> dict[str, Any]:
        """
        Valida el dataset antes del entrenamiento.

        Args:
            data_yaml_path: Ruta al archivo data.yaml

        Returns:
            Información del dataset validado
        """
        self.logger.info(f"Validando dataset: {data_yaml_path}")

        try:
            # Validar archivo data.yaml
            validate_data_yaml_path(data_yaml_path)

            # Cargar información del dataset
            with open(data_yaml_path, encoding='utf-8') as f:
                dataset_info = yaml.safe_load(f)

            # Validar rutas de datos directamente en la estructura del dataset
            # Ignorar las rutas del data.yaml y buscar en la estructura real
            dataset_dir = Path(data_yaml_path).parent

            # Mapeo de claves del YAML a nombres de carpetas reales
            split_mapping = {
                'train': 'train',
                'val': 'valid'  # La clave es 'val' pero la carpeta es 'valid'
            }

            for split_key, folder_name in split_mapping.items():
                # Buscar directamente en la estructura del dataset
                split_images_path = dataset_dir / folder_name / "images"
                if not split_images_path.exists():
                    raise FileNotFoundError(
                        f"Ruta de {split_key} no encontrada: {split_images_path}"
                    )

            self.logger.info("✅ Dataset validado:")
            self.logger.info(f"   - Clases: {dataset_info.get('nc', 'N/A')}")
            self.logger.info(f"   - Nombres: {dataset_info.get('names', [])}")
            self.logger.info(f"   - Train: {dataset_info.get('train', 'N/A')}")
            self.logger.info(f"   - Val: {dataset_info.get('val', 'N/A')}")

            return dataset_info

        except Exception as e:
            self.logger.error(f"❌ Error al validar el dataset: {e}")
            raise

    def train(self,
              data_yaml_path: str,
              epochs: int | None = None,
              img_size: int | None = None,
              batch_size: int | None = None,
              workers: int | None = None,
              seed: int | None = None,
              **kwargs) -> str:
        """
        Entrena el modelo YOLO.

        Args:
            data_yaml_path: Ruta al archivo data.yaml
            epochs: Número de épocas (usa config por defecto)
            img_size: Tamaño de imagen (usa config por defecto)
            batch_size: Tamaño del batch (usa config por defecto)
            workers: Número de workers (usa config por defecto)
            seed: Semilla de reproducibilidad (usa config por defecto)
            **kwargs: Parámetros adicionales para el entrenamiento

        Returns:
            Ruta al mejor modelo entrenado
        """
        # Usar valores de configuración por defecto
        epochs = epochs or config.training_epochs
        img_size = img_size or config.img_size
        batch_size = batch_size or config.batch_size
        workers = workers or config.workers
        seed = seed if seed is not None else config.seed

        self.logger.info("🚀 Iniciando entrenamiento del modelo")
        self.logger.info(f"   - Experimento: {self.experiment_name}")
        self.logger.info(f"   - Épocas: {epochs}")
        self.logger.info(f"   - Tamaño imagen: {img_size}")
        self.logger.info(f"   - Batch size: {batch_size}")
        self.logger.info(f"   - Workers: {workers}")
        self.logger.info(f"   - Semilla: {seed}")

        try:
            # Validar dataset
            self.validate_dataset(data_yaml_path)

            # Fail-fast: evaluate() usa por defecto split="test" (no "val")
            # para obtener una métrica final no sesgada, ya que val se usa
            # para seleccionar los pesos de best.pt durante el entrenamiento.
            # Si el split de test no existe, es mejor fallar ahora que
            # gastar horas de cómputo entrenando y recién enterarse al
            # llamar a evaluate().
            dataset_dir = Path(data_yaml_path).parent
            test_images_path = dataset_dir / "test" / "images"
            if not test_images_path.exists():
                raise FileNotFoundError(
                    f"Split de 'test' no encontrado: {test_images_path}. "
                    "evaluate() evalúa por defecto en split='test' (no 'val') "
                    "para reportar una métrica final no sesgada, así que el "
                    "dataset debe incluir una carpeta test/images antes de "
                    "iniciar el entrenamiento."
                )

            # Cargar modelo si no está cargado
            if self.model is None:
                self.load_model()
            assert self.model is not None

            # Configurar parámetros de entrenamiento.
            #
            # Los valores de augmentación no son los de fábrica: el dataset es
            # chico (~1.7k imágenes de entrenamiento) y cada familia se fotografió
            # en pocas sesiones, así que fondo/iluminación están correlacionados
            # con la clase. La augmentación fuerte de color, geometría y mosaico
            # rompe ese atajo y obliga al modelo a mirar la morfología del bicho.
            # Todo es sobreescribible vía **kwargs.
            train_kwargs = {
                'data': data_yaml_path,
                'epochs': epochs,
                'imgsz': img_size,
                'batch': batch_size,
                'workers': workers,
                'name': self.experiment_name,
                'seed': seed,
                'deterministic': True,
                'save': True,
                'save_period': 10,  # Guardar cada 10 épocas
                'patience': 30,     # Early stopping (dataset chico: converge lento)
                'cos_lr': True,
                'amp': True,        # mixed precision: entra en 6 GB de VRAM
                # --- augmentación anti-atajo ---
                'hsv_h': 0.02, 'hsv_s': 0.8, 'hsv_v': 0.5,  # color/iluminación de sesión
                'degrees': 20.0,    # el espécimen en bandeja no tiene "arriba"
                'translate': 0.15, 'scale': 0.6, 'shear': 5.0,
                'fliplr': 0.5, 'flipud': 0.5,
                'mosaic': 1.0, 'close_mosaic': 15,
                'mixup': 0.1,
                'erasing': 0.4,     # ocluye partes: penaliza memorizar la textura del fondo
                'verbose': True,
                **kwargs
            }

            # Iniciar entrenamiento
            self.logger.info("🏋️ Iniciando entrenamiento...")
            self.training_results = self.model.train(**train_kwargs)

            # Obtener ruta del mejor modelo
            best_model_path = f"runs/detect/{self.experiment_name}/weights/best.pt"

            if os.path.exists(best_model_path):
                self.logger.info("✅ Entrenamiento completado exitosamente")
                self.logger.info(f"   - Mejor modelo: {best_model_path}")
                return best_model_path
            else:
                raise FileNotFoundError(f"Modelo entrenado no encontrado: {best_model_path}")

        except Exception as e:
            self.logger.error(f"❌ Error durante el entrenamiento: {e}")
            raise

    def evaluate(self,
                model_path: str,
                data_yaml_path: str,
                split: str = "test",
                conf_threshold: float | None = None,
                iou_threshold: float | None = None) -> Any:
        """
        Evalúa el modelo entrenado en un split held-out.

        Por defecto evalúa en 'test', no en 'val': el split de validación ya
        se usó para elegir los pesos de best.pt durante el entrenamiento, así
        que reportarlo como métrica final estaría sesgado.

        Args:
            model_path: Ruta al modelo entrenado
            data_yaml_path: Ruta al archivo data.yaml
            split: Split del dataset a evaluar ('test', 'val' o 'train')
            conf_threshold: Umbral de confianza (usa config por defecto)
            iou_threshold: Umbral de IoU (usa config por defecto)

        Returns:
            Objeto de métricas de Ultralytics (DetMetrics)
        """
        conf_threshold = conf_threshold or config.confidence_threshold
        iou_threshold = iou_threshold or config.iou_threshold

        self.logger.info(f"📊 Evaluando rendimiento del modelo (split: {split})")
        self.logger.info(f"   - Modelo: {model_path}")
        self.logger.info(f"   - Umbral confianza: {conf_threshold}")
        self.logger.info(f"   - Umbral IoU: {iou_threshold}")

        try:
            eval_model = YOLO(model_path)

            metrics = eval_model.val(
                data=data_yaml_path,
                conf=conf_threshold,
                iou=iou_threshold,
                split=split,
                plots=True,
                verbose=True
            )
            self.eval_metrics = metrics

            self.logger.info("✅ Evaluación completada")
            self.logger.info(f"   - mAP50: {metrics.box.map50:.4f}")
            self.logger.info(f"   - mAP50-95: {metrics.box.map:.4f}")
            self.logger.info(f"   - Precision: {metrics.box.mp:.4f}")
            self.logger.info(f"   - Recall: {metrics.box.mr:.4f}")

            return metrics

        except Exception as e:
            self.logger.error(f"❌ Error durante la evaluación: {e}")
            raise

    def get_training_summary(self) -> dict[str, Any]:
        """
        Obtiene un resumen del entrenamiento.

        Prioriza las métricas de evaluate() (evaluación explícita, por
        defecto en el split de test) sobre las métricas internas del
        entrenamiento, que corresponden al split de validación usado para
        seleccionar los pesos.

        Returns:
            Resumen del entrenamiento
        """
        if self.eval_metrics is not None:
            box = self.eval_metrics.box
            return {
                "experiment_name": self.experiment_name,
                "model_path": f"runs/detect/{self.experiment_name}/weights/best.pt",
                "evaluated_split": "test",
                "metrics": {
                    "map50": float(box.map50),
                    "map50_95": float(box.map),
                    "precision": float(box.mp),
                    "recall": float(box.mr),
                },
            }

        if self.training_results is None:
            return {"error": "No hay resultados de entrenamiento disponibles"}

        try:
            metrics = self.training_results.results_dict

            return {
                "experiment_name": self.experiment_name,
                "model_path": f"runs/detect/{self.experiment_name}/weights/best.pt",
                "evaluated_split": "val (entrenamiento, sin evaluate() explícito)",
                "metrics": {
                    "map50": metrics.get("metrics/mAP50(B)", 0),
                    "map50_95": metrics.get("metrics/mAP50-95(B)", 0),
                    "precision": metrics.get("metrics/precision(B)", 0),
                    "recall": metrics.get("metrics/recall(B)", 0),
                },
                "training_info": {
                    "epochs": metrics.get("epoch", 0),
                    "total_time": metrics.get("train/epoch", 0),
                },
            }
        except Exception as e:
            self.logger.error(f"Error al obtener resumen: {e}")
            return {"error": str(e)}

    def save_training_config(self, output_path: str = "results"):
        """
        Guarda la configuración del entrenamiento.

        Args:
            output_path: Directorio donde guardar la configuración
        """
        config_data = {
            "experiment_name": self.experiment_name,
            "model_name": config.model_name,
            "training_epochs": config.training_epochs,
            "img_size": config.img_size,
            "batch_size": config.batch_size,
            "workers": config.workers,
            "confidence_threshold": config.confidence_threshold,
            "iou_threshold": config.iou_threshold,
        }

        config_file = Path(output_path) / f"{self.experiment_name}_config.yaml"

        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)

            self.logger.info(f"✅ Configuración guardada en: {config_file}")

        except Exception as e:
            self.logger.error(f"❌ Error al guardar configuración: {e}")
            raise
