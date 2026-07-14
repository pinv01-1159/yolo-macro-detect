"""
Configuración centralizada para el proyecto YOLO Macroinvertebrados.

Este módulo maneja toda la configuración del proyecto, incluyendo
variables de entorno, validaciones y valores por defecto.

Autor: Kevin Galeano
Proyecto: PINV01-1159
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path.cwd() / ".env")


class Config:
    """
    Clase de configuración centralizada.

    Maneja todas las variables de configuración del proyecto,
    incluyendo conexiones con Roboflow, parámetros de entrenamiento,
    configuración de inferencia y evaluación BMWP.
    """

    def __init__(self):
        """Inicializa la configuración con valores por defecto."""
        # Configuración de Roboflow (opcional a este nivel: solo se exige
        # cuando algo la usa de verdad, ver DatasetManager.setup_roboflow_connection)
        self.roboflow_api_key: str = self._get_env_var("ROBOFLOW_API_KEY", default="")
        self.roboflow_workspace: str = self._get_env_var(
            "ROBOFLOW_WORKSPACE", default="pinv011159"
        )
        self.roboflow_project: str = self._get_env_var(
            "ROBOFLOW_PROJECT", default="macroinvertebrados-acuaticos"
        )

        # Configuración del modelo
        self.model_name: str = self._get_env_var("MODEL_NAME", default="yolov8x.pt")
        self.experiment_name: str = self._get_env_var("EXPERIMENT_NAME", default="macros")
        self.training_epochs: int = int(self._get_env_var("TRAINING_EPOCHS", default="50"))
        self.img_size: int = int(self._get_env_var("IMG_SIZE", default="640"))
        self.batch_size: int = int(self._get_env_var("BATCH_SIZE", default="16"))
        self.workers: int = int(self._get_env_var("WORKERS", default="8"))
        self.seed: int = int(self._get_env_var("SEED", default="42"))

        # Configuración de inferencia
        self.confidence_threshold: float = float(
            self._get_env_var("CONFIDENCE_THRESHOLD", default="0.3")
        )
        self.iou_threshold: float = float(self._get_env_var("IOU_THRESHOLD", default="0.6"))

        # Configuración de logging
        self.log_level: str = self._get_env_var("LOG_LEVEL", default="INFO")
        self.save_results: bool = (
            self._get_env_var("SAVE_RESULTS", default="True").lower() == "true"
        )

        # Configuración BMWP
        self.enable_bmwp: bool = (
            self._get_env_var("ENABLE_BMWP", default="True").lower() == "true"
        )
        self.bmwp_confidence_weight: bool = (
            self._get_env_var("BMWP_CONFIDENCE_WEIGHT", default="True").lower() == "true"
        )

    def _get_env_var(self, name: str, default: str | None = None) -> str:
        """
        Obtiene una variable de entorno.

        Args:
            name: Nombre de la variable
            default: Valor por defecto

        Returns:
            Valor de la variable de entorno
        """
        return os.getenv(name, default) or ""

    def validate(self) -> bool:
        """
        Valida la configuración.

        La presencia de ROBOFLOW_API_KEY no se valida acá: no toda
        operación del pipeline la necesita (p. ej. --predict). La exige
        DatasetManager.setup_roboflow_connection() cuando corresponde.

        Returns:
            True si la configuración es válida
        """
        try:
            if not (0.0 <= self.confidence_threshold <= 1.0):
                return False

            if not (0.0 <= self.iou_threshold <= 1.0):
                return False

            if self.training_epochs <= 0:
                return False

            if self.img_size <= 0:
                return False

            if self.batch_size <= 0:
                return False

            if self.workers <= 0:
                return False

            return True

        except Exception:
            return False

    def get_roboflow_config(self) -> dict:
        """
        Obtiene la configuración de Roboflow.

        Returns:
            Diccionario con configuración de Roboflow
        """
        return {
            "api_key": self.roboflow_api_key,
            "workspace": self.roboflow_workspace,
            "project": self.roboflow_project
        }

    def get_training_config(self) -> dict:
        """
        Obtiene la configuración de entrenamiento.

        Returns:
            Diccionario con configuración de entrenamiento
        """
        return {
            "model_name": self.model_name,
            "experiment_name": self.experiment_name,
            "epochs": self.training_epochs,
            "img_size": self.img_size,
            "batch_size": self.batch_size,
            "workers": self.workers,
            "seed": self.seed,
        }

    def get_inference_config(self) -> dict:
        """
        Obtiene la configuración de inferencia.

        Returns:
            Diccionario con configuración de inferencia
        """
        return {
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "img_size": self.img_size,
            "enable_bmwp": self.enable_bmwp,
            "bmwp_confidence_weight": self.bmwp_confidence_weight
        }

    def get_bmwp_config(self) -> dict:
        """
        Obtiene la configuración BMWP.

        Returns:
            Diccionario con configuración BMWP
        """
        return {
            "enable_bmwp": self.enable_bmwp,
            "bmwp_confidence_weight": self.bmwp_confidence_weight
        }

    def __str__(self) -> str:
        """
        Representación en string de la configuración.

        Returns:
            String con la configuración
        """
        return f"""
Configuración YOLO Macroinvertebrados:
=====================================

Roboflow:
  - API Key: {'✓' if self.roboflow_api_key else '✗'}
  - Workspace: {self.roboflow_workspace}
  - Project: {self.roboflow_project}

Modelo:
  - Nombre: {self.model_name}
  - Experimento: {self.experiment_name}
  - Épocas: {self.training_epochs}
  - Tamaño imagen: {self.img_size}
  - Batch size: {self.batch_size}
  - Workers: {self.workers}
  - Semilla: {self.seed}

Inferencia:
  - Umbral confianza: {self.confidence_threshold}
  - Umbral IoU: {self.iou_threshold}

BMWP:
  - Habilitado: {self.enable_bmwp}
  - Ponderación confianza: {self.bmwp_confidence_weight}

Logging:
  - Nivel: {self.log_level}
  - Guardar resultados: {self.save_results}

Validación: {'✓' if self.validate() else '✗'}
        """.strip()


# Instancia global de configuración
config = Config()
