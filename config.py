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

# El .env vive junto al codigo, no en el directorio desde el que se invoca:
# con Path.cwd() la configuracion cambiaba en silencio segun donde se corriera.
load_dotenv(Path(__file__).resolve().parent / ".env")


class Config:
    """
    Clase de configuración centralizada.

    Maneja todas las variables de configuración del proyecto,
    incluyendo conexiones con Roboflow, parámetros de entrenamiento,
    configuración de inferencia y evaluación BMWP.
    """

    def __init__(self):
        """Inicializa la configuración con valores por defecto."""
        # Problemas detectados al leer el entorno; se muestran en __str__ en
        # vez de reventar, para que el usuario vea que valor quedo efectivo.
        self._config_warnings: list[str] = []
        # Configuración de Roboflow (opcional a este nivel: solo se exige
        # cuando algo la usa de verdad, ver DatasetManager.setup_roboflow_connection)
        self.roboflow_api_key: str = self._get_env_var("ROBOFLOW_API_KEY", default="")
        self.roboflow_workspace: str = self._get_env_var(
            "ROBOFLOW_WORKSPACE", default="pinv011159"
        )
        # El proyecto por defecto es el del split corregido. El original
        # (macroinvertebrados-acuaticos) reparte las rafagas del mismo
        # especimen entre train y test; ver docs/leakage_analysis.md.
        self.roboflow_project: str = self._get_env_var(
            "ROBOFLOW_PROJECT", default="macroinvertebrados-split-limpio-kuhq3"
        )

        # Configuración del modelo
        self.model_name: str = self._get_env_var("MODEL_NAME", default="yolo11s.pt")
        self.experiment_name: str = self._get_env_var("EXPERIMENT_NAME", default="macros")
        self.training_epochs: int = self._get_int("TRAINING_EPOCHS", 200)
        self.img_size: int = self._get_int("IMG_SIZE", 640)
        self.batch_size: int = self._get_int("BATCH_SIZE", 16)
        self.workers: int = self._get_int("WORKERS", 8)
        self.seed: int = self._get_int("SEED", 42)
        self.attention_reg_lambda: float = self._get_float("ATTENTION_REG_LAMBDA", 0.0)

        # Configuración de inferencia
        self.confidence_threshold: float = self._get_float("CONFIDENCE_THRESHOLD", 0.3)
        self.iou_threshold: float = self._get_float("IOU_THRESHOLD", 0.6)

        # Configuración de logging
        self.log_level: str = self._get_env_var("LOG_LEVEL", default="INFO")
        # Configuración BMWP
        self.enable_bmwp: bool = (
            self._get_env_var("ENABLE_BMWP", default="True").lower() == "true"
        )

    def _get_int(self, name: str, default: int) -> int:
        """Entero de entorno, tolerante a valores vacios o mal formados.

        `int(os.getenv(...))` reventaba en el constructor con `IMG_SIZE=`,
        antes de que `validate()` llegara a correr: el programa moria con un
        ValueError crudo en vez de avisar que la configuracion estaba mal.
        """
        crudo = self._get_env_var(name, default=str(default)).strip()
        try:
            return int(crudo)
        except ValueError:
            self._config_warnings.append(
                f"{name}={crudo!r} no es un entero; se usa {default}."
            )
            return default

    def _get_float(self, name: str, default: float) -> float:
        """Flotante de entorno, con la misma tolerancia que `_get_int`."""
        crudo = self._get_env_var(name, default=str(default)).strip()
        try:
            return float(crudo)
        except ValueError:
            self._config_warnings.append(
                f"{name}={crudo!r} no es un numero; se usa {default}."
            )
            return default

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

            # YOLO exige que el lado sea multiplo del stride maximo (32); con
            # 641 el modelo reescala en silencio y la resolucion efectiva deja
            # de ser la declarada.
            if self.img_size <= 0 or self.img_size % 32 != 0:
                return False

            if self.batch_size <= 0:
                return False

            if self.workers <= 0:
                return False

            return True

        except Exception:
            return False

    def validation_errors(self) -> list[str]:
        """Motivos concretos por los que `validate()` fallaria, mas los avisos
        acumulados al leer el entorno. Existe para que el usuario no reciba
        solo un booleano cuando la configuracion esta mal."""
        errores = list(self._config_warnings)
        if not (0.0 <= self.confidence_threshold <= 1.0):
            errores.append(f"CONFIDENCE_THRESHOLD={self.confidence_threshold} fuera de [0,1].")
        if not (0.0 <= self.iou_threshold <= 1.0):
            errores.append(f"IOU_THRESHOLD={self.iou_threshold} fuera de [0,1].")
        if self.training_epochs <= 0:
            errores.append(f"TRAINING_EPOCHS={self.training_epochs} debe ser > 0.")
        if self.img_size <= 0:
            errores.append(f"IMG_SIZE={self.img_size} debe ser > 0.")
        elif self.img_size % 32 != 0:
            errores.append(f"IMG_SIZE={self.img_size} debe ser multiplo de 32 (YOLO).")
        if self.batch_size <= 0:
            errores.append(f"BATCH_SIZE={self.batch_size} debe ser > 0.")
        if self.workers <= 0:
            errores.append(f"WORKERS={self.workers} debe ser > 0.")
        return errores

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
        }

    def get_bmwp_config(self) -> dict:
        """
        Obtiene la configuración BMWP.

        Returns:
            Diccionario con configuración BMWP
        """
        return {"enable_bmwp": self.enable_bmwp}

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

Logging:
  - Nivel: {self.log_level}

Validación: {'✓' if self.validate() else '✗'}
        """.strip()


# Instancia global de configuración
config = Config()
