"""
Sistema de logging para el proyecto YOLO Macroinvertebrados.

Este módulo proporciona un sistema de logging configurado y consistente
para todo el proyecto.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

from config import config


def setup_logger(name: str = "yolo_macro_detect",
                log_file: str | None = None,
                level: str | None = None) -> logging.Logger:
    """
    Configura y retorna un logger personalizado para el proyecto.

    Args:
        name: Nombre del logger
        log_file: Ruta al archivo de log (opcional)
        level: Nivel de logging (opcional, usa config por defecto)

    Returns:
        Logger configurado
    """
    # Obtener nivel de logging
    log_level = level or config.log_level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Crear logger
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)

    # Evitar duplicar handlers
    if logger.handlers:
        return logger

    # Crear formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Handler para consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Handler para archivo (si se especifica)
    if log_file:
        # Crear directorio de logs si no existe
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_training_logger(experiment_name: str) -> logging.Logger:
    """
    Obtiene un logger específico para entrenamiento.

    Args:
        experiment_name: Nombre del experimento

    Returns:
        Logger configurado para entrenamiento
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/training_{experiment_name}_{timestamp}.log"

    return setup_logger(
        name=f"training_{experiment_name}",
        log_file=log_file
    )


def get_inference_logger() -> logging.Logger:
    """
    Obtiene un logger específico para inferencia.

    Returns:
        Logger configurado para inferencia
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/inference_{timestamp}.log"

    return setup_logger(
        name="inference",
        log_file=log_file
    )


# Logger principal del proyecto
logger = setup_logger()
